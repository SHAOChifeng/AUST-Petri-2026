#!/usr/bin/env python3
"""
方案 A：灌溉调度优化
=====================
AI + CPN 决策支持 — 3块地 × 1台泵 × 6种灌溉顺序

核心机制：顺序喂入（sequential feeding）
  - 不在 Ready 中一次性放入所有地块
  - 而是按顺序逐个放入：先放 A → A 灌溉完成 → 泵归还 → 再放 B
  - 这样精确控制了灌溉顺序，后续操作（施肥/侦察/喷洒）并行进行

流程：
  1. LSTM 预测未来 7 天气温
  2. Hargreaves 蒸散方程 → 每块地灌溉时长
  3. 顺序喂入 CPN 仿真 × 6 种顺序
  4. 输出最优调度方案及节约时间

用法：python scheme_A.py
"""

import sys, os, json, math, itertools
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── cpnpy ───────────────────────────────────────────────────────────────
from cpnpy.cpn.cpn_imp import CPN, Place, Transition, Arc, Marking, EvaluationContext
from cpnpy.cpn.colorsets import EnumeratedColorSet, UnitColorSet

# ═══════════════════════════════════════════════════════════════════════
# 第一阶段：AI 感知 — LSTM 温度预测 + Hargreaves 蒸散
# ═══════════════════════════════════════════════════════════════════════

def load_lstm_prediction():
    """加载 LSTM 预测结果，选取一个夏季场景。"""
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "lstm_agri", "models"
    )
    preds_data = np.load(os.path.join(models_dir, "predictions.npz"), allow_pickle=True)
    targets = preds_data["targets"]  # (1588, 7, 6)
    preds = preds_data["preds"]      # (1588, 7, 6)
    test_dates = preds_data["T_test"]  # (1588,)

    avg_temp = targets[:, :, 0].mean(axis=1)
    summer_idx = np.where(avg_temp > 25)[0]
    idx = summer_idx[len(summer_idx) // 3] if len(summer_idx) > 0 else -1

    T_pred = preds[idx, :, 0]  # 7-day predicted T2M (°C)
    T_true = targets[idx, :, 0]  # 7-day actual T2M (°C)
    date = test_dates[idx]

    return T_pred, T_true, date


def hargreaves_et(T_mean, T_max=None, T_min=None, Ra=35.0):
    """Hargreaves-Samani 蒸散方程 (1985)"""
    if T_max is None:
        T_max = T_mean + 5
    if T_min is None:
        T_min = T_mean - 5
    T_diff = max(T_max - T_min, 0.1)
    T_adj = max(T_mean + 17.8, 0.1)
    return 0.0023 * Ra * math.sqrt(T_diff) * T_adj


def compute_irrigation_durations(T_pred_7day):
    """LSTM 温度 → 蒸散量 → 每块地灌溉时长"""
    T_mean = T_pred_7day.mean()
    T_max = T_mean + 5.0
    T_min = T_mean - 5.0

    ET_current = hargreaves_et(T_mean, T_max, T_min)
    ET_baseline = hargreaves_et(20.0, 25.0, 15.0)
    ET_ratio = ET_current / ET_baseline

    base_per_ha = 15.0  # 1 ha 在基准温度下需要 15 min
    areas = {"A": 5.0, "B": 3.0, "C": 2.0}

    durations = {}
    for field, area in areas.items():
        durations[field] = max(round(base_per_ha * area * ET_ratio), 15)

    return durations, {
        "T_mean": T_mean, "T_max": T_max, "T_min": T_min,
        "ET_current": ET_current, "ET_baseline": ET_baseline,
        "ET_ratio": ET_ratio, "base_per_ha": base_per_ha, "areas": areas,
    }


# ═══════════════════════════════════════════════════════════════════════
# 第二阶段：顺序喂入 CPN 仿真
# ═══════════════════════════════════════════════════════════════════════

def simulate_sequential_feeding(order, durations):
    """
    顺序喂入仿真：按给定顺序逐个放入地块，精确控制灌溉顺序。

    机制：
      1. 将第一个地块放入 Ready + 泵可用
      2. 运行 CPN 直到该地块灌溉完成（泵归还）
      3. 将下一个地块放入 Ready
      4. 重复直到所有地块完成全部操作

    这确保了灌溉顺序严格按 order 执行，而后续操作（施肥/侦察/喷洒）
    可以在不同地块间并行进行。
    """
    FIELD_CS = EnumeratedColorSet(["A", "B", "C"], timed=True, name="FIELD")
    UNIT_CS = UnitColorSet(timed=True, name="E")

    # 构建单地块 CPN（所有地块经过相同的操作序列）
    p_ready = Place("Ready", FIELD_CS)
    p_pump = Place("Pump", UNIT_CS)
    p_irrigated = Place("Irrigated", FIELD_CS)
    p_fertilized = Place("Fertilized", FIELD_CS)
    p_scouted = Place("Scouted", FIELD_CS)
    p_done = Place("Done", FIELD_CS)

    t_irrigate = Transition("Irrigate", variables=["field", "e"], transition_delay=0)
    t_fertilize = Transition("Fertilize", variables=["field"], transition_delay=30)
    t_scout = Transition("Scout", variables=["field"], transition_delay=20)
    t_spray = Transition("Spray", variables=["field"], transition_delay=25)

    arcs = [
        Arc(p_ready, t_irrigate, "field"),
        Arc(p_pump, t_irrigate, "e"),
        Arc(t_irrigate, p_irrigated, "field"),
        Arc(t_irrigate, p_pump, "e"),
        Arc(p_irrigated, t_fertilize, "field"),
        Arc(t_fertilize, p_fertilized, "field"),
        Arc(p_fertilized, t_scout, "field"),
        Arc(t_scout, p_scouted, "field"),
        Arc(p_scouted, t_spray, "field"),
        Arc(t_spray, p_done, "field"),
    ]

    cpn = CPN()
    for p in [p_ready, p_pump, p_irrigated, p_fertilized, p_scouted, p_done]:
        cpn.add_place(p)
    for t in [t_irrigate, t_fertilize, t_scout, t_spray]:
        cpn.add_transition(t)
    for a in arcs:
        cpn.add_arc(a)

    mk = Marking()
    mk.set_tokens("Pump", ["e"])
    for pn in ["Ready", "Irrigated", "Fertilized", "Scouted", "Done"]:
        mk.set_tokens(pn, [])

    ctx = EvaluationContext()
    global_trace = []
    field_times = {}

    # 逐个喂入地块
    for field in order:
        # 设置灌溉时长
        t_irrigate.transition_delay = durations[field]

        # 放入地块
        mk.set_tokens("Ready", [field])

        # 运行该地块的灌溉
        field_start = mk.global_clock
        field_trace = _run_cpn_phase(cpn, mk, ctx, target_place="Irrigated",
                                     expected_field=field)
        global_trace.extend(field_trace)

    # 所有地块已灌溉，继续运行直到所有地块完成后续操作
    _run_cpn_until_done(cpn, mk, ctx, global_trace, expected_count=len(order))

    # 计算每块地的时间
    for e in global_trace:
        # 从快照信息推断属于哪个地块——简化：用变迁发生后 Irrigated 里的令牌
        pass

    return global_trace, mk.global_clock


def _run_cpn_phase(cpn, mk, ctx, target_place, expected_field):
    """运行 CPN 直到特定地块出现在目标库所。"""
    trace = []
    for _ in range(20):
        enabled = [t for t in cpn.transitions if cpn.is_enabled(t, mk, ctx)]
        if not enabled:
            old = mk.global_clock
            cpn.advance_global_clock(mk)
            if mk.global_clock == old:
                break
            enabled = [t for t in cpn.transitions if cpn.is_enabled(t, mk, ctx)]
            if not enabled:
                break

        t = enabled[0]
        cb = mk.global_clock
        cpn.fire_transition(t, mk, ctx)
        trace.append({
            "transition": t.name,
            "start": cb,
            "delay": t.transition_delay,
            "end": cb + t.transition_delay,
        })

        # 检查目标库所是否已包含期望的地块
        ms = mk.get_multiset(target_place)
        if ms and any(tok.value == expected_field for tok in ms.tokens):
            break

    return trace


def _run_cpn_until_done(cpn, mk, ctx, trace, expected_count):
    """运行 CPN 直到所有地块完成。"""
    for _ in range(50):
        enabled = [t for t in cpn.transitions if cpn.is_enabled(t, mk, ctx)]
        if not enabled:
            old = mk.global_clock
            cpn.advance_global_clock(mk)
            if mk.global_clock == old:
                break
            enabled = [t for t in cpn.transitions if cpn.is_enabled(t, mk, ctx)]
            if not enabled:
                break

        t = enabled[0]
        cb = mk.global_clock
        cpn.fire_transition(t, mk, ctx)
        trace.append({
            "transition": t.name,
            "start": cb,
            "delay": t.transition_delay,
            "end": cb + t.transition_delay,
        })

        # 检查是否所有地块都完成了
        done_ms = mk.get_multiset("Done")
        if len(done_ms.tokens) >= expected_count:
            break


# ═══════════════════════════════════════════════════════════════════════
# 第三阶段：枚举 → 仿真 → 对比
# ═══════════════════════════════════════════════════════════════════════

def run_all_orders(durations):
    """枚举 6 种灌溉顺序，运行仿真，排序输出。"""
    results = []
    for order in itertools.permutations(["A", "B", "C"]):
        trace, total = simulate_sequential_feeding(list(order), durations)
        results.append({
            "order": list(order),
            "total": total,
            "trace": trace,
            "n_steps": len(trace),
        })
    results.sort(key=lambda r: r["total"])
    return results


# ═══════════════════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════════════════

def print_results(results, weather_info, durations):
    print("=" * 78)
    print("  方案 A：灌溉调度优化 — AI + CPN 决策支持")
    print("=" * 78)

    print(f"\n  ── 天气条件 (LSTM 预测) ──")
    print(f"  日均温: {weather_info['T_mean']:.1f}°C  "
          f"(日较差 ±5°C → Tmax={weather_info['T_max']:.1f}°C, Tmin={weather_info['T_min']:.1f}°C)")
    print(f"  ET₀: {weather_info['ET_current']:.2f} mm/day  "
          f"(基准 20°C: {weather_info['ET_baseline']:.2f}, 比率 {weather_info['ET_ratio']:.2f}×)")

    print(f"\n  ── 灌溉时长 (AI + Hargreaves 推导) ──")
    for f in ["A", "B", "C"]:
        area = weather_info["areas"][f]
        print(f"  Field_{f}: {area:.0f} ha × {weather_info['base_per_ha']:.0f} min/ha "
              f"× {weather_info['ET_ratio']:.2f} ET 比 = {durations[f]} min")

    print(f"\n  ── 调度优化结果 (6 种灌溉顺序) ──")
    print(f"  {'排名':<4} {'顺序':<20} {'总时间':<12} {'步数':<6}")
    print(f"  {'─'*4} {'─'*20} {'─'*12} {'─'*6}")

    best = results[0]
    worst = results[-1]
    for i, r in enumerate(results):
        marker = "★" if i == 0 else ""
        order_str = " → ".join(r["order"])
        print(f"  {marker}{i+1:<3} {order_str:<20} "
              f"{r['total']} min ({r['total']/60:.1f}h)  "
              f"{r['n_steps']} 步")

    saving = worst["total"] - best["total"]
    print(f"\n  ── 优化效果 ──")
    print(f"  最优: {' → '.join(best['order'])} — {best['total']} min ({best['total']/60:.1f}h)")
    print(f"  最差: {' → '.join(worst['order'])} — {worst['total']} min ({worst['total']/60:.1f}h)")
    print(f"  节省: {saving} min ({saving/worst['total']*100:.1f}%)" if saving > 0 else
          f"  所有顺序总时间相同 — 因为后续操作可并行，瓶颈在灌溉总时长")

    print(f"\n  ── 可解释性 ──")
    print(f"  每一步执行轨迹都记录了哪个操作、何时开始、耗时多久。")
    print(f"  农学家可以：")
    print(f"    1. 检查每块地的灌溉等待时间")
    print(f"    2. 验证泵的利用率")
    print(f"    3. 做 what-if 分析（如：加一台泵能省多少时间？）")

    print(f"\n{'='*78}")
    print(f"  ✓ 方案 A 完成")
    print(f"{'='*78}")

    return best, worst, saving


def make_visualization(results, weather_info, durations, out_dir):
    """生成对比图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # ── Panel 1: 总完成时间对比 ──
    ax = axes[0, 0]
    orders_str = [" → ".join(r["order"]) for r in results]
    totals = [r["total"] for r in results]
    colors = ["#2E7D32" if i == 0 else "#E53935" if i == len(results)-1 else "#90A4AE"
              for i in range(len(results))]
    bars = ax.barh(range(len(results)), totals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(orders_str, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Total Completion Time (min)", fontsize=10)
    ax.set_title("Irrigation Order vs. Total Completion Time", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    for i, (bar, t) in enumerate(zip(bars, totals)):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f"{t} min ({t/60:.1f}h)", va="center", fontsize=8, color="#333")
    ax.set_xlim(0, max(totals) * 1.2)

    # ── Panel 2: 最优顺序执行轨迹 ──
    ax = axes[0, 1]
    best = results[0]
    op_colors = {"Irrigate": "#E53935", "Fertilize": "#43A047",
                 "Scout": "#1E88E5", "Spray": "#FB8C00"}
    # 从 trace 推断每个操作的归属
    trace = best["trace"]
    y_labels = ["Irrigate", "Fertilize", "Scout", "Spray"]
    y_pos = {op: 3 - i for i, op in enumerate(y_labels)}

    for e in trace:
        tn = e["transition"]
        if tn in op_colors:
            y = y_pos[tn]
            ax.barh(y, e["delay"], 0.5, left=e["start"],
                   color=op_colors[tn], edgecolor="white", linewidth=0.5)
            mid = e["start"] + e["delay"] / 2
            ax.text(mid, y, tn[:4], ha="center", va="center",
                   fontsize=6, fontweight="bold", color="white")

    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_title(f"Optimal Sequence: {' → '.join(best['order'])}", fontsize=11, fontweight="bold")

    # ── Panel 3: 灌溉等待时间 ──
    ax = axes[1, 0]
    # 从 trace 中提取灌溉事件
    irr_events = [(i, e) for i, e in enumerate(trace) if e["transition"] == "Irrigate"]
    fields_ordered = [f"Field_{chr(65+i)}" for i in range(len(irr_events))]
    wait_times = [e["start"] for _, e in irr_events]
    # 第一个灌溉等待 0（假设），后续等待 = 开始时间 - 前一个结束时间
    actual_wait = [0]
    for i in range(1, len(irr_events)):
        actual_wait.append(irr_events[i][1]["start"] - irr_events[i-1][1]["end"])

    ax.bar(range(len(irr_events)), actual_wait, color="#FFCDD2", edgecolor="#E53935")
    for i, w in enumerate(actual_wait):
        ax.text(i, w + 1, f"{w} min", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(irr_events)))
    ax.set_xticklabels([f"Field_{f}" for f in best["order"]], fontsize=10)
    ax.set_ylabel("Wait Time (min)", fontsize=10)
    ax.set_title("Pump Wait Time per Field", fontsize=11, fontweight="bold")

    # ── Panel 4: AI 输入 → CPN 参数 映射 ──
    ax = axes[1, 1]
    ax.axis("off")
    text = (
        f"AI → CPN Parameter Pipeline\n"
        f"{'─'*40}\n"
        f"LSTM Tmean:  {weather_info['T_mean']:.1f}°C\n"
        f"Hargreaves ET₀: {weather_info['ET_current']:.2f} mm/d\n"
        f"ET Ratio:     {weather_info['ET_ratio']:.2f}×\n\n"
        f"Field A (5ha): {durations['A']} min irrig.\n"
        f"Field B (3ha): {durations['B']} min irrig.\n"
        f"Field C (2ha): {durations['C']} min irrig.\n\n"
        f"{len(results)} orders simulated\n"
        f"Optimal: {'→'.join(results[0]['order'])}\n"
        f"  = {results[0]['total']} min ({results[0]['total']/60:.1f}h)\n\n"
        f"Interpretable: every step\n"
        f"  in execution trace is\n"
        f"  auditable by agronomists."
    )
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=9,
            fontfamily="monospace", va="top", linespacing=1.6,
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#CCC"))

    plt.tight_layout()
    out_path = os.path.join(out_dir, "scheme_A_results.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  图表已保存: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. AI 感知 ──
    print("Loading LSTM predictions...")
    T_pred, T_true, date = load_lstm_prediction()
    print(f"  Scenario date: {date}")
    print(f"  LSTM predicted 7-day T: {T_pred.round(1)}°C (MAE={np.abs(T_pred - T_true).mean():.2f}°C)")

    print("\nComputing irrigation durations...")
    durations, weather_info = compute_irrigation_durations(T_pred)
    for f in ["A", "B", "C"]:
        print(f"  Field_{f}: {weather_info['areas'][f]:.0f} ha → {durations[f]} min")

    # ── 2. 枚举仿真 ──
    print("\nEnumerating 6 irrigation orders with sequential feeding...")
    results = run_all_orders(durations)

    # ── 3. 输出 ──
    best, worst, saving = print_results(results, weather_info, durations)

    # ── 4. 可视化 ──
    chart_path = make_visualization(results, weather_info, durations, out_dir)

    # ── 5. 保存 ──
    result_path = os.path.join(out_dir, "scheme_A_results.txt")
    with open(result_path, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print_results(results, weather_info, durations)
        sys.stdout = old_stdout
    print(f"\n  结果已保存: {result_path}")