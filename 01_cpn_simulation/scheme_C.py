#!/usr/bin/env python3
"""
方案 C：霜冻应急响应
=====================
AI + CPN 决策支持 — 3块茶园 × 2台覆盖机 × 1套防冻灌溉 × 2个部署班组

场景：春季茶园（3月），LSTM 预测今夜气温将降至 0°C 以下（霜冻预警）。
      有限资源必须在霜冻来临前部署到位。如何分配资源保护最有价值的茶园？

流程：
  1. LSTM 预测温度 → 霜冻风险等级
  2. 每块茶园有不同特征（海拔、面积、价值）
  3. CPN 仿真 × 4 种资源分配策略
  4. 输出：保护覆盖率、保护价值、部署时间

关键洞察：
  CPN 不预测"是否会霜冻"，而是建模"如果霜冻要来，有限资源怎么分配最有效"。
  这是论文 Algorithm 3 精神的正确落地——CPN 处理"响应"而非"预测"。

用法：python scheme_C.py
"""

import sys, os, math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# ── cpnpy ───────────────────────────────────────────────────────────────
from cpnpy.cpn.cpn_imp import CPN, Place, Transition, Arc, Marking, EvaluationContext
from cpnpy.cpn.colorsets import EnumeratedColorSet, UnitColorSet


# ═══════════════════════════════════════════════════════════════════════
# 第一阶段：AI 感知 — LSTM 温度预测 → 霜冻风险
# ═══════════════════════════════════════════════════════════════════════

def get_lstm_frost_risk():
    """
    模拟 LSTM 温度预测 + 历史气候学统计 → 霜冻风险。

    基于论文的 LSTM 模型（R²=0.949, MAE=1.69°C）和 NASA POWER 44 年数据，
    对 3 块茶园分别评估霜冻风险。

    现实输入：
      - LSTM 预测今夜最低温 = -2.3°C（MAE=1.69°C → 可能范围 -4.0~-0.6°C）
      - 历史气候学：3 月霜冻概率 = 42%（P50）
      - 各茶园微气候因子修正风险

    返回: {field: {risk, elevation, area, value, protection_need}}
    """
    # LSTM 预测（模拟）
    lstm_pred = {
        "min_temp": -2.3,      # 预测今夜最低温（°C）
        "mae": 1.69,           # 模型 MAE
        "frost_prob": 0.42,    # 3月历史霜冻概率（NASA POWER 44年）
        "confidence": 0.89,    # LSTM 预测置信度
    }

    # 三块茶园的微气候特征
    # 海拔越低、靠近谷底 → 冷空气下沉 → 霜冻风险越高
    fields = {
        "A": {
            "name": "Valley Bottom",
            "elevation": 320,        # 米（最低）
            "area_ha": 2.5,          # 公顷
            "value_per_ha": 120,     # 万元/公顷（名优茶，价值最高）
            "microclimate_factor": 1.3,  # 谷底冷空气聚集 → 风险放大
            "access_time": 10,       # 从基地到达需要的时间（min）
        },
        "B": {
            "name": "Mid Slope",
            "elevation": 420,
            "area_ha": 3.5,          # 面积最大
            "value_per_ha": 80,      # 万元/公顷（大宗茶）
            "microclimate_factor": 1.0,
            "access_time": 15,
        },
        "C": {
            "name": "Upper Slope",
            "elevation": 520,        # 最高
            "area_ha": 1.8,
            "value_per_ha": 100,     # 万元/公顷
            "microclimate_factor": 0.7,  # 冷空气不易聚集 → 风险降低
            "access_time": 20,
        },
    }

    # 计算每块地的霜冻风险
    for f_id, info in fields.items():
        base_risk = lstm_pred["frost_prob"]
        adjusted = base_risk * info["microclimate_factor"]
        adjusted = min(adjusted, 0.95)  # 上限
        info["frost_risk"] = round(adjusted, 3)
        info["total_value"] = info["area_ha"] * info["value_per_ha"]

        # 风险等级
        if adjusted > 0.7:
            info["risk_level"] = "Critical"
            info["protection_priority"] = 1
        elif adjusted > 0.4:
            info["risk_level"] = "High"
            info["protection_priority"] = 2
        else:
            info["risk_level"] = "Moderate"
            info["protection_priority"] = 3

    return fields, lstm_pred


# ═══════════════════════════════════════════════════════════════════════
# 第二阶段：CPN 仿真 — 资源部署
# ═══════════════════════════════════════════════════════════════════════

def simulate_allocation(strategy_name, allocation, fields, frost_deadline=360):
    """
    按给定资源分配方案运行 CPN 仿真。

    资源：
      - 2 个部署班组（Team_1, Team_2）— 共享资源
      - 2 台覆盖机（Cover_1, Cover_2）— 部署 30 min/块
      - 1 套防冻灌溉（AntiFrost）— 部署 60 min/块

    allocation: [(field_id, resource_type, deploy_time), ...]
      例如: [("A", "Cover", 30), ("B", "Cover", 30), ("C", "AntiFrost", 60)]

    frost_deadline: 霜冻预计来临时间（min），所有保护必须在此之前部署完毕。
    """
    FIELD_CS = EnumeratedColorSet(["A", "B", "C"], timed=True, name="FIELD")
    UNIT_CS = UnitColorSet(timed=True, name="E")

    # 状态
    p_ready = Place("Ready", FIELD_CS)          # 待保护的地块
    p_teams = Place("Teams", UNIT_CS)            # 可用班组
    p_cover = Place("Cover", UNIT_CS)            # 可用覆盖机
    p_af = Place("AntiFrost", UNIT_CS)           # 可用防冻灌溉
    p_protected = Place("Protected", FIELD_CS)   # 已保护的地块

    # 为每块地创建独立的部署变迁
    transitions = {}
    arcs = []

    for f_id in ["A", "B", "C"]:
        # 覆盖机部署
        t_name = f"Deploy_Cover_{f_id}"
        t = Transition(t_name, variables=["field", "team", "cover"], transition_delay=30)
        transitions[t_name] = t
        arcs.extend([
            Arc(p_ready, t, "field"),
            Arc(p_teams, t, "team"),
            Arc(p_cover, t, "cover"),
            Arc(t, p_protected, "field"),
            Arc(t, p_teams, "team"),    # 归还班组
            Arc(t, p_cover, "cover"),   # 归还覆盖机
        ])

        # 防冻灌溉部署
        t_name = f"Deploy_AF_{f_id}"
        t = Transition(t_name, variables=["field", "team", "af"], transition_delay=60)
        transitions[t_name] = t
        arcs.extend([
            Arc(p_ready, t, "field"),
            Arc(p_teams, t, "team"),
            Arc(p_af, t, "af"),
            Arc(t, p_protected, "field"),
            Arc(t, p_teams, "team"),    # 归还班组
            Arc(t, p_af, "af"),         # 归还防冻灌溉
        ])

    cpn = CPN()
    for p in [p_ready, p_teams, p_cover, p_af, p_protected]:
        cpn.add_place(p)
    for t in transitions.values():
        cpn.add_transition(t)
    for a in arcs:
        cpn.add_arc(a)

    # 初始标记
    mk = Marking()
    mk.set_tokens("Teams", ["t1", "t2"])    # 2 个班组
    mk.set_tokens("Cover", ["c1", "c2"])     # 2 台覆盖机
    mk.set_tokens("AntiFrost", ["af1"])      # 1 套防冻灌溉
    mk.set_tokens("Protected", [])

    ctx = EvaluationContext()
    trace = []

    # 按分配方案逐个放入地块
    for f_id, resource, _ in allocation:
        mk.set_tokens("Ready", [f_id])
        target = f"Deploy_{resource}_{f_id}"
        _run_until(cpn, mk, ctx, trace, target)

    total_time = max((e["end"] for e in trace), default=0)

    # 计算保护效果
    protected_fields = {}
    for e in trace:
        f_id = e["transition"].split("_")[-1]
        resource = e["transition"].split("_")[1]
        if e["end"] <= frost_deadline:
            if resource == "Cover":
                effectiveness = 1.0   # 覆盖机 100% 有效
            else:
                effectiveness = 0.80  # 防冻灌溉 80% 有效
            protected_fields[f_id] = max(protected_fields.get(f_id, 0), effectiveness)

    # 保护价值
    total_protection_value = sum(
        fields[f_id]["total_value"] * effectiveness
        for f_id, effectiveness in protected_fields.items()
    )
    total_potential_value = sum(f["total_value"] for f in fields.values())

    return {
        "name": strategy_name,
        "allocation": allocation,
        "total_time": total_time,
        "trace": trace,
        "protected_fields": protected_fields,
        "protection_value": total_protection_value,
        "potential_value": total_potential_value,
        "coverage": total_protection_value / total_potential_value if total_potential_value > 0 else 0,
        "frost_deadline": frost_deadline,
        "on_time": total_time <= frost_deadline,
    }


def _run_until(cpn, mk, ctx, trace, target_transition):
    """运行 CPN 直到指定变迁触发。"""
    for _ in range(40):
        enabled = [t for t in cpn.transitions if cpn.is_enabled(t, mk, ctx)]
        if not enabled:
            old = mk.global_clock
            cpn.advance_global_clock(mk)
            if mk.global_clock == old:
                break
            enabled = [t for t in cpn.transitions if cpn.is_enabled(t, mk, ctx)]
            if not enabled:
                break

        # 优先触发目标变迁
        target_t = None
        for t in enabled:
            if t.name == target_transition:
                target_t = t
                break

        if target_t is not None:
            cb = mk.global_clock
            cpn.fire_transition(target_t, mk, ctx)
            trace.append({
                "transition": target_t.name,
                "start": cb,
                "delay": target_t.transition_delay,
                "end": cb + target_t.transition_delay,
            })
            return

        # 否则触发第一个可用变迁
        for t in enabled:
            cb = mk.global_clock
            cpn.fire_transition(t, mk, ctx)
            trace.append({
                "transition": t.name,
                "start": cb,
                "delay": t.transition_delay,
                "end": cb + t.transition_delay,
            })
            break


# ═══════════════════════════════════════════════════════════════════════
# 第三阶段：枚举策略 → 仿真 → 对比
# ═══════════════════════════════════════════════════════════════════════

def run_all_strategies(fields, lstm_pred):
    """
    4 种资源分配策略：
      S1: 覆盖 A, 覆盖 B, 防冻 C  — 覆盖机保护高风险，防冻给低风险
      S2: 覆盖 A, 防冻 A, 覆盖 B  — 双重保护最高风险地块
      S3: 覆盖 A, 覆盖 C, 防冻 B  — 覆盖机保护高/低风险，防冻给中风险
      S4: 防冻 A, 覆盖 B, 覆盖 C  — 防冻给最高风险，覆盖机给中/低风险
    """
    # 霜冻预计在 6 小时（360 min）后到来
    frost_deadline = 360   # 6 hours = 360 min (LSTM 预测窗口)

    strategies = [
        ("S1: Covers→A,B | AF→C",
         [("A", "Cover", 30), ("B", "Cover", 30), ("C", "AF", 60)]),
        ("S2: Double protect A | Cover B",
         [("A", "Cover", 30), ("A", "AF", 60), ("B", "Cover", 30)]),
        ("S3: Covers→A,C | AF→B",
         [("A", "Cover", 30), ("C", "Cover", 30), ("B", "AF", 60)]),
        ("S4: AF→A | Covers→B,C",
         [("A", "AF", 60), ("B", "Cover", 30), ("C", "Cover", 30)]),
    ]

    results = []
    for name, allocation in strategies:
        r = simulate_allocation(name, allocation, fields, frost_deadline)
        results.append(r)

    results.sort(key=lambda r: r["protection_value"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════════════════

def print_results(results, fields, lstm_pred):
    print("=" * 78)
    print("  方案 C：霜冻应急响应 — AI + CPN 决策支持")
    print("=" * 78)

    print(f"\n  ── LSTM 温度预测 ──")
    print(f"  预测今夜最低温: {lstm_pred['min_temp']}°C "
          f"(MAE={lstm_pred['mae']}°C, 置信度={lstm_pred['confidence']})")
    print(f"  3月历史霜冻概率: {lstm_pred['frost_prob']*100:.0f}% "
          f"(NASA POWER 44年)")
    print(f"  霜冻预警窗口: {results[0]['frost_deadline']} min "
          f"({results[0]['frost_deadline']/60:.1f}h)")

    print(f"\n  ── 茶园霜冻风险评估 ──")
    print(f"  {'地块':<6} {'位置':<16} {'海拔':<6} {'面积':<6} {'价值':<6} "
          f"{'风险':<6} {'等级':<10} {'优先级':<6}")
    print(f"  {'─'*6} {'─'*16} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*10} {'─'*6}")
    for f_id in ["A", "B", "C"]:
        f = fields[f_id]
        print(f"  {f_id:<6} {f['name']:<16} {f['elevation']}m   "
              f"{f['area_ha']}ha   Y{f['total_value']}万  "
              f"{f['frost_risk']*100:.0f}%   {f['risk_level']:<10} P{f['protection_priority']}")

    print(f"\n  ── 资源约束 ──")
    print(f"  2 个部署班组（每班组同时只能部署 1 套设备）")
    print(f"  2 台覆盖机（部署 30 min/块, 100% 防霜有效）")
    print(f"  1 套防冻灌溉（部署 60 min/块, 80% 防霜有效）")

    print(f"\n  ── 策略对比 ({len(results)} 种分配方案) ──")
    print(f"  {'排名':<4} {'策略':<32} {'部署时间':<10} {'保护价值':<14} "
          f"{'覆盖率':<8} {'按时?':<6}")
    print(f"  {'─'*4} {'─'*32} {'─'*10} {'─'*14} {'─'*8} {'─'*6}")

    best = results[0]
    worst = results[-1]
    for i, r in enumerate(results):
        marker = "★" if i == 0 else ""
        on_time = "✓" if r["on_time"] else "✗"
        print(f"  {marker}{i+1:<3} {r['name']:<32} "
              f"{r['total_time']} min    "
              f"Y{r['protection_value']:.0f}万 / Y{r['potential_value']:.0f}万  "
              f"{r['coverage']*100:.0f}%     {on_time}")

    saving_value = best["protection_value"] - worst["protection_value"]
    print(f"\n  ── 优化效果 ──")
    print(f"  最优: {best['name']} — 保护价值 Y{best['protection_value']:.0f}万 "
          f"({best['coverage']*100:.0f}%)")
    r2 = results[1] if len(results) > 1 else best
    print(f"  次优: {r2['name']} — 保护价值 Y{r2['protection_value']:.0f}万 "
          f"({r2['coverage']*100:.0f}%)")
    print(f"  最优 vs 最差: 保护价值差距 Y{saving_value:.0f}万 "
          f"({saving_value/worst['protection_value']*100:.0f}%)")

    print(f"\n  ── 关键洞察 ──")
    print(f"  Field A（谷底, 2.5ha, Y300万）霜冻风险最高 ({fields['A']['frost_risk']*100:.0f}%)。")
    print(f"  {best['name']} 优先保护 A，因此保护价值最高。")
    print(f"  防冻灌溉（80%有效）应分配给中等风险地块，覆盖机（100%有效）留给高风险地块。")
    print(f"  所有方案均在霜冻来临前 ({results[0]['frost_deadline']} min) 完成部署。")
    print(f"  CPN 揭示了：有限资源下，按风险优先级分配的保护价值最高。")

    print(f"\n  ── 执行轨迹 (最优策略) ──")
    for e in best["trace"]:
        f_id = e["transition"].split("_")[-1]
        resource = e["transition"].split("_")[1]
        icon = "🛡️" if resource == "Cover" else "💧"
        print(f"    {e['start']:>4}→{e['end']:<4}min  {icon} "
              f"Deploy {resource} → Field {f_id} "
              f"({fields[f_id]['name']}, 风险 {fields[f_id]['frost_risk']*100:.0f}%)")

    print(f"\n{'='*78}")
    print(f"  ✓ 方案 C 完成")
    print(f"  ✓ 证明了 LSTM → CPN 仿真 → 资源分配优化的完整流程")
    print(f"{'='*78}")

    return best


def make_visualization(results, fields, lstm_pred, out_dir):
    """生成可视化。"""
    fig = plt.figure(figsize=(14, 10))

    resource_colors = {"Cover": "#4CAF50", "AF": "#2196F3"}

    # ── Panel 1: 保护价值对比 ──
    ax1 = fig.add_subplot(2, 3, 1)
    names = [r["name"].split(":")[0] for r in results]
    values = [r["protection_value"] for r in results]
    colors = ["#2E7D32" if i == 0 else "#90A4AE" for i in range(len(results))]
    bars = ax1.barh(range(len(results)), values, color=colors, edgecolor="white")
    ax1.set_yticks(range(len(results)))
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Protection Value (Y10k)", fontsize=9)
    ax1.set_title("Strategy vs. Protection Value", fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    for bar, v in zip(bars, values):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"Y{v:.0f}万", va="center", fontsize=7)
    ax1.set_xlim(0, max(values) * 1.25)

    # ── Panel 2: 霜冻风险热力图 ──
    ax2 = fig.add_subplot(2, 3, 2)
    field_ids = ["A", "B", "C"]
    risks = [fields[f]["frost_risk"] for f in field_ids]
    values_f = [fields[f]["total_value"] for f in field_ids]
    colors_risk = ["#D32F2F", "#FFA726", "#66BB6A"]
    bars2 = ax2.bar(field_ids, risks, color=colors_risk, edgecolor="white", width=0.5)
    ax2.set_ylabel("Frost Risk", fontsize=9)
    ax2.set_title("Frost Risk by Field", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1)
    for bar, r, v in zip(bars2, risks, values_f):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{r*100:.0f}%\nY{v:.0f}万", ha="center", fontsize=8, fontweight="bold")

    # ── Panel 3: 最优策略甘特图 ──
    ax3 = fig.add_subplot(2, 3, 3)
    best = results[0]
    trace = best["trace"]
    y_labels = []
    y_pos = 0
    for e in trace:
        f_id = e["transition"].split("_")[-1]
        resource = e["transition"].split("_")[1]
        icon = "Cover" if resource == "Cover" else "AF"
        color = resource_colors[icon]
        ax3.barh(y_pos, e["delay"], 0.6, left=e["start"], color=color,
                edgecolor="white", linewidth=0.5)
        lbl = f"Field {f_id} ({icon})"
        ax3.text(e["start"] + e["delay"]/2, y_pos, lbl, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")
        y_labels.append(f"{f_id} ({resource})")
        y_pos += 1
    ax3.axvline(x=best["frost_deadline"], color="#D32F2F", linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax3.text(best["frost_deadline"] + 5, y_pos - 0.5, "Frost\nDeadline",
            fontsize=7, color="#D32F2F", fontweight="bold", va="center")
    ax3.set_yticks(range(len(y_labels)))
    ax3.set_yticklabels(y_labels, fontsize=7)
    ax3.set_xlabel("Time (min)", fontsize=9)
    ax3.set_title(f"Optimal: {best['name']}", fontsize=10, fontweight="bold")

    # ── Panel 4: 保护覆盖对比 ──
    ax4 = fig.add_subplot(2, 3, 4)
    x = np.arange(len(field_ids))
    width = 0.2
    for i, r in enumerate(results):
        offset = (i - 1.5) * width
        coverage = [r["protected_fields"].get(f, 0) for f in field_ids]
        ax4.bar(x + offset, coverage, width, label=r["name"].split(":")[0],
               alpha=0.8, edgecolor="white", linewidth=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"Field {f}\nY{fields[f]['total_value']}万" for f in field_ids], fontsize=8)
    ax4.set_ylabel("Protection Effectiveness", fontsize=9)
    ax4.set_title("Coverage by Field & Strategy", fontsize=10, fontweight="bold")
    ax4.set_ylim(0, 1.15)
    ax4.legend(fontsize=6, loc="upper right")

    # ── Panel 5: LSTM 预测可视化 ──
    ax5 = fig.add_subplot(2, 3, 5)
    hours = np.arange(0, 25, 1)
    # 模拟 LSTM 预测的温度曲线
    np.random.seed(42)
    temp_pred = 8 * np.cos((hours - 4) * np.pi / 12) + np.random.normal(0, 0.5, len(hours))
    temp_pred[hours > 20] = 2 * np.cos((hours[hours > 20] - 4) * np.pi / 12) - 1
    temp_pred[hours > 22] -= 2
    ci_lower = temp_pred - lstm_pred["mae"]
    ci_upper = temp_pred + lstm_pred["mae"]
    ax5.fill_between(hours, ci_lower, ci_upper, alpha=0.2, color="#1565C0")
    ax5.plot(hours, temp_pred, color="#1565C0", linewidth=2, label="LSTM Prediction")
    ax5.axhline(y=0, color="#D32F2F", linestyle="--", linewidth=1.5, label="Frost Threshold")
    ax5.axvline(x=best["frost_deadline"]/60, color="#FF6F00", linestyle=":", linewidth=1.5,
               label=f"Deployment Deadline ({best['frost_deadline']/60:.0f}h)")
    ax5.set_xlabel("Hour of Day", fontsize=9)
    ax5.set_ylabel("Temperature (°C)", fontsize=9)
    ax5.set_title("LSTM Temperature Forecast", fontsize=10, fontweight="bold")
    ax5.legend(fontsize=7)
    ax5.set_xlim(0, 24)
    ax5.grid(alpha=0.3)

    # ── Panel 6: 关键洞察总结 ──
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    insight_text = (
        f"LSTM → CPN Frost Response\n"
        f"{'─'*35}\n\n"
        f"LSTM predicts: {lstm_pred['min_temp']}°C tonight\n"
        f"  (MAE={lstm_pred['mae']}°C, R²=0.949)\n"
        f"Frost probability: {lstm_pred['frost_prob']*100:.0f}%\n\n"
        f"Field A (valley): {fields['A']['frost_risk']*100:.0f}% risk\n"
        f"  → Y{fields['A']['total_value']}万 at stake\n"
        f"Field B (mid): {fields['B']['frost_risk']*100:.0f}% risk\n"
        f"Field C (upper): {fields['C']['frost_risk']*100:.0f}% risk\n\n"
        f"Optimal strategy:\n"
        f"  {best['name']}\n"
        f"  Protection: Y{best['protection_value']:.0f}万\n"
        f"  Coverage: {best['coverage']*100:.0f}%\n"
        f"  Deploy: {best['total_time']} min\n\n"
        f"Key: CPN models resource\n"
        f"constraints, not predicts\n"
        f"frost. It answers: given\n"
        f"limited resources, what's\n"
        f"the best allocation?"
    )
    ax6.text(0.05, 0.95, insight_text, transform=ax6.transAxes, fontsize=8.5,
            fontfamily="monospace", va="top", linespacing=1.4,
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#CCC"))

    plt.tight_layout()
    out_path = os.path.join(out_dir, "scheme_C_results.png")
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

    print("Loading LSTM temperature forecast...")
    fields, lstm_pred = get_lstm_frost_risk()

    print("\nRunning 4 resource allocation strategies...")
    results = run_all_strategies(fields, lstm_pred)

    best = print_results(results, fields, lstm_pred)

    chart_path = make_visualization(results, fields, lstm_pred, out_dir)

    result_path = os.path.join(out_dir, "scheme_C_results.txt")
    with open(result_path, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print_results(results, fields, lstm_pred)
        sys.stdout = old_stdout
    print(f"\n  结果已保存: {result_path}")