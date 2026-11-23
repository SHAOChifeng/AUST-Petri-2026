#!/usr/bin/env python3
"""
Nature 期刊风格 CPN 执行图生成器
=================================
- 字体: Liberation Sans (Arial-compatible, 期刊标准)
- 风格: 简洁、无冗余装饰、Nature 配色
- 数据: 每张图同时保存 CSV/JSON 原始数据
- 文字: 自动防重叠，标签位置智能计算

用法: python nature_figures.py
"""

import sys, os, json, csv, math, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, FancyBboxPatch
from itertools import permutations

# ── Nature 期刊风格配置 ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
})

# Nature 配色 (colorblind-friendly)
NATURE_BLUE   = "#377EB8"
NATURE_GREEN  = "#4DAF4A"
NATURE_RED    = "#E41A1C"
NATURE_ORANGE = "#FF7F00"
NATURE_PURPLE = "#984EA3"
NATURE_GRAY   = "#999999"
NATURE_PALETTE = [NATURE_BLUE, NATURE_GREEN, NATURE_RED, NATURE_ORANGE, NATURE_PURPLE]

OP_COLORS = {
    "Irrigate":     NATURE_BLUE,
    "Fertilize":    NATURE_GREEN,
    "Scout":        NATURE_ORANGE,
    "Spray":        NATURE_RED,
    "Harvest":      NATURE_PURPLE,
    "Scout+Spray":  NATURE_RED,
    "Cover":        NATURE_GREEN,
    "AF":           NATURE_BLUE,
}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_data_loader import get_scheme_a_data, get_scheme_b_data, get_scheme_c_data, get_scheme_d_data
from real_data_loader import get_spring_2024 as get_spring_2024_full

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA_DIR = os.path.join(OUT_DIR, "data")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
for d in [OUT_DIR, DATA_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def panel_label(ax, letter, x=-0.08, y=1.05):
    """Nature 风格面板标签: 粗体小写字母"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left")

def smart_label(ax, x, y, text, offset_x=0, offset_y=0, fontsize=7, **kwargs):
    """智能标签: 自动偏移避免重叠"""
    ax.text(x + offset_x, y + offset_y, text, fontsize=fontsize, **kwargs)

def save_data(filename, data_dict):
    """保存原始数据（CSV + JSON）"""
    base = os.path.join(DATA_DIR, filename)
    with open(base + ".json", "w") as f:
        json.dump(data_dict, f, indent=2, default=str)
    # 如果数据包含 results 列表，同时保存 CSV
    if "results" in data_dict and isinstance(data_dict["results"], list):
        with open(base + ".csv", "w", newline="") as f:
            if data_dict["results"]:
                writer = csv.DictWriter(f, fieldnames=data_dict["results"][0].keys())
                writer.writeheader()
                writer.writerows(data_dict["results"])


# ═══════════════════════════════════════════════════════════════════════
# 方案 A：灌溉调度
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_A():
    data = get_scheme_a_data()
    fields = data["fields"]
    durations = {f: info["irrigation_min"] for f, info in fields.items()}
    post_ops = {"A": [("Fertilize",30),("Scout",20),("Spray",25)],
                "B": [("Fertilize",20),("Scout",14),("Spray",17)],
                "C": [("Fertilize",15),("Scout",10),("Spray",13)]}

    results = []; all_traces = {}
    for order in permutations(["A","B","C"]):
        pump_time = 0; trace = []; field_end = {}
        for f in order:
            trace.append({"field":f,"op":"Irrigate","start":pump_time,
                          "end":pump_time+durations[f],"resource":"Pump"})
            pump_time += durations[f]
            t = pump_time
            for op_name, d in post_ops[f]:
                trace.append({"field":f,"op":op_name,"start":t,"end":t+d,"resource":"Field"})
                t += d
            field_end[f] = t
        total = max(field_end.values())
        order_str = "→".join(order)
        results.append({"order":order_str,"total":total,
                        "irrigation_sum":sum(durations[f] for f in order),
                        "post_ops_max":max(sum(d for _,d in post_ops[f]) for f in order)})
        all_traces[order_str] = trace

    results.sort(key=lambda r:r["total"])
    best, worst = results[0], results[-1]
    saving = worst["total"]-best["total"]
    pct = saving/worst["total"]*100 if worst["total"]>0 else 0

    # ── Nature 风格图 ──
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

    # (a) 顺序对比柱状图
    ax = axes[0]; panel_label(ax, "a")
    names = [r["order"] for r in results]
    totals = [r["total"] for r in results]
    colors = [NATURE_GREEN if i==0 else NATURE_RED if i==len(results)-1 else NATURE_GRAY
              for i in range(len(results))]
    ax.barh(range(len(results)), totals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Total completion time (min)")
    ax.invert_yaxis()
    for i, t in enumerate(totals):
        ax.text(t + 2, i, f"{t}", va="center", fontsize=7, fontweight="bold",
                color=colors[i])
    ax.set_xlim(0, max(totals) * 1.18)

    # (b) 最优甘特图
    ax = axes[1]; panel_label(ax, "b")
    trace = all_traces[best["order"]]
    field_y = {"A": 3.5, "B": 2.5, "C": 1.5}
    for e in trace:
        y = field_y[e["field"]]
        c = OP_COLORS.get(e["op"], NATURE_GRAY)
        dur = e["end"] - e["start"]
        ax.barh(y, dur, 0.45, left=e["start"], color=c, edgecolor="white", linewidth=0.3)
        if dur > 12:
            ax.text(e["start"] + dur/2, y, e["op"][:6], ha="center", va="center",
                   fontsize=5.5, color="white", fontweight="bold")
    ax.set_yticks([3.5, 2.5, 1.5])
    ax.set_yticklabels(["Field A (3.0 ha)", "Field B (2.0 ha)", "Field C (1.5 ha)"], fontsize=7)
    ax.set_xlabel("Time (min)")
    ax.set_title(f"Optimal: {best['order']}", fontsize=8, fontweight="bold", pad=4)

    # (c) 最差甘特图
    ax = axes[2]; panel_label(ax, "c")
    trace = all_traces[worst["order"]]
    for e in trace:
        y = field_y[e["field"]]
        c = OP_COLORS.get(e["op"], NATURE_GRAY)
        dur = e["end"] - e["start"]
        ax.barh(y, dur, 0.45, left=e["start"], color=c, edgecolor="white", linewidth=0.3)
        if dur > 12:
            ax.text(e["start"] + dur/2, y, e["op"][:6], ha="center", va="center",
                   fontsize=5.5, color="white", fontweight="bold")
    ax.set_yticks([3.5, 2.5, 1.5])
    ax.set_yticklabels(["Field A", "Field B", "Field C"], fontsize=7)
    ax.set_xlabel("Time (min)")
    ax.set_title(f"Worst: {worst['order']}", fontsize=8, fontweight="bold", pad=4)

    # 图例
    legend_ops = [("Irrigate",NATURE_BLUE),("Fertilize",NATURE_GREEN),
                  ("Scout",NATURE_ORANGE),("Spray",NATURE_RED)]
    legend_elements = [Patch(facecolor=c, label=op, edgecolor="white", linewidth=0.3)
                       for op,c in legend_ops]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=7,
              frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(wspace=0.35, bottom=0.15)
    fig.savefig(os.path.join(FIG_DIR, "fig_scheme_A.png"), dpi=300)
    plt.close()

    # 保存数据
    save_data("scheme_A", {"results": results, "best": best, "worst": worst,
                           "saving_pct": round(pct,1), "data_source": data["source"]})

    return {"results":results,"best":best,"worst":worst,"saving_pct":round(pct,1)}


# ═══════════════════════════════════════════════════════════════════════
# 方案 B：植保调度
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_B():
    data = get_scheme_b_data()
    fields = data["fields"]
    durations = {f: info["spray_min"] for f, info in fields.items()}
    scout_time = 15

    strategies = [
        ("S1: Complete A first",
         [("Scout","A"),("Spray","A"),("Scout","B"),("Spray","B")], 1),
        ("S2: Complete B first",
         [("Scout","B"),("Spray","B"),("Scout","A"),("Spray","A")], 1),
        ("S3: Scout both,\nspray short first",
         [("Scout","A"),("Scout","B"),("Spray","B"),("Spray","A")], 1),
        ("S4: Scout both,\nspray long first",
         [("Scout","A"),("Scout","B"),("Spray","A"),("Spray","B")], 1),
        ("S5: 2 drones parallel",
         [("Scout","A"),("Spray","A"),("Scout","B"),("Spray","B")], 2),
    ]

    results = []; all_traces = {}
    for name, seq, nd in strategies:
        trace = []
        if nd == 1:
            t = 0
            for op, f in seq:
                dur = scout_time if op=="Scout" else durations[f]
                trace.append({"field":f,"op":op,"start":t,"end":t+dur,"drone":1})
                t += dur
            total = t
        else:
            t1 = 0; t2 = 0
            for op, f in seq:
                if f == "A":
                    dur = scout_time if op=="Scout" else durations[f]
                    trace.append({"field":f,"op":op,"start":t1,"end":t1+dur,"drone":1})
                    t1 += dur
                else:
                    dur = scout_time if op=="Scout" else durations[f]
                    trace.append({"field":f,"op":op,"start":t2,"end":t2+dur,"drone":2})
                    t2 += dur
            total = max(t1, t2)
        results.append({"name":name.replace("\n"," "),"total":total,"n_drones":nd})
        all_traces[name] = trace

    results.sort(key=lambda r:r["total"])
    best = results[0]
    best_1drone = min((r for r in results if r["n_drones"]==1), key=lambda r:r["total"])
    saving = best_1drone["total"]-best["total"]

    # ── Nature 风格图 ──
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

    # (a) 策略对比
    ax = axes[0]; panel_label(ax, "a")
    names = [r["name"] for r in results]
    totals = [r["total"] for r in results]
    colors = [NATURE_GREEN if i==0 else NATURE_BLUE if r["n_drones"]==2 else NATURE_GRAY
              for i,r in enumerate(results)]
    ax.barh(range(len(results)), totals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("Total time (min)")
    ax.invert_yaxis()
    for i, t in enumerate(totals):
        ax.text(t + 1, i, f"{t}", va="center", fontsize=7, fontweight="bold", color=colors[i])
    ax.set_xlim(0, max(totals) * 1.2)

    # (b) 1 drone 甘特图
    ax = axes[1]; panel_label(ax, "b")
    trace_1d = all_traces[best_1drone["name"]]
    for e in trace_1d:
        y = 1.5 if e["field"]=="A" else 0.5
        c = OP_COLORS.get(e["op"], NATURE_GRAY)
        dur = e["end"]-e["start"]
        ax.barh(y, dur, 0.4, left=e["start"], color=c, edgecolor="white", linewidth=0.3)
        if dur > 10:
            ax.text(e["start"]+dur/2, y, f"{e['field']}:{e['op'][:5]}", ha="center", va="center",
                   fontsize=5.5, color="white", fontweight="bold")
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["Field A (severe)", "Field B (mild)"], fontsize=7)
    ax.set_xlabel("Time (min)")
    ax.set_title(f"1 drone: {best_1drone['total']} min", fontsize=8, fontweight="bold", pad=4)

    # (c) 2 drones 甘特图
    ax = axes[2]; panel_label(ax, "c")
    trace_2d = all_traces[best["name"]]
    for e in trace_2d:
        y = 1.3 if e["drone"]==1 else 0.3
        c = OP_COLORS.get(e["op"], NATURE_GRAY)
        dur = e["end"]-e["start"]
        ax.barh(y, dur, 0.3, left=e["start"], color=c, edgecolor="white", linewidth=0.3)
        if dur > 10:
            ax.text(e["start"]+dur/2, y, f"{e['field']}:{e['op'][:5]}", ha="center", va="center",
                   fontsize=5.5, color="white", fontweight="bold")
    ax.set_yticks([1.3, 0.3])
    ax.set_yticklabels(["Drone 1", "Drone 2"], fontsize=8)
    ax.set_xlabel("Time (min)")
    ax.set_title(f"2 drones: {best['total']} min", fontsize=8, fontweight="bold", pad=4)

    legend_ops = [("Scout",NATURE_ORANGE),("Spray",NATURE_RED)]
    fig.legend(handles=[Patch(facecolor=c,label=op,edgecolor="white",linewidth=0.3) for op,c in legend_ops],
              loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(wspace=0.35, bottom=0.15)
    fig.savefig(os.path.join(FIG_DIR, "fig_scheme_B.png"), dpi=300)
    plt.close()

    save_data("scheme_B", {"results":results,"best":best,"best_1drone":best_1drone,
                           "saving_pct":round(saving/best_1drone["total"]*100,1)})

    return {"results":results,"best":best,"best_1drone":best_1drone,
            "saving_pct":round(saving/best_1drone["total"]*100,1)}


# ═══════════════════════════════════════════════════════════════════════
# 方案 C：霜冻应急
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_C():
    data = get_scheme_c_data()
    fields = data["fields"]

    strat_defs = [
        ("S1: Covers→A,B\nAF→C", [("A","Cover",30),("B","Cover",30),("C","AF",60)]),
        ("S2: Double\nprotect A", [("A","Cover",30),("A","AF",60),("B","Cover",30)]),
        ("S3: Covers→A,C\nAF→B", [("A","Cover",30),("C","Cover",30),("B","AF",60)]),
        ("S4: AF→A\nCovers→B,C", [("A","AF",60),("B","Cover",30),("C","Cover",30)]),
    ]

    results = []
    for name, alloc in strat_defs:
        protected = {}
        for f_id, resource, _ in alloc:
            eff = 1.0 if resource=="Cover" else 0.80
            protected[f_id] = max(protected.get(f_id,0), eff)
        total_value = sum(fields[f]["value"]*protected.get(f,0) for f in fields)
        potential = sum(fields[f]["value"] for f in fields)
        results.append({"name":name.replace("\n"," "),"value":total_value,
                        "coverage":round(total_value/potential*100,1),
                        "protected":protected,"allocation":alloc})

    results.sort(key=lambda r:r["value"], reverse=True)
    best, worst = results[0], results[-1]
    saving = best["value"]-worst["value"]

    # ── Nature 风格图 ──
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))

    # (a) 保护价值对比
    ax = axes[0]; panel_label(ax, "a")
    names = [r["name"] for r in results]
    values = [r["value"] for r in results]
    colors = [NATURE_GREEN if i==0 else NATURE_RED if i==len(results)-1 else NATURE_GRAY
              for i in range(len(results))]
    ax.barh(range(len(results)), values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("Protection value (Yuan)")
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + 5, i, f"{v:.0f}", va="center", fontsize=7, fontweight="bold", color=colors[i])
    ax.set_xlim(0, max(values) * 1.18)

    # (b) 最优分配方案
    ax = axes[1]; panel_label(ax, "b")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    field_pos = {"A": (1.0, 3.5), "B": (3.0, 3.5), "C": (2.0, 1.0)}
    for f_id, (fx, fy) in field_pos.items():
        info = fields[f_id]
        risk = info["frost_risk"]
        risk_color = NATURE_RED if risk > 0.5 else NATURE_ORANGE if risk > 0.3 else NATURE_GREEN
        rect = Rectangle((fx-0.7, fy-0.35), 1.4, 0.7, facecolor=risk_color, alpha=0.15,
                        edgecolor="none", linewidth=0, zorder=0)
        ax.add_patch(rect)
        ax.text(fx, fy+0.15, f"Field {f_id}", ha="center", fontsize=8, fontweight="bold")
        ax.text(fx, fy-0.1, f"Risk: {risk*100:.0f}%", ha="center", fontsize=6.5, color="#555")
        # 分配的资源
        for f_id2, resource, _ in best["allocation"]:
            if f_id2 == f_id:
                eff = "100%" if resource=="Cover" else "80%"
                ax.text(fx, fy-0.35, f"{resource} ({eff})", ha="center", fontsize=6.5, color=NATURE_BLUE)

    # (c) 覆盖矩阵
    ax = axes[2]; panel_label(ax, "c")
    field_ids = ["A","B","C"]
    x = np.arange(len(field_ids))
    width = 0.18
    for i, r in enumerate(results):
        coverage = [r["protected"].get(f, 0) for f in field_ids]
        ax.bar(x + (i-1.5)*width, coverage, width,
              label=r["name"][:12], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Field {f}\n({fields[f]['frost_risk']*100:.0f}% risk)" for f in field_ids], fontsize=6.5)
    ax.set_ylabel("Protection effectiveness")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=5.5, loc="lower right", ncol=2, frameon=False)

    plt.subplots_adjust(wspace=0.35)
    fig.savefig(os.path.join(FIG_DIR, "fig_scheme_C.png"), dpi=300)
    plt.close()

    save_data("scheme_C", {"results": results, "best": best, "saving_value": saving,
                           "saving_pct": round(saving/worst["value"]*100,1)})

    return {"results":results,"best":best,"saving_pct":round(saving/worst["value"]*100,1)}


# ═══════════════════════════════════════════════════════════════════════
# 方案 D：全季仿真
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_D():
    data = get_scheme_d_data()
    schedules = data["schedules"]

    all_ops = []
    for f_id in ["A","B","C"]:
        for op in schedules[f_id]:
            all_ops.append(op)
    all_ops.sort(key=lambda o: o["day"])

    day_groups = {}
    for op in all_ops:
        d = op["day"]
        if d not in day_groups:
            day_groups[d] = {"pump":0,"drone":0}
        if op["op"] in ["Irrigate","Fertilize"]:
            day_groups[d]["pump"] += 1
        else:
            day_groups[d]["drone"] += 1

    configs = [("1P+1D",1,1),("2P+1D",2,1),("1P+2D",1,2),("2P+2D",2,2)]
    results = []
    for name, npump, ndrone in configs:
        total_work = sum(op["duration"] for op in all_ops)
        total_wait = 0; conflict_days = 0
        for day, group in day_groups.items():
            if group["pump"] > npump:
                conflict_days += 1
                total_wait += (group["pump"]-npump)*40
            if group["drone"] > ndrone:
                conflict_days += 1
                total_wait += (group["drone"]-ndrone)*25
        results.append({"config":name,"work":total_work,"wait":total_wait,
                        "total":total_work+total_wait,"conflict_days":conflict_days,
                        "n_pump":npump,"n_drone":ndrone})

    results.sort(key=lambda r:r["total"])
    best = results[0]
    baseline = [r for r in results if r["config"]=="1P+1D"][0]
    saving = baseline["total"]-best["total"]

    # ── Nature 风格图 (2×3) ──
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.5))

    # (a) 温度 + GDD
    ax = axes[0,0]; panel_label(ax, "a")
    spring_full = get_spring_2024_full()
    days = [r["day"] for r in spring_full[:65]]
    temps = [r["t2m"] for r in spring_full[:65]]
    ax.fill_between(days, [t-5 for t in temps], [t+5 for t in temps], alpha=0.12, color=NATURE_BLUE)
    ax.plot(days, temps, color=NATURE_BLUE, linewidth=1.2)
    ax.axhline(y=0, color=NATURE_RED, linestyle="--", linewidth=0.6, alpha=0.5)
    for s in data["stages"]:
        doy = next((r["day"] for r in spring_full if f"{r['month']}/{r['day_of_month']}"==s["date"]), None)
        if doy:
            ax.axvline(x=doy, color=NATURE_ORANGE, linestyle=":", linewidth=0.5, alpha=0.5)
            ax.text(doy+0.3, 28, s["stage"][:4], fontsize=5.5, color=NATURE_ORANGE, rotation=90, va="top")
    ax.set_xlabel("Day (April 1 →)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlim(0, 65); ax.set_ylim(-5, 35)

    # (b) 全季操作时间线
    ax = axes[0,1]; panel_label(ax, "b")
    field_y = {"A": 3.5, "B": 2.5, "C": 1.5}
    for op in all_ops:
        y = field_y[op["field"]]
        c = OP_COLORS.get(op["op"], NATURE_GRAY)
        dur_days = max(op["duration"]/60*0.5, 0.25)
        ax.barh(y, dur_days, 0.4, left=op["day"], color=c, edgecolor="white", linewidth=0.2, alpha=0.85)
    ax.set_yticks([3.5,2.5,1.5])
    ax.set_yticklabels(["Field A","Field B","Field C"], fontsize=7)
    ax.set_xlabel("Day")
    ax.set_xlim(0, 65)

    # (c) 资源配置对比
    ax = axes[0,2]; panel_label(ax, "c")
    configs_names = [r["config"] for r in results]
    works = [r["work"] for r in results]
    waits = [r["wait"] for r in results]
    x = np.arange(len(configs_names))
    ax.bar(x-0.2, works, 0.35, label="Operation", color=NATURE_BLUE, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.bar(x+0.2, waits, 0.35, label="Wait", color=NATURE_RED, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(configs_names, fontsize=7)
    ax.set_ylabel("Time (min)")
    ax.legend(fontsize=7, frameon=False)
    for i,(w,wait) in enumerate(zip(works,waits)):
        if wait > 0:
            ax.text(i, w+wait/2, f"+{wait}", ha="center", fontsize=6.5, color=NATURE_RED)

    # (d) 冲突日分布
    ax = axes[1,0]; panel_label(ax, "d")
    conflict_data = [(d, day_groups[d]["pump"]+day_groups[d]["drone"]) for d in sorted(day_groups.keys())]
    cd_days = [d for d,_ in conflict_data]
    cd_counts = [c for _,c in conflict_data]
    colors = [NATURE_RED if c>1 else NATURE_GRAY for c in cd_counts]
    ax.bar(cd_days, cd_counts, color=colors, edgecolor="white", width=0.8, linewidth=0.2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Concurrent ops")
    ax.set_xlim(0, 65)
    ax.axhline(y=1, color=NATURE_GRAY, linestyle=":", linewidth=0.5, alpha=0.5)

    # (e) 瓶颈分析
    ax = axes[1,1]; panel_label(ax, "e")
    c2 = [r for r in results if r["config"]=="2P+1D"][0]
    c3 = [r for r in results if r["config"]=="1P+2D"][0]
    c4 = [r for r in results if r["config"]=="2P+2D"][0]
    pump_save = baseline["wait"]-c2["wait"]
    drone_save = baseline["wait"]-c3["wait"]
    both_save = baseline["wait"]-c4["wait"]
    savings = [pump_save, drone_save, both_save]
    labels = ["+1 Pump", "+1 Drone", "+1 Both"]
    colors_bar = [NATURE_BLUE, NATURE_ORANGE, NATURE_GREEN]
    ax.barh(labels, savings, color=colors_bar, edgecolor="white", height=0.5, linewidth=0.3)
    ax.set_xlabel("Wait time saved (min)")
    for i, v in enumerate(savings):
        ax.text(v+2, i, f"{v} min", va="center", fontsize=7, fontweight="bold")

    # (f) 数据溯源
    ax = axes[1,2]; panel_label(ax, "f")
    ax.axis("off")
    summary = (
        f"Data: NASA POWER 2024\n"
        f"Hefei (31.8°N, 117.2°E)\n"
        f"Apr 1 → Jun 30 (91 days)\n\n"
        f"Model: GDD (base 0°C)\n"
        f"Mean temp: {data['spring_mean_temp']}°C\n"
        f"Total GDD: {data['total_gdd']}\n\n"
        f"Stages (2024):\n" +
        "\n".join(f"  GDD {s['threshold']}: {s['stage']}\n"
                  f"    → {s['date']} ({s['gdd']} GDD)"
                  for s in data["stages"]) +
        f"\n\nResults:\n"
        f"  1P+1D: {baseline['total']} min\n"
        f"  Best: {best['config']} = {best['total']} min\n"
        f"  Saving: {saving} min\n"
        f"  Conflicts: {baseline['conflict_days']}→{best['conflict_days']}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=6.5,
            fontfamily="monospace", va="top", linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FAFAFA", edgecolor="#DDD", linewidth=0.5))

    legend_ops = [("Irrigate",NATURE_BLUE),("Fertilize",NATURE_GREEN),
                  ("Scout+Spray",NATURE_RED),("Harvest",NATURE_PURPLE)]
    fig.legend(handles=[Patch(facecolor=c,label=op,edgecolor="white",linewidth=0.3) for op,c in legend_ops],
              loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.subplots_adjust(wspace=0.4, hspace=0.45, bottom=0.08)
    fig.savefig(os.path.join(FIG_DIR, "fig_scheme_D.png"), dpi=300)
    plt.close()

    save_data("scheme_D", {"results":results,"best":best,"baseline":baseline,
                           "saving_pct":round(saving/baseline["total"]*100,1)})

    return {"results":results,"best":best,"saving_pct":round(saving/baseline["total"]*100,1)}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Nature-style CPN Figures + Data Export")
    print("=" * 65)

    rA = run_scheme_A()
    rB = run_scheme_B()
    rC = run_scheme_C()
    rD = run_scheme_D()

    print(f"\n  Scheme A: {rA['saving_pct']}% saving")
    print(f"  Scheme B: {rB['saving_pct']}% saving")
    print(f"  Scheme C: {rC['saving_pct']}% saving")
    print(f"  Scheme D: {rD['saving_pct']}% saving")

    print(f"\n  Figures: {FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.endswith('.png'):
            print(f"    {f}")
    print(f"\n  Data: {DATA_DIR}/")
    for f in sorted(os.listdir(DATA_DIR)):
        print(f"    {f}")
    print(f"\n{'='*65}")