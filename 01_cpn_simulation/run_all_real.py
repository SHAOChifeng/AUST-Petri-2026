#!/usr/bin/env python3
"""
全方案真实数据验证 + CPN执行图 — 一键运行 A+B+C+D

用法：python run_all_real.py
输出：文本结果 + 每方案2-3张CPN执行图 (Gantt/对比/瓶颈)
"""

import sys, os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_data_loader import get_scheme_a_data, get_scheme_b_data, get_scheme_c_data, get_scheme_d_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

OP_COLORS = {"Irrigate": "#2196F3", "Fertilize": "#4CAF50", "Scout": "#FFC107",
             "Spray": "#FF9800", "Harvest": "#F44336", "Cover": "#4CAF50",
             "AF": "#2196F3", "Scout+Spray": "#FF9800"}


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

    results = []
    all_traces = {}
    for order in permutations(["A","B","C"]):
        pump_time = 0
        trace = []
        field_end = {}
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
        results.append({"order":order_str,"total":total})
        all_traces[order_str] = trace

    results.sort(key=lambda r:r["total"])
    best, worst = results[0], results[-1]
    saving = worst["total"]-best["total"]
    pct = saving/worst["total"]*100 if worst["total"]>0 else 0

    # ── 可视化 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: 所有顺序对比
    ax = axes[0]
    names = [r["order"] for r in results]
    totals = [r["total"] for r in results]
    colors = ["#2E7D32" if i==0 else "#E53935" if i==len(results)-1 else "#90A4AE" for i in range(len(results))]
    ax.barh(range(len(results)), totals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Total Completion Time (min)", fontsize=10)
    ax.set_title("Irrigation order vs. completion time", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    for i,t in enumerate(totals):
        ax.text(t+2, i, f"{t} min", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, max(totals)*1.15)

    # Panel 2: 最优顺序甘特图
    ax = axes[1]
    trace = all_traces[best["order"]]
    field_y = {"A":3.5, "B":2.5, "C":1.5}
    for e in trace:
        y = field_y[e["field"]]
        c = OP_COLORS.get(e["op"], "#999")
        ax.barh(y, e["end"]-e["start"], 0.4, left=e["start"], color=c, edgecolor="white", linewidth=0.5)
        mid = e["start"]+(e["end"]-e["start"])/2
        ax.text(mid, y, e["op"][:4], ha="center", va="center", fontsize=6, fontweight="bold", color="white")
    ax.set_yticks([3.5,2.5,1.5])
    ax.set_yticklabels(["Field A (3ha)","Field B (2ha)","Field C (1.5ha)"], fontsize=8)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_title(f"Best: {best['order']} ({best['total']} min)", fontsize=10, fontweight="bold")

    # Panel 3: 最差顺序甘特图
    ax = axes[2]
    trace = all_traces[worst["order"]]
    for e in trace:
        y = field_y[e["field"]]
        c = OP_COLORS.get(e["op"], "#999")
        ax.barh(y, e["end"]-e["start"], 0.4, left=e["start"], color=c, edgecolor="white", linewidth=0.5)
        mid = e["start"]+(e["end"]-e["start"])/2
        ax.text(mid, y, e["op"][:4], ha="center", va="center", fontsize=6, fontweight="bold", color="white")
    ax.set_yticks([3.5,2.5,1.5])
    ax.set_yticklabels(["Field A","Field B","Field C"], fontsize=8)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_title(f"Worst: {worst['order']} ({worst['total']} min)", fontsize=10, fontweight="bold")

    legend = [Patch(facecolor=c, label=op) for op,c in OP_COLORS.items() if op in ["Irrigate","Fertilize","Scout","Spray"]]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scheme_A_cpn_gantt.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return {"data":data,"results":results,"best":best,"worst":worst,
            "saving_min":saving,"saving_pct":round(pct,1),"traces":all_traces}


# ═══════════════════════════════════════════════════════════════════════
# 方案 B：植保调度
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_B():
    data = get_scheme_b_data()
    fields = data["fields"]
    durations = {f:info["spray_min"] for f,info in fields.items()}
    scout_time = 15

    strategies = [
        ("S1: A first", [("Scout","A"),("Spray","A"),("Scout","B"),("Spray","B")], 1),
        ("S2: B first", [("Scout","B"),("Spray","B"),("Scout","A"),("Spray","A")], 1),
        ("S3: Scout both,\nspray short first", [("Scout","A"),("Scout","B"),("Spray","B"),("Spray","A")], 1),
        ("S4: Scout both,\nspray long first", [("Scout","A"),("Scout","B"),("Spray","A"),("Spray","B")], 1),
        ("S5: 2 drones\nparallel", [("Scout","A"),("Spray","A"),("Scout","B"),("Spray","B")], 2),
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
        results.append({"name":name,"total":total,"n_drones":nd})
        all_traces[name] = trace

    results.sort(key=lambda r:r["total"])
    best = results[0]
    best_1drone = min((r for r in results if r["n_drones"]==1), key=lambda r:r["total"])
    saving = best_1drone["total"]-best["total"]

    # ── 可视化 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: 策略对比
    ax = axes[0]
    names = [r["name"].replace("\n"," ") for r in results]
    totals = [r["total"] for r in results]
    colors = ["#2E7D32" if i==0 else "#1565C0" if r["n_drones"]==2 else "#90A4AE" for i,r in enumerate(results)]
    ax.barh(range(len(results)), totals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Total Time (min)", fontsize=10)
    ax.set_title("Strategy vs. time", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    for i,t in enumerate(totals):
        ax.text(t+1, i, f"{t} min", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, max(totals)*1.2)

    # Panel 2: 1 drone 甘特图
    ax = axes[1]
    trace_1d = all_traces[best_1drone["name"]]
    field_y = {"A":1.5, "B":0.5}
    for e in trace_1d:
        y = field_y[e["field"]]
        c = OP_COLORS.get(e["op"], "#999")
        ax.barh(y, e["end"]-e["start"], 0.4, left=e["start"], color=c, edgecolor="white", linewidth=0.5)
        ax.text(e["start"]+(e["end"]-e["start"])/2, y, e["op"][:5], ha="center", va="center",
               fontsize=6, fontweight="bold", color="white")
    ax.set_yticks([1.5,0.5])
    ax.set_yticklabels(["Field A (severe)","Field B (mild)"], fontsize=8)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_title(f"1 Drone: {best_1drone['total']} min", fontsize=10, fontweight="bold")

    # Panel 3: 2 drones 甘特图
    ax = axes[2]
    trace_2d = all_traces[best["name"]]
    drone_y = {1:1.3, 2:0.3}
    for e in trace_2d:
        y = drone_y[e["drone"]]
        c = OP_COLORS.get(e["op"], "#999")
        ax.barh(y, e["end"]-e["start"], 0.3, left=e["start"], color=c, edgecolor="white", linewidth=0.5)
        ax.text(e["start"]+(e["end"]-e["start"])/2, y, f"{e['field']}:{e['op'][:5]}", ha="center", va="center",
               fontsize=6, fontweight="bold", color="white")
    ax.set_yticks([1.3,0.3])
    ax.set_yticklabels(["Drone 1","Drone 2"], fontsize=9)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_title(f"2 Drones: {best['total']} min ({saving} min saved)", fontsize=10, fontweight="bold")

    legend = [Patch(facecolor=c, label=op) for op,c in OP_COLORS.items() if op in ["Scout","Spray"]]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scheme_B_cpn_gantt.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return {"data":data,"results":results,"best":best,"best_1drone":best_1drone,
            "saving_min":saving,"saving_pct":round(saving/best_1drone["total"]*100,1) if best_1drone["total"]>0 else 0}


# ═══════════════════════════════════════════════════════════════════════
# 方案 C：霜冻应急
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_C():
    data = get_scheme_c_data()
    fields = data["fields"]

    strat_defs = [
        ("S1: Covers→A,B\nAF→C", [("A","Cover",30),("B","Cover",30),("C","AF",60)]),
        ("S2: Double protect A\nCover B", [("A","Cover",30),("A","AF",60),("B","Cover",30)]),
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
        results.append({"name":name,"protected":protected,"value":total_value,
                        "potential":potential,"coverage":total_value/potential,
                        "allocation":alloc})

    results.sort(key=lambda r:r["value"], reverse=True)
    best, worst = results[0], results[-1]
    saving = best["value"]-worst["value"]

    # ── 可视化 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel 1: 保护价值对比
    ax = axes[0]
    names = [r["name"].replace("\n"," ") for r in results]
    values = [r["value"] for r in results]
    colors = ["#2E7D32" if i==0 else "#E53935" if i==len(results)-1 else "#90A4AE" for i in range(len(results))]
    ax.barh(range(len(results)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Protection Value (Yuan)", fontsize=10)
    ax.set_title("Protection value by strategy", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    for i,v in enumerate(values):
        ax.text(v+5, i, f"Y{v:.0f}", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, max(values)*1.2)

    # Panel 2: 资源分配图 (最优)
    ax = axes[1]
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title(f"Best: {best['name'].split(chr(10))[0]}", fontsize=10, fontweight="bold")
    # 画三个地块
    field_pos = {"A":(1.0,3.5), "B":(3.0,3.5), "C":(2.0,1.0)}
    for f_id, (fx, fy) in field_pos.items():
        info = fields[f_id]
        risk_color = "#D32F2F" if info["frost_risk"]>0.5 else "#FFA726" if info["frost_risk"]>0.3 else "#66BB6A"
        rect = Rectangle((fx-0.8, fy-0.4), 1.6, 0.8, facecolor=risk_color, alpha=0.3, edgecolor="#333", linewidth=2)
        ax.add_patch(rect)
        ax.text(fx, fy+0.2, f"Field {f_id}", ha="center", fontsize=9, fontweight="bold")
        ax.text(fx, fy-0.15, f"Risk {info['frost_risk']*100:.0f}%  Y{info['value']}", ha="center", fontsize=7)
        # 画分配的资源
        for f_id2, resource, _ in best["allocation"]:
            if f_id2 == f_id:
                icon = "[C]" if resource=="Cover" else "[AF]"
                eff = "100%" if resource=="Cover" else "80%"
                ax.text(fx, fy-0.45, f"{icon} {resource} ({eff})", ha="center", fontsize=7, color="#1565C0")

    # Panel 3: 覆盖对比矩阵
    ax = axes[2]
    field_ids = ["A","B","C"]
    x = np.arange(len(field_ids))
    width = 0.2
    for i, r in enumerate(results):
        coverage = [r["protected"].get(f,0) for f in field_ids]
        ax.bar(x+(i-1.5)*width, coverage, width, label=r["name"].split("\n")[0][:15], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Field {f}\nRisk {fields[f]['frost_risk']*100:.0f}%" for f in field_ids], fontsize=8)
    ax.set_ylabel("Protection Effectiveness", fontsize=10)
    ax.set_title("Coverage by Field & Strategy", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scheme_C_cpn_gantt.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return {"data":data,"results":results,"best":best,
            "saving_value":saving,"saving_pct":round(saving/worst["value"]*100,1) if worst["value"]>0 else 0}


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
    all_ops.sort(key=lambda o:o["day"])

    day_groups = {}
    for op in all_ops:
        d = op["day"]
        if d not in day_groups:
            day_groups[d] = {"pump":0,"drone":0}
        if op["op"] in ["Irrigate","Fertilize"]:
            day_groups[d]["pump"] += 1
        else:
            day_groups[d]["drone"] += 1

    configs = [("C1: 1P+1D",1,1),("C2: 2P+1D",2,1),("C3: 1P+2D",1,2),("C4: 2P+2D",2,2)]
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
        results.append({"name":name,"work":total_work,"wait":total_wait,
                        "total":total_work+total_wait,"conflict_days":conflict_days})

    results.sort(key=lambda r:r["total"])
    best = results[0]
    baseline = [r for r in results if "1P+1D" in r["name"]][0]
    saving = baseline["total"]-best["total"]

    # ── 可视化 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: 温度 + GDD
    ax = axes[0,0]
    spring = get_scheme_a_data()["daily_temps"]  # reuse real temps
    days = [r["day"] for r in spring]
    temps = [r["t2m"] for r in spring]
    ax.fill_between(days[:60], [t-5 for t in temps[:60]], [t+5 for t in temps[:60]], alpha=0.15, color="#1565C0")
    ax.plot(days[:60], temps[:60], color="#1565C0", linewidth=1.5)
    ax.axhline(y=0, color="#F44336", linestyle="--", alpha=0.4)
    stages = data["stages"]
    for s in stages:
        ax.axvline(x=s["date"].split("/")[1] if "/" in s["date"] else 0, color="#FF9800", linestyle=":", alpha=0.4)
    ax.set_xlabel("Day (Apr 1 →)", fontsize=9)
    ax.set_ylabel("Temperature (°C)", fontsize=9)
    ax.set_title("NASA POWER 2024: Temperature & GDD Stages", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2)

    # Panel 2: 操作甘特图 (全季)
    ax = axes[0,1]
    field_y = {"A":3.5, "B":2.5, "C":1.5}
    for op in all_ops:
        y = field_y[op["field"]]
        c = OP_COLORS.get(op["op"], "#999")
        dur_days = max(op["duration"]/60*0.5, 0.3)
        ax.barh(y, dur_days, 0.4, left=op["day"], color=c, edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_yticks([3.5,2.5,1.5])
    ax.set_yticklabels(["Field A (3ha)","Field B (4ha)","Field C (2.5ha)"], fontsize=8)
    ax.set_xlabel("Day", fontsize=9)
    ax.set_title("Season Operations Timeline", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 65)

    # Panel 3: 资源配置对比
    ax = axes[0,2]
    names = [r["name"].split(":")[1].strip() for r in results]
    works = [r["work"] for r in results]
    waits = [r["wait"] for r in results]
    x = np.arange(len(names))
    ax.bar(x-0.2, works, 0.4, label="Operation Time", color="#90CAF9", edgecolor="white")
    ax.bar(x+0.2, waits, 0.4, label="Wait Time", color="#F44336", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Time (min)", fontsize=9)
    ax.set_title("Operation vs. Wait Time", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    for i,(w,wait) in enumerate(zip(works,waits)):
        if wait > 0:
            ax.text(i, w+wait/2, f"+{wait}", ha="center", fontsize=7, color="#D32F2F")

    # Panel 4: 冲突日分布
    ax = axes[1,0]
    conflict_days_list = [(d,day_groups[d]["pump"]+day_groups[d]["drone"]) for d in sorted(day_groups.keys())]
    cd_days = [d for d,_ in conflict_days_list]
    cd_counts = [c for _,c in conflict_days_list]
    colors = ["#F44336" if c>1 else "#90A4AE" for c in cd_counts]
    ax.bar(cd_days, cd_counts, color=colors, edgecolor="white", width=0.8)
    ax.set_xlabel("Day", fontsize=9)
    ax.set_ylabel("Concurrent Operations", fontsize=9)
    ax.set_title(f"Daily Operation Density (1P+1D: {baseline['conflict_days']} conflict days)", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 65)
    ax.grid(alpha=0.2, axis="y")

    # Panel 5: 瓶颈分析
    ax = axes[1,1]
    c2 = [r for r in results if "2P+1D" in r["name"]][0]
    c3 = [r for r in results if "1P+2D" in r["name"]][0]
    c4 = [r for r in results if "2P+2D" in r["name"]][0]
    pump_save = baseline["wait"]-c2["wait"]
    drone_save = baseline["wait"]-c3["wait"]
    both_save = baseline["wait"]-c4["wait"]
    ax.barh(["Pump (+1)","Drone (+1)","Both (+1+1)"], [pump_save,drone_save,both_save],
            color=["#2196F3","#FF9800","#4CAF50"], edgecolor="white")
    ax.set_xlabel("Wait Time Saved (min)", fontsize=9)
    ax.set_title("Bottleneck Analysis: Which Resource Matters?", fontsize=10, fontweight="bold")
    for i,v in enumerate([pump_save,drone_save,both_save]):
        ax.text(v+2, i, f"{v} min", va="center", fontsize=9, fontweight="bold")

    # Panel 6: 数据溯源
    ax = axes[1,2]
    ax.axis("off")
    summary = (
        f"Scheme D: Full Season Simulation\n"
        f"{'─'*35}\n\n"
        f"Data: NASA POWER 2024\n"
        f"  Hefei (31.8°N, 117.2°E)\n"
        f"  Apr 1 → Jun 30 (91 days)\n\n"
        f"Model: GDD (base=0°C)\n"
        f"  Mean temp: {data['spring_mean_temp']}°C\n"
        f"  Total GDD: {data['total_gdd']}\n\n"
        f"Stages (2024 actual):\n"
        + "\n".join(f"  GDD {s['threshold']}: {s['stage']}\n    → {s['date']} ({s['gdd']} GDD)"
                     for s in data["stages"]) +
        f"\n\nResults:\n"
        f"  1P+1D: {baseline['total']} min\n"
        f"  Best: {best['name']} = {best['total']} min\n"
        f"  Saving: {saving} min ({round(saving/baseline['total']*100,1)}%)\n"
        f"  Conflict days: {baseline['conflict_days']} → {best['conflict_days']}\n\n"
        f"Key insight:\n"
        f"CPN reveals that GDD naturally\n"
        f"stages operations, but additional\n"
        f"resources reduce wait time\n"
        f"on conflict days."
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=7.5,
            fontfamily="monospace", va="top", linespacing=1.2,
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#CCC"))

    legend = [Patch(facecolor=c, label=op) for op,c in OP_COLORS.items() if op in ["Irrigate","Fertilize","Scout+Spray","Harvest"]]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scheme_D_cpn_gantt.png"), dpi=200, bbox_inches="tight")
    plt.close()

    return {"data":data,"results":results,"best":best,"baseline":baseline,
            "saving_min":saving,"saving_pct":round(saving/baseline["total"]*100,1) if baseline["total"]>0 else 0}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  全方案真实数据验证 — NASA POWER + PlantVillage")
    print("  CPN 执行图自动生成中...")
    print("=" * 70)

    rA = run_scheme_A()
    rB = run_scheme_B()
    rC = run_scheme_C()
    rD = run_scheme_D()

    all_results = {"A": rA, "B": rB, "C": rC, "D": rD}

    # 文字摘要
    for scheme_id, label in [("A","Irrigation"),("B","Pest Control"),("C","Frost"),("D","Full Season")]:
        r = all_results[scheme_id]
        print(f"\n{'─'*50}")
        print(f"  Scheme {scheme_id}: {label}")
        if scheme_id == "A":
            print(f"  Best: {r['best']['order']} → {r['best']['total']} min")
            print(f"  Worst: {r['worst']['order']} → {r['worst']['total']} min")
            print(f"  Saving: {r['saving_min']} min ({r['saving_pct']}%)")
        elif scheme_id == "B":
            print(f"  Best: {r['best']['name']} → {r['best']['total']} min")
            print(f"  Saving: {r['saving_min']} min ({r['saving_pct']}%)")
        elif scheme_id == "C":
            print(f"  Best: {r['best']['name'].split(chr(10))[0]} → Y{r['best']['value']:.0f}")
            print(f"  Saving: Y{r['saving_value']:.0f} ({r['saving_pct']}%)")
        elif scheme_id == "D":
            print(f"  Best: {r['best']['name']} → {r['best']['total']} min")
            print(f"  Saving: {r['saving_min']} min ({r['saving_pct']}%)")

    # 保存 JSON
    with open(os.path.join(OUT_DIR, "all_schemes_real_data.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  ✓ 文字结果: {OUT_DIR}/all_schemes_real_data.json")
    print(f"  ✓ CPN执行图: {FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"      {f}")
    print(f"{'='*70}")