#!/usr/bin/env python3
"""
Nature 期刊风格 CPN 图生成器 v6
================================
- 每个子图独立绘制，数据不混淆
- Petri 网执行状态图 (cpnpy + matplotlib)
- 图例统一右上角
"""

import sys, os, json, csv, math, shutil, datetime, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch, FancyArrowPatch
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_data_loader import get_scheme_a_data, get_scheme_b_data, get_scheme_c_data, get_scheme_d_data
from real_data_loader import get_spring_2024 as get_spring_2024_full

# cpnpy
from cpnpy.cpn.cpn_imp import CPN as _CPN, Place as _Place, Transition as _Trans, Arc as _Arc, Marking as _Marking
from cpnpy.cpn.colorsets import EnumeratedColorSet, UnitColorSet

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(PROJ_DIR, "output")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
DATA_DIR = os.path.join(OUT_DIR, "data")
PAPER_FIG_DIR = os.path.join(PROJ_DIR, "..", "..", "figures")
for d in [OUT_DIR, FIG_DIR, DATA_DIR, PAPER_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

VERSION = datetime.datetime.now().strftime("v%Y%m%d_%H%M%S")
VERSION_DIR = os.path.join(OUT_DIR, "versions", VERSION)
os.makedirs(VERSION_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans", "Arial"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

NATURE_BLUE="#377EB8"; NATURE_GREEN="#4DAF4A"; NATURE_RED="#E41A1C"
NATURE_ORANGE="#FF7F00"; NATURE_PURPLE="#984EA3"; NATURE_GRAY="#999999"

OP_COLORS = {"Irrigate":NATURE_BLUE,"Fertilize":NATURE_GREEN,"Scout":NATURE_ORANGE,
             "Spray":NATURE_RED,"Harvest":NATURE_PURPLE}


def publish(fig, name):
    """保存 PNG (预览) + SVG (Adobe Illustrator 可编辑)"""
    base = os.path.splitext(name)[0]
    for p in [os.path.join(FIG_DIR, name), os.path.join(VERSION_DIR, name),
              os.path.join(PAPER_FIG_DIR, name)]:
        try: fig.savefig(p, dpi=300)
        except PermissionError: pass
    # SVG 版本 (Adobe Illustrator 可编辑)
    svg_name = base + ".svg"
    for p in [os.path.join(FIG_DIR, svg_name), os.path.join(VERSION_DIR, svg_name),
              os.path.join(PAPER_FIG_DIR, svg_name)]:
        try: fig.savefig(p, format="svg", dpi=300)
        except PermissionError: pass
    plt.close()

def save_data(name, d):
    base = os.path.join(DATA_DIR, name)
    with open(base+".json","w") as f: json.dump(d, f, indent=2, default=str)
    if "results" in d and d["results"]:
        # 写 UTF-8 BOM，Excel 才能正确识别中文
        with open(base+".csv","wb") as f:
            f.write(b'\xef\xbb\xbf')  # UTF-8 BOM
            content = io.StringIO()
            w = csv.DictWriter(content, fieldnames=d["results"][0].keys())
            w.writeheader(); w.writerows(d["results"])
            f.write(content.getvalue().encode("utf-8"))
    else:
        print(f"  [SKIP CSV] {name}: results={d.get('results','MISSING')[:1] if d.get('results') else 'EMPTY'}")


# ═══════════════════════════════════════════════════════════════════════
# Petri 网状态图绘制 (cpnpy 构建 + matplotlib 渲染)
# ═══════════════════════════════════════════════════════════════════════

def draw_petri_execution_states(cpn_spec, state_sequence, filename, figsize=(10, 7)):
    """
    绘制 Petri 网执行状态序列图。

    每个面板展示:
      - CPN 结构 (○ 库所, □ 变迁, → 弧)
      - ● 令牌分布 (红色圆点 = 令牌, 数字 = 数量)
      - 橙色填充 = 正在触发的变迁
      - 绿色边框 = 当前使能的变迁
      - 底部标注触发条件

    cpn_spec: {places, transitions, arcs}
    state_sequence: [{desc, marking, firing, enabled, condition}]
    """
    n_states = len(state_sequence)
    ncols = min(3, n_states)
    nrows = math.ceil(n_states / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_states == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    # 预计算节点位置
    node_pos = {}
    for name, x, y in cpn_spec["places"]:
        node_pos[name] = (x, y, "place", 0.28)  # radius
    for name, x, y, w, h in cpn_spec["transitions"]:
        node_pos[name] = (x, y, "trans", (w, h))

    for idx, state in enumerate(state_sequence):
        ax = axes[idx]
        ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 5.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(state["desc"], fontsize=7.5, fontweight="bold", pad=6)

        marking = state["marking"]
        firing = state.get("firing")
        enabled_set = set(state.get("enabled", []))

        # 画弧 (P→T 或 T→P, 绝不 P→P 或 T→T)
        drawn_arcs = set()
        for src_name, tgt_name in cpn_spec["arcs"]:
            if src_name not in node_pos or tgt_name not in node_pos:
                continue
            sx, sy, st, sr = node_pos[src_name]
            tx, ty, tt, tr = node_pos[tgt_name]
            # 验证: 必须是 P→T 或 T→P
            if st == tt:
                continue  # 跳过 P→P 或 T→T 的错误弧
            dx, dy = tx - sx, ty - sy
            ln = math.hypot(dx, dy)
            if ln < 1e-9: continue
            # 偏移到节点边缘
            if st == "place":
                sx += dx/ln * sr; sy += dy/ln * sr
            else:
                w, h = sr; sx += dx/ln * (w/2); sy += dy/ln * (h/2)
            if tt == "place":
                tx -= dx/ln * tr; ty -= dy/ln * tr
            else:
                w, h = tr; tx -= dx/ln * (w/2); ty -= dy/ln * (h/2)
            # 去重
            arc_key = (round(sx,2), round(sy,2), round(tx,2), round(ty,2))
            if arc_key in drawn_arcs: continue
            drawn_arcs.add(arc_key)
            ax.annotate("", xy=(tx, ty), xytext=(sx, sy),
                       arrowprops=dict(arrowstyle="->", color="#555", lw=1.0), zorder=5)

        # 画变迁
        for name, x, y, w, h in cpn_spec["transitions"]:
            is_firing = (name == firing)
            is_enabled = (name in enabled_set)
            fc = NATURE_ORANGE if is_firing else "#F0F0F0"
            ec = NATURE_GREEN if is_enabled else ("#E53935" if is_firing else "#555")
            lw = 2.5 if is_firing else (2.0 if is_enabled else 1.2)
            rect = Rectangle((x - w/2, y - h/2), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, zorder=10)
            ax.add_patch(rect)
            fs = 6 if len(name) <= 8 else 5
            ax.text(x, y, name, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=11)

        # 画库所
        for name, x, y in cpn_spec["places"]:
            c = Circle((x, y), 0.28, facecolor="white", edgecolor="#333", linewidth=1.5, zorder=10)
            ax.add_patch(c)
            ax.text(x, y - 0.45, name, ha="center", va="top", fontsize=6, fontweight="bold", zorder=11)
            # 令牌
            n_tok = marking.get(name, 0)
            if n_tok > 0:
                if n_tok == 1:
                    dot = Circle((x, y), 0.06, facecolor=NATURE_RED, edgecolor="none", zorder=20)
                    ax.add_patch(dot)
                elif n_tok <= 4:
                    for i in range(n_tok):
                        a = 2 * math.pi * i / n_tok
                        dot = Circle((x + 0.10*math.cos(a), y + 0.10*math.sin(a)),
                                   0.045, facecolor=NATURE_RED, edgecolor="none", zorder=20)
                        ax.add_patch(dot)
                else:
                    dot = Circle((x, y), 0.06, facecolor=NATURE_RED, edgecolor="none", zorder=20)
                    ax.add_patch(dot)
                    ax.text(x + 0.15, y + 0.15, str(n_tok), fontsize=6, color=NATURE_RED,
                           fontweight="bold", zorder=21)

        # 触发条件标注
        condition = state.get("condition", "")
        if condition:
            ax.text(3.2, -0.3, condition, ha="center", fontsize=5.5, color="#555",
                   fontstyle="italic", transform=ax.transData)

    # 图例
    legend_text = ("○ Place  □ Transition  ● Token  "
                   "Orange=Firing  Green=Enabled  \n"
                   "Arc: Place↔Transition only")
    fig.text(0.5, 0.01, legend_text, ha="center", fontsize=6, color="#555",
            fontstyle="italic", transform=fig.transFigure)

    # 隐藏多余子图
    for idx in range(n_states, len(axes)):
        axes[idx].axis("off")

    plt.subplots_adjust(wspace=0.25, hspace=0.40, bottom=0.05)
    publish(fig, filename)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 方案 A：灌溉调度
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_A():
    data = get_scheme_a_data()
    fields = data["fields"]
    durs = {f:info["irrigation_min"] for f,info in fields.items()}
    post = {"A":[("Fertilize",30),("Scout",20),("Spray",25)],
            "B":[("Fertilize",20),("Scout",14),("Spray",17)],
            "C":[("Fertilize",15),("Scout",10),("Spray",13)]}

    results=[]; traces={}
    for order in permutations(["A","B","C"]):
        pt=0; tr=[]; fe={}
        for f in order:
            tr.append({"f":f,"op":"Irrigate","s":pt,"e":pt+durs[f]})
            pt+=durs[f]; t=pt
            for on,d in post[f]: tr.append({"f":f,"op":on,"s":t,"e":t+d}); t+=d
            fe[f]=t
        total=max(fe.values()); os_="→".join(order)
        results.append({"order":os_,"total":total}); traces[os_]=tr
    results.sort(key=lambda r:r["total"])
    best,worst=results[0],results[-1]
    saving=worst["total"]-best["total"]
    pct=saving/worst["total"]*100 if worst["total"]>0 else 0

    # ── 子图: 顺序对比 ──
    fig1,ax=plt.subplots(figsize=(3.5,2.4))
    names=[r["order"] for r in results]; totals=[r["total"] for r in results]
    colors=[NATURE_GREEN if i==0 else NATURE_RED if i==len(results)-1 else NATURE_GRAY for i in range(len(results))]
    ax.barh(range(len(results)),totals,color=colors,edgecolor="white",height=0.6)
    ax.set_yticks(range(len(results))); ax.set_yticklabels(names,fontsize=7)
    ax.set_xlabel("Total time (min)"); ax.invert_yaxis()
    mx=max(totals)
    for i,t in enumerate(totals):
        ax.text(t+mx*0.01, i, f"{t}", va="center", fontsize=7, fontweight="bold", color=colors[i])
    ax.set_xlim(0,mx*1.10)
    ax.set_title("Irrigation order vs. completion time",fontsize=9,fontweight="bold")
    publish(fig1,"fig_A1_bar.png")

    # ── 子图: 最优甘特图 ──
    fig2,ax=plt.subplots(figsize=(4.0,2.0))
    fy={"A":3.5,"B":2.5,"C":1.5}
    tr=traces[best["order"]]
    for e in tr:
        y=fy[e["f"]]; c=OP_COLORS.get(e["op"],NATURE_GRAY)
        ax.barh(y, e["e"]-e["s"], 0.45, left=e["s"], color=c, edgecolor="white", linewidth=0.3)
    ax.set_yticks([3.5,2.5,1.5]); ax.set_yticklabels(["Field A","Field B","Field C"],fontsize=7)
    ax.set_xlabel("Time (min)"); ax.set_title(f"Best: {best['order']} ({best['total']} min)",fontsize=9,fontweight="bold")
    ax.legend(handles=[Patch(facecolor=c,label=op,edgecolor="white") for op,c in
              [("Irrigate",NATURE_BLUE),("Fertilize",NATURE_GREEN),("Scout",NATURE_ORANGE),("Spray",NATURE_RED)]],
              fontsize=6,loc="upper right",ncol=4,frameon=False)
    publish(fig2,"fig_A2_gantt_best.png")

    # ── 子图: 最差甘特图 ──
    fig3,ax=plt.subplots(figsize=(4.0,2.0))
    tr=traces[worst["order"]]
    for e in tr:
        y=fy[e["f"]]; c=OP_COLORS.get(e["op"],NATURE_GRAY)
        ax.barh(y, e["e"]-e["s"], 0.45, left=e["s"], color=c, edgecolor="white", linewidth=0.3)
    ax.set_yticks([3.5,2.5,1.5]); ax.set_yticklabels(["Field A","Field B","Field C"],fontsize=7)
    ax.set_xlabel("Time (min)"); ax.set_title(f"Worst: {worst['order']} ({worst['total']} min)",fontsize=9,fontweight="bold")
    publish(fig3,"fig_A3_gantt_worst.png")

    save_data("scheme_A",{"results":results,"best":best,"worst":worst,"saving_pct":round(pct,1)})

    # ── 组图: 三面板 ──
    figC,axesC=plt.subplots(1,3,figsize=(7.5,2.6))
    plt.subplots_adjust(left=0.06, right=0.94, wspace=0.40, top=0.88, bottom=0.18)
    # 面板 a: 顺序对比
    ax=axesC[0]; ax.text(-0.10,1.06,"a",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    names=[r["order"] for r in results]; totals=[r["total"] for r in results]
    colors=[NATURE_GREEN if i==0 else NATURE_RED if i==len(results)-1 else NATURE_GRAY for i in range(len(results))]
    ax.barh(range(len(results)),totals,color=colors,edgecolor="white",height=0.6)
    ax.set_yticks(range(len(results))); ax.set_yticklabels(names,fontsize=7)
    ax.set_xlabel("Total time (min)"); ax.invert_yaxis()
    mx=max(totals)
    for i,t in enumerate(totals):
        ax.text(t+mx*0.01, i, f"{t}", va="center", fontsize=7, fontweight="bold", color=colors[i])
    ax.set_xlim(0,mx*1.10)
    # 面板 b: 最优甘特图
    ax=axesC[1]; ax.text(-0.10,1.06,"b",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    fy={"A":3.5,"B":2.5,"C":1.5}
    for e in traces[best["order"]]:
        y=fy[e["f"]]; c=OP_COLORS.get(e["op"],NATURE_GRAY)
        ax.barh(y, e["e"]-e["s"], 0.45, left=e["s"], color=c, edgecolor="white", linewidth=0.3)
    ax.set_yticks([3.5,2.5,1.5]); ax.set_yticklabels(["Field A","Field B","Field C"],fontsize=7)
    ax.set_xlabel("Time (min)"); ax.set_title(f"Best: {best['order']} ({best['total']} min)",fontsize=8,fontweight="bold",pad=3)
    # 面板 c: 最差甘特图
    ax=axesC[2]; ax.text(-0.10,1.06,"c",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    for e in traces[worst["order"]]:
        y=fy[e["f"]]; c=OP_COLORS.get(e["op"],NATURE_GRAY)
        ax.barh(y, e["e"]-e["s"], 0.45, left=e["s"], color=c, edgecolor="white", linewidth=0.3)
    ax.set_yticks([3.5,2.5,1.5]); ax.set_yticklabels(["Field A","Field B","Field C"],fontsize=7)
    ax.set_xlabel("Time (min)"); ax.set_title(f"Worst: {worst['order']} ({worst['total']} min)",fontsize=8,fontweight="bold",pad=3)
    figC.legend(handles=[Patch(facecolor=c,label=op,edgecolor="white") for op,c in
               [("Irrigate",NATURE_BLUE),("Fertilize",NATURE_GREEN),("Scout",NATURE_ORANGE),("Spray",NATURE_RED)]],
               loc="lower center",ncol=4,fontsize=6,frameon=False,bbox_to_anchor=(0.5,-0.02))
    publish(figC,"fig_scheme_A.png")

    # ── Petri 网执行状态图 (6 步, 展示使能条件) ──
    cpn_spec = {
        "places": [("Ready", 0.5, 3.8), ("Pump", 0.5, 1.0), ("Irrigated", 3.2, 3.8),
                   ("Fertilized", 3.2, 1.0), ("Done", 6.0, 2.4)],
        "transitions": [("Irrigate", 1.8, 3.8, 0.8, 0.35), ("Fertilize", 1.8, 1.0, 0.8, 0.35),
                       ("Spray", 4.5, 2.4, 0.8, 0.35)],
        "arcs": [("Ready","Irrigate"),("Irrigate","Irrigated"),("Pump","Irrigate"),("Irrigate","Pump"),
                 ("Irrigated","Fertilize"),("Fertilize","Fertilized"),
                 ("Fertilized","Spray"),("Spray","Done")],
    }
    states = [
        {"desc":"S0: Initial — 3 fields ready, 1 pump idle",
         "marking":{"Ready":3,"Pump":1,"Irrigated":0,"Fertilized":0,"Done":0},
         "firing":None,"enabled":["Irrigate"],
         "condition":"Irrigate enabled: Ready>=1, Pump>=1"},
        {"desc":"S1: Irrigate A fires — pump locked, Ready→2",
         "marking":{"Ready":2,"Pump":0,"Irrigated":0,"Fertilized":0,"Done":0},
         "firing":"Irrigate","enabled":[],
         "condition":"Irrigate disabled: Pump=0  (token consumed, returns after delay)"},
        {"desc":"S2: Irrigate completes — token returns, A→Irrigated",
         "marking":{"Ready":2,"Pump":1,"Irrigated":1,"Fertilized":0,"Done":0},
         "firing":None,"enabled":["Irrigate","Fertilize"],
         "condition":"Irrigate enabled (Ready>=1,Pump>=1), Fertilize enabled (Irrigated>=1)"},
        {"desc":"S3: Irrigate B fires — pump busy, Fertilize A fires in parallel",
         "marking":{"Ready":1,"Pump":0,"Irrigated":1,"Fertilized":1,"Done":0},
         "firing":"Irrigate","enabled":["Fertilize"],
         "condition":"Fertilize A enabled: Irrigated≥1  (no pump needed for fertilize)"},
        {"desc":"S4: All irrigated, A fertilized, B fertilizing, C waiting",
         "marking":{"Ready":0,"Pump":1,"Irrigated":1,"Fertilized":1,"Done":0},
         "firing":None,"enabled":["Fertilize","Spray"],
         "condition":"Fertilize enabled (Irrigated≥1) & Spray enabled (Fertilized≥1)"},
        {"desc":"S5: Spray A fires — final field operation, A→Done",
         "marking":{"Ready":0,"Pump":1,"Irrigated":0,"Fertilized":0,"Done":1},
         "firing":"Spray","enabled":[],
         "condition":"Spray A fires: Fertilized≥1  (no external resource needed)"},
        {"desc":"S6: All 3 fields complete (Done=3), pump idle",
         "marking":{"Ready":0,"Pump":1,"Irrigated":0,"Fertilized":0,"Done":3},
         "firing":None,"enabled":[],
         "condition":"Deadlock: no enabled transitions, all fields processed"},
    ]
    draw_petri_execution_states(cpn_spec, states, "fig_A_petri.png", figsize=(12, 8))

    return {"saving_pct":round(pct,1)}


# ═══════════════════════════════════════════════════════════════════════
# 方案 B：植保调度
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_B():
    data=get_scheme_b_data(); fields=data["fields"]
    durs={f:info["spray_min"] for f,info in fields.items()}; st=15
    strategies=[
        ("S1: A first",[("Scout","A"),("Spray","A"),("Scout","B"),("Spray","B")],1),
        ("S2: B first",[("Scout","B"),("Spray","B"),("Scout","A"),("Spray","A")],1),
        ("S3: Scout both, spray short 1st",[("Scout","A"),("Scout","B"),("Spray","B"),("Spray","A")],1),
        ("S4: Scout both, spray long 1st",[("Scout","A"),("Scout","B"),("Spray","A"),("Spray","B")],1),
        ("S5: 2 drones parallel",[("Scout","A"),("Spray","A"),("Scout","B"),("Spray","B")],2),
    ]
    results=[]; all_traces={}
    for name,seq,nd in strategies:
        tr=[]
        if nd==1:
            t=0
            for op,f in seq: dur=st if op=="Scout" else durs[f]; tr.append({"f":f,"op":op,"s":t,"e":t+dur,"dr":1}); t+=dur
            total=t
        else:
            t1=0;t2=0
            for op,f in seq:
                if f=="A": dur=st if op=="Scout" else durs[f]; tr.append({"f":f,"op":op,"s":t1,"e":t1+dur,"dr":1}); t1+=dur
                else: dur=st if op=="Scout" else durs[f]; tr.append({"f":f,"op":op,"s":t2,"e":t2+dur,"dr":2}); t2+=dur
            total=max(t1,t2)
        results.append({"name":name,"total":total,"n_drones":nd,"_key":name})
        all_traces[name]=tr
    results.sort(key=lambda r:r["total"])
    best=results[0]; best1=min((r for r in results if r["n_drones"]==1),key=lambda r:r["total"])
    saving=best1["total"]-best["total"]

    # 图1: 策略对比
    fig1,ax=plt.subplots(figsize=(3.5,2.4))
    names=[r["name"] for r in results]; totals=[r["total"] for r in results]
    colors=[NATURE_GREEN if i==0 else NATURE_BLUE if r["n_drones"]==2 else NATURE_GRAY for i,r in enumerate(results)]
    ax.barh(range(len(results)),totals,color=colors,edgecolor="white",height=0.6)
    ax.set_yticks(range(len(results))); ax.set_yticklabels(names,fontsize=6.5)
    ax.set_xlabel("Total time (min)"); ax.invert_yaxis()
    mx=max(totals)
    for i,t in enumerate(totals):
        ax.text(t+mx*0.01, i, f"{t}", va="center", fontsize=7, fontweight="bold", color=colors[i])
    ax.set_xlim(0,mx*1.10)
    ax.set_title("Strategy vs. completion time",fontsize=9,fontweight="bold")
    publish(fig1,"fig_B1_bar.png")

    # 图2: 1 drone 甘特图
    fig2,ax=plt.subplots(figsize=(4.0,1.8))
    tr=all_traces[best1["_key"]]
    for e in tr:
        y=1.5 if e["f"]=="A" else 0.5; c=OP_COLORS.get(e["op"],NATURE_GRAY)
        ax.barh(y,e["e"]-e["s"],0.4,left=e["s"],color=c,edgecolor="white",linewidth=0.3)
    ax.set_yticks([1.5,0.5]); ax.set_yticklabels(["Field A (severe)","Field B (mild)"],fontsize=7)
    ax.set_xlabel("Time (min)"); ax.set_title(f"1 drone: {best1['total']} min",fontsize=9,fontweight="bold")
    ax.legend(handles=[Patch(facecolor=c,label=op,edgecolor="white") for op,c in [("Scout",NATURE_ORANGE),("Spray",NATURE_RED)]],
              fontsize=6,loc="upper right",ncol=2,frameon=False)
    publish(fig2,"fig_B2_gantt_1drone.png")

    # 图3: 2 drones 甘特图
    fig3,ax=plt.subplots(figsize=(4.0,1.8))
    tr=all_traces[best["_key"]]
    for e in tr:
        y=1.3 if e["dr"]==1 else 0.3; c=OP_COLORS.get(e["op"],NATURE_GRAY)
        ax.barh(y,e["e"]-e["s"],0.3,left=e["s"],color=c,edgecolor="white",linewidth=0.3)
    ax.set_yticks([1.3,0.3]); ax.set_yticklabels(["Drone 1","Drone 2"],fontsize=8)
    ax.set_xlabel("Time (min)"); ax.set_title(f"2 drones: {best['total']} min",fontsize=9,fontweight="bold")
    publish(fig3,"fig_B3_gantt_2drone.png")

    save_data("scheme_B",{"results":results,"best":best,"saving_pct":round(saving/best1["total"]*100,1)})

    # ── 组图: 三面板 ──
    figC,axesC=plt.subplots(1,3,figsize=(7.5,2.6))
    plt.subplots_adjust(left=0.06, right=0.94, wspace=0.40, top=0.88, bottom=0.18)
    ax=axesC[0]; ax.text(-0.10,1.06,"a",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    names=[r["name"] for r in results]; totals=[r["total"] for r in results]
    colors=[NATURE_GREEN if i==0 else NATURE_BLUE if r["n_drones"]==2 else NATURE_GRAY for i,r in enumerate(results)]
    ax.barh(range(len(results)),totals,color=colors,edgecolor="white",height=0.6)
    ax.set_yticks(range(len(results))); ax.set_yticklabels(names,fontsize=6.5)
    ax.set_xlabel("Total time (min)"); ax.invert_yaxis()
    mx=max(totals)
    for i,t in enumerate(totals):
        ax.text(t+mx*0.01, i, f"{t}", va="center", fontsize=7, fontweight="bold", color=colors[i])
    ax.set_xlim(0,mx*1.10)
    for idx,(tr,title) in enumerate([
        (all_traces[best1["_key"]],f"1 drone: {best1['total']} min"),
        (all_traces[best["_key"]],f"2 drones: {best['total']} min")]):
        ax=axesC[idx+1]; ax.text(-0.10,1.06,"bc"[idx],transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
        if idx==0:
            for e in tr:
                y=1.5 if e["f"]=="A" else 0.5; c=OP_COLORS.get(e["op"],NATURE_GRAY)
                ax.barh(y,e["e"]-e["s"],0.4,left=e["s"],color=c,edgecolor="white",linewidth=0.3)
            ax.set_yticks([1.5,0.5]); ax.set_yticklabels(["Field A","Field B"],fontsize=7)
        else:
            for e in tr:
                y=1.3 if e["dr"]==1 else 0.3; c=OP_COLORS.get(e["op"],NATURE_GRAY)
                ax.barh(y,e["e"]-e["s"],0.3,left=e["s"],color=c,edgecolor="white",linewidth=0.3)
            ax.set_yticks([1.3,0.3]); ax.set_yticklabels(["Drone 1","Drone 2"],fontsize=8)
        ax.set_xlabel("Time (min)"); ax.set_title(title,fontsize=8,fontweight="bold",pad=3)
    figC.legend(handles=[Patch(facecolor=c,label=op,edgecolor="white") for op,c in [("Scout",NATURE_ORANGE),("Spray",NATURE_RED)]],
               loc="lower center",ncol=2,fontsize=6,frameon=False,bbox_to_anchor=(0.5,-0.02))
    publish(figC,"fig_scheme_B.png")

    # ── Petri 网执行状态图 (1 drone, 5 步) ──
    cpn_spec = {
        "places": [("Ready", 0.5, 2.8), ("Drone", 5.5, 2.8), ("Scouted", 2.5, 2.8), ("Done", 5.5, 0.5)],
        "transitions": [("Scout", 1.5, 2.8, 0.7, 0.3), ("Spray", 3.8, 2.8, 0.7, 0.3)],
        "arcs": [("Ready","Scout"),("Scout","Scouted"),("Drone","Scout"),("Scout","Drone"),
                 ("Scouted","Spray"),("Spray","Done"),("Drone","Spray"),("Spray","Drone")],
    }
    states = [
        {"desc":"S0: Initial — 2 fields ready, 1 drone idle",
         "marking":{"Ready":2,"Drone":1,"Scouted":0,"Done":0},
         "firing":None,"enabled":["Scout"],
         "condition":"Scout enabled: Ready≥1 & Drone≥1"},
        {"desc":"S1: Scout A fires — drone locked, Ready→1",
         "marking":{"Ready":1,"Drone":0,"Scouted":0,"Done":0},
         "firing":"Scout","enabled":[],
         "condition":"Scout disabled: Drone=0  (returns after 15 min delay)"},
        {"desc":"S2: Scout A complete — drone returns, A→Scouted",
         "marking":{"Ready":1,"Drone":1,"Scouted":1,"Done":0},
         "firing":None,"enabled":["Scout","Spray"],
         "condition":"Scout enabled (Ready≥1,Drone≥1) & Spray enabled (Scouted≥1,Drone≥1)"},
        {"desc":"S3: Spray A fires — drone spraying A, B still waiting",
         "marking":{"Ready":1,"Drone":0,"Scouted":0,"Done":0},
         "firing":"Spray","enabled":[],
         "condition":"Spray consumes Scouted token + Drone; B waits for drone"},
        {"desc":"S4: Spray A done, Scout B fires — drone now scouting B",
         "marking":{"Ready":0,"Drone":0,"Scouted":0,"Done":1},
         "firing":"Scout","enabled":[],
         "condition":"A complete (Done=1), Scout B: Ready≥1,Drone≥1"},
        {"desc":"S5: All complete — both fields done, drone idle",
         "marking":{"Ready":0,"Drone":1,"Scouted":0,"Done":2},
         "firing":None,"enabled":[],
         "condition":"Deadlock: no enabled transitions, total 90 min"},
    ]
    draw_petri_execution_states(cpn_spec, states, "fig_B_petri_1drone.png", figsize=(12, 8))

    return {"saving_pct":round(saving/best1["total"]*100,1)}


# ═══════════════════════════════════════════════════════════════════════
# 方案 C：霜冻应急
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_C():
    data=get_scheme_c_data(); fields=data["fields"]
    strats=[
        ("S1: Covers→A,B, AF→C",[("A","Cover",30),("B","Cover",30),("C","AF",60)]),
        ("S2: Double protect A",[("A","Cover",30),("A","AF",60),("B","Cover",30)]),
        ("S3: Covers→A,C, AF→B",[("A","Cover",30),("C","Cover",30),("B","AF",60)]),
        ("S4: AF→A, Covers→B,C",[("A","AF",60),("B","Cover",30),("C","Cover",30)]),
    ]
    results=[]
    for name,alloc in strats:
        prot={}
        for f_id,res,_ in alloc: prot[f_id]=max(prot.get(f_id,0),1.0 if res=="Cover" else 0.80)
        tv=sum(fields[f]["value"]*prot.get(f,0) for f in fields)
        pot=sum(fields[f]["value"] for f in fields)
        results.append({"name":name,"value":tv,"coverage":round(tv/pot*100,1)})
    results.sort(key=lambda r:r["value"],reverse=True)
    best,worst=results[0],results[-1]; saving=best["value"]-worst["value"]

    # 图1: 策略对比
    fig1,ax=plt.subplots(figsize=(3.5,2.4))
    names=[r["name"] for r in results]; values=[r["value"] for r in results]
    colors=[NATURE_GREEN if i==0 else NATURE_RED if i==len(results)-1 else NATURE_GRAY for i in range(len(results))]
    ax.barh(range(len(results)),values,color=colors,edgecolor="white",height=0.6)
    ax.set_yticks(range(len(results))); ax.set_yticklabels(names,fontsize=6.5)
    ax.set_xlabel("Protection value (Yuan)"); ax.invert_yaxis()
    mx=max(values)
    for i,v in enumerate(values):
        ax.text(v+mx*0.01, i, f"{v:.0f}", va="center", fontsize=7, fontweight="bold", color=colors[i])
    ax.set_xlim(0,mx*1.10)
    ax.set_title("Strategy vs. protection value",fontsize=9,fontweight="bold")
    publish(fig1,"fig_C1_bar.png")

    # 图2: 覆盖矩阵
    fig2,ax=plt.subplots(figsize=(4.0,2.2))
    fids=["A","B","C"]; x=np.arange(3); w=0.18
    for i,(r,(_,alloc)) in enumerate(zip(results,strats)):
        prot={}
        for fid,res,_ in alloc: prot[fid]=max(prot.get(fid,0),1.0 if res=="Cover" else 0.80)
        cov=[prot.get(f,0) for f in fids]
        ax.bar(x+(i-1.5)*w,cov,w,label=r["name"][:12],alpha=0.85,edgecolor="white",linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([f"Field {f}\n({fields[f]['frost_risk']*100:.0f}% risk)" for f in fids],fontsize=6.5)
    ax.set_ylabel("Protection effectiveness"); ax.set_ylim(0,1.15)
    ax.legend(fontsize=5.5,loc="upper right",ncol=2,frameon=False)
    ax.set_title("Coverage by field & strategy",fontsize=9,fontweight="bold")
    publish(fig2,"fig_C2_coverage.png")

    save_data("scheme_C",{"results":results,"best":best,"saving_pct":round(saving/worst["value"]*100,1)})

    # ── 组图: 两面板 ──
    figC,axesC=plt.subplots(1,2,figsize=(7.5,2.8))
    plt.subplots_adjust(left=0.07, right=0.93, wspace=0.40, top=0.90, bottom=0.12)
    ax=axesC[0]; ax.text(-0.10,1.06,"a",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    names=[r["name"] for r in results]; values=[r["value"] for r in results]
    colors=[NATURE_GREEN if i==0 else NATURE_RED if i==len(results)-1 else NATURE_GRAY for i in range(len(results))]
    ax.barh(range(len(results)),values,color=colors,edgecolor="white",height=0.6)
    ax.set_yticks(range(len(results))); ax.set_yticklabels(names,fontsize=6.5)
    ax.set_xlabel("Protection value (Yuan)"); ax.invert_yaxis()
    mx=max(values)
    for i,v in enumerate(values):
        ax.text(v+mx*0.01, i, f"{v:.0f}", va="center", fontsize=7, fontweight="bold", color=colors[i])
    ax.set_xlim(0,mx*1.10)
    ax=axesC[1]; ax.text(-0.10,1.06,"b",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    fids=["A","B","C"]; x=np.arange(3); w=0.18
    for i,(r,(_,alloc)) in enumerate(zip(results,strats)):
        prot={}
        for fid,res,_ in alloc: prot[fid]=max(prot.get(fid,0),1.0 if res=="Cover" else 0.80)
        cov=[prot.get(f,0) for f in fids]
        ax.bar(x+(i-1.5)*w,cov,w,label=r["name"][:12],alpha=0.85,edgecolor="white",linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([f"Field {f}" for f in fids],fontsize=7)
    ax.set_ylabel("Protection effectiveness"); ax.set_ylim(0,1.15)
    ax.legend(fontsize=5.5,loc="upper right",ncol=2,frameon=False)
    publish(figC,"fig_scheme_C.png")

    # ── Petri 网执行状态图 (资源分配, 6 步) ──
    cpn_spec = {
        "places": [("Ready", 0.3, 4.5), ("Teams", 0.3, 2.5), ("Cover", 0.3, 0.5),
                   ("AF", 4.0, 0.5), ("Protected", 6.0, 2.5)],
        "transitions": [("Deploy\nCover", 2.2, 4.0, 0.7, 0.45), ("Deploy\nAF", 2.2, 0.5, 0.7, 0.45)],
        "arcs": [("Ready","Deploy\nCover"),("Teams","Deploy\nCover"),("Cover","Deploy\nCover"),
                 ("Deploy\nCover","Protected"),("Deploy\nCover","Teams"),("Deploy\nCover","Cover"),
                 ("Ready","Deploy\nAF"),("Teams","Deploy\nAF"),("AF","Deploy\nAF"),
                 ("Deploy\nAF","Protected"),("Deploy\nAF","Teams"),("Deploy\nAF","AF")],
    }
    states = [
        {"desc":"S0: Initial — 3 fields, 2 teams, 2 covers, 1 AF",
         "marking":{"Ready":3,"Teams":2,"Cover":2,"AF":1,"Protected":0},
         "firing":None,"enabled":["Deploy\nCover","Deploy\nAF"],
         "condition":"Deploy Cover enabled: Ready≥1 & Teams≥1 & Cover≥1"},
        {"desc":"S1: Deploy Cover→A — 1 team+1 cover consumed",
         "marking":{"Ready":2,"Teams":1,"Cover":1,"AF":1,"Protected":0},
         "firing":"Deploy\nCover","enabled":["Deploy\nCover"],
         "condition":"Deploy Cover still enabled: Ready≥1,Teams≥1,Cover≥1"},
        {"desc":"S2: Deploy Cover→B (parallel with A) — 2nd team busy",
         "marking":{"Ready":1,"Teams":0,"Cover":0,"AF":1,"Protected":0},
         "firing":"Deploy\nCover","enabled":[],
         "condition":"Deploy Cover disabled: Cover=0. AF enabled but Teams=0"},
        {"desc":"S3: Both covers done — resources returned, A,B protected",
         "marking":{"Ready":1,"Teams":2,"Cover":2,"AF":1,"Protected":2},
         "firing":None,"enabled":["Deploy\nAF"],
         "condition":"Deploy AF enabled: Ready≥1,Teams≥1,AF≥1 (Cover=2, no conflict)"},
        {"desc":"S4: Deploy AF→C — 1 team+AF consumed, last field",
         "marking":{"Ready":0,"Teams":1,"Cover":2,"AF":0,"Protected":2},
         "firing":"Deploy\nAF","enabled":[],
         "condition":"AF consumed, no more operations needed"},
        {"desc":"S5: All 3 fields protected — resources returned, idle",
         "marking":{"Ready":0,"Teams":2,"Cover":2,"AF":1,"Protected":3},
         "firing":None,"enabled":[],
         "condition":"Deadlock: all fields protected, S1 strategy complete"},
    ]
    draw_petri_execution_states(cpn_spec, states, "fig_C_petri.png", figsize=(12, 8))

    return {"saving_pct":round(saving/worst["value"]*100,1)}


# ═══════════════════════════════════════════════════════════════════════
# 方案 D：全季仿真 — 每个子图独立绘制
# ═══════════════════════════════════════════════════════════════════════

def run_scheme_D():
    data=get_scheme_d_data(); schedules=data["schedules"]
    all_ops=[]
    for f_id in ["A","B","C"]:
        for op in schedules[f_id]: all_ops.append(op)
    all_ops.sort(key=lambda o:o["day"])

    day_groups={}
    for op in all_ops:
        d=op["day"]
        if d not in day_groups: day_groups[d]={"pump":0,"drone":0}
        if op["op"] in ["Irrigate","Fertilize"]: day_groups[d]["pump"]+=1
        else: day_groups[d]["drone"]+=1

    configs=[("1P+1D",1,1),("2P+1D",2,1),("1P+2D",1,2),("2P+2D",2,2)]
    results=[]
    for name,npump,ndrone in configs:
        tw=sum(op["duration"] for op in all_ops); twait=0; cd=0
        for day,g in day_groups.items():
            if g["pump"]>npump: cd+=1; twait+=(g["pump"]-npump)*40
            if g["drone"]>ndrone: cd+=1; twait+=(g["drone"]-ndrone)*25
        results.append({"config":name,"work":tw,"wait":twait,"total":tw+twait,"conflict_days":cd})
    results.sort(key=lambda r:r["total"])
    best=results[0]; baseline=[r for r in results if r["config"]=="1P+1D"][0]
    saving=baseline["total"]-best["total"]

    spring=get_spring_2024_full()

    # ══ 子图 D1: 温度曲线 + GDD 阶段标记 ══
    fig1,ax=plt.subplots(figsize=(4.0,2.8))
    days=[r["day"] for r in spring[:65]]; temps=[r["t2m"] for r in spring[:65]]
    ax.fill_between(days,[t-5 for t in temps],[t+5 for t in temps],alpha=0.12,color=NATURE_BLUE)
    ax.plot(days,temps,color=NATURE_BLUE,linewidth=1.2)
    ax.axhline(y=0,color=NATURE_RED,linestyle="--",linewidth=0.6,alpha=0.5)
    for s in data["stages"]:
        doy=next((r["day"] for r in spring if f"{r['month']}/{r['day_of_month']}"==s["date"]),None)
        if doy:
            ax.axvline(x=doy,color=NATURE_ORANGE,linestyle=":",linewidth=0.5,alpha=0.5)
            ax.text(doy+0.5,30,s["stage"][:8],fontsize=5.5,color=NATURE_ORANGE,rotation=90,va="top")
    ax.set_xlabel("Day (April 1 → June 30)"); ax.set_ylabel("Temperature (°C)")
    ax.set_title("NASA POWER 2024: Daily temperature & GDD stages",fontsize=9,fontweight="bold")
    ax.set_xlim(0,65); ax.set_ylim(-5,38)
    publish(fig1,"fig_D1_temperature.png")

    # ══ 子图 D2: 三地块 GDD 累积曲线 ══
    fig2,ax=plt.subplots(figsize=(4.5,2.8))
    gdd_factors = {"A": 1.10, "B": 1.00, "C": 0.90}
    for f_id, color in [("A", NATURE_RED), ("B", NATURE_BLUE), ("C", NATURE_GREEN)]:
        factor = gdd_factors[f_id]
        gdd = 0; gdd_curve = []
        for r in spring[:65]:
            gdd += max(r["t2m"] * factor, 0)
            gdd_curve.append(gdd)
        ax.plot([r["day"] for r in spring[:65]], gdd_curve, color=color, linewidth=1.5, label=f"Field {f_id}")
    for threshold, stage_name in [(200,"Jointing"),(400,"Booting"),(600,"Heading"),(800,"GrainFill"),(1000,"Maturity")]:
        ax.axhline(y=threshold, color=NATURE_GRAY, linestyle=":", linewidth=0.5, alpha=0.5)
        ax.text(66, threshold, stage_name, fontsize=5.5, color=NATURE_GRAY, va="center")
    ax.set_xlabel("Day"); ax.set_ylabel("Accumulated GDD (°C)")
    ax.set_xlim(0, 65); ax.set_ylim(0, 1100)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_title("GDD accumulation by field (microclimate factor)", fontsize=9, fontweight="bold")
    publish(fig2,"fig_D2_gdd.png")

    # ══ 子图 D3: 资源配置对比 ══
    fig3,ax=plt.subplots(figsize=(3.5,2.2))
    cn=[r["config"] for r in results]; ws=[r["work"] for r in results]; wa=[r["wait"] for r in results]
    x=np.arange(4)
    ax.bar(x-0.2,ws,0.35,label="Operation",color=NATURE_BLUE,alpha=0.7,edgecolor="white",linewidth=0.3)
    ax.bar(x+0.2,wa,0.35,label="Wait",color=NATURE_RED,alpha=0.7,edgecolor="white",linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(cn,fontsize=7)
    ax.set_ylabel("Time (min)"); ax.legend(fontsize=7,frameon=False)
    # 等待时间标注放在条的上方，贴紧
    ymax=max(ws)+max(wa)
    for i,(w,wait) in enumerate(zip(ws,wa)):
        if wait>0: ax.text(i+0.2, w+wait+ymax*0.02, f"+{wait}", ha="center", fontsize=6.5, color=NATURE_RED)
    ax.set_ylim(0,ymax*1.10)
    ax.set_title("Resource configuration comparison",fontsize=9,fontweight="bold")
    publish(fig3,"fig_D3_configs.png")

    # ══ 子图 D4: 冲突日分布 ══
    fig4,ax=plt.subplots(figsize=(4.5,2.0))
    cd=[(d,day_groups[d]["pump"]+day_groups[d]["drone"]) for d in sorted(day_groups.keys())]
    cdd=[d for d,_ in cd]; cdc=[c for _,c in cd]
    colors=[NATURE_RED if c>1 else NATURE_GRAY for c in cdc]
    ax.bar(cdd,cdc,color=colors,edgecolor="white",width=0.8,linewidth=0.2)
    ax.set_xlabel("Day"); ax.set_ylabel("Concurrent operations"); ax.set_xlim(0,65)
    ax.axhline(y=1,color=NATURE_GRAY,linestyle=":",linewidth=0.5,alpha=0.5)
    ax.set_title("Daily operation density (1 pump + 1 drone)",fontsize=9,fontweight="bold")
    publish(fig4,"fig_D4_conflicts.png")

    # ══ 子图 D5: 瓶颈分析 ══
    fig5,ax=plt.subplots(figsize=(3.5,2.0))
    c2=[r for r in results if r["config"]=="2P+1D"][0]
    c3=[r for r in results if r["config"]=="1P+2D"][0]
    c4=[r for r in results if r["config"]=="2P+2D"][0]
    sv=[baseline["wait"]-c2["wait"],baseline["wait"]-c3["wait"],baseline["wait"]-c4["wait"]]
    ax.barh(["+1 Pump","+1 Drone","+1 Both"],sv,color=[NATURE_BLUE,NATURE_ORANGE,NATURE_GREEN],
            edgecolor="white",height=0.5,linewidth=0.3)
    ax.set_xlabel("Wait time saved (min)")
    for i,v in enumerate(sv):
        ax.text(v+1, i, f"{v} min", va="center", fontsize=7, fontweight="bold")
    ax.set_title("Bottleneck: which resource matters?",fontsize=9,fontweight="bold")
    publish(fig5,"fig_D5_bottleneck.png")

    save_data("scheme_D",{"results":results,"best":best,"saving_pct":round(saving/baseline["total"]*100,1)})

    # ── 组图: 2×3 面板 ──
    figC,axesC=plt.subplots(2,3,figsize=(7.5,5.5))
    plt.subplots_adjust(left=0.07, right=0.93, wspace=0.38, hspace=0.42, top=0.95, bottom=0.05)
    # a: 温度
    ax=axesC[0,0]; ax.text(-0.08,1.05,"a",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    days=[r["day"] for r in spring[:65]]; temps=[r["t2m"] for r in spring[:65]]
    ax.fill_between(days,[t-5 for t in temps],[t+5 for t in temps],alpha=0.12,color=NATURE_BLUE)
    ax.plot(days,temps,color=NATURE_BLUE,linewidth=1.2)
    ax.axhline(y=0,color=NATURE_RED,linestyle="--",linewidth=0.6,alpha=0.5)
    for s in data["stages"]:
        doy=next((r["day"] for r in spring if f"{r['month']}/{r['day_of_month']}"==s["date"]),None)
        if doy: ax.axvline(x=doy,color=NATURE_ORANGE,linestyle=":",linewidth=0.5,alpha=0.5)
    ax.set_xlabel("Day"); ax.set_ylabel("Temp (°C)"); ax.set_xlim(0,65); ax.set_ylim(-5,35)
    # b: 三地块 GDD 累积曲线
    ax=axesC[0,1]; ax.text(-0.08,1.05,"b",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    # 为每块地计算 GDD 累积
    gdd_factors = {"A": 1.10, "B": 1.00, "C": 0.90}
    for f_id, color in [("A", NATURE_RED), ("B", NATURE_BLUE), ("C", NATURE_GREEN)]:
        factor = gdd_factors[f_id]
        gdd = 0; gdd_curve = []
        for r in spring[:65]:
            gdd += max(r["t2m"] * factor, 0)
            gdd_curve.append(gdd)
        ax.plot([r["day"] for r in spring[:65]], gdd_curve, color=color, linewidth=1.5, label=f"Field {f_id}")
    # GDD 阈值线
    for threshold, stage_name in [(200,"Jointing"),(400,"Booting"),(600,"Heading"),(800,"GrainFill"),(1000,"Maturity")]:
        ax.axhline(y=threshold, color=NATURE_GRAY, linestyle=":", linewidth=0.5, alpha=0.5)
        ax.text(66, threshold, stage_name, fontsize=5.5, color=NATURE_GRAY, va="center")
    ax.set_xlabel("Day"); ax.set_ylabel("Accumulated GDD (°C)")
    ax.set_xlim(0, 65); ax.set_ylim(0, 1100)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    # c: 配置对比
    ax=axesC[0,2]; ax.text(-0.08,1.05,"c",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    cn=[r["config"] for r in results]; ws=[r["work"] for r in results]; wa=[r["wait"] for r in results]
    x=np.arange(4)
    ax.bar(x-0.2,ws,0.35,label="Operation",color=NATURE_BLUE,alpha=0.7,edgecolor="white",linewidth=0.3)
    ax.bar(x+0.2,wa,0.35,label="Wait",color=NATURE_RED,alpha=0.7,edgecolor="white",linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(cn,fontsize=7)
    ax.set_ylabel("Time (min)"); ax.legend(fontsize=6,loc="upper right",frameon=False)
    # d: 冲突日
    ax=axesC[1,0]; ax.text(-0.08,1.05,"d",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    cd=[(d,day_groups[d]["pump"]+day_groups[d]["drone"]) for d in sorted(day_groups.keys())]
    cdd=[d for d,_ in cd]; cdc=[c for _,c in cd]
    colors_bar=[NATURE_RED if c>1 else NATURE_GRAY for c in cdc]
    ax.bar(cdd,cdc,color=colors_bar,edgecolor="white",width=0.8,linewidth=0.2)
    ax.set_xlabel("Day"); ax.set_ylabel("Concurrent ops"); ax.set_xlim(0,65)
    ax.axhline(y=1,color=NATURE_GRAY,linestyle=":",linewidth=0.5,alpha=0.5)
    # e: 瓶颈
    ax=axesC[1,1]; ax.text(-0.08,1.05,"e",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    c2=[r for r in results if r["config"]=="2P+1D"][0]
    c3=[r for r in results if r["config"]=="1P+2D"][0]
    c4=[r for r in results if r["config"]=="2P+2D"][0]
    sv=[baseline["wait"]-c2["wait"],baseline["wait"]-c3["wait"],baseline["wait"]-c4["wait"]]
    ax.barh(["+1 Pump","+1 Drone","+1 Both"],sv,color=[NATURE_BLUE,NATURE_ORANGE,NATURE_GREEN],
            edgecolor="white",height=0.5,linewidth=0.3)
    ax.set_xlabel("Wait time saved (min)")
    for i,v in enumerate(sv): ax.text(v+1, i, f"{v}", va="center", fontsize=7, fontweight="bold")
    # f: 摘要
    ax=axesC[1,2]; ax.text(-0.08,1.05,"f",transform=ax.transAxes,fontsize=10,fontweight="bold",va="top")
    ax.axis("off")
    summary=(f"NASA POWER 2024, Hefei\n"
             f"GDD model (base 0°C)\n"
             f"Mean temp: {data['spring_mean_temp']}°C\n"
             f"Total GDD: {data['total_gdd']}\n\n"
             f"5 growth stages:\n"
             f"Jointing→Booting→Heading\n"
             f"→GrainFill→Maturity\n\n"
             f"1P+1D: {baseline['total']} min\n"
             f"Best: {best['total']} min\n"
             f"Saving: {saving} min\n"
             f"({round(saving/baseline['total']*100,1)}%)\n\n"
             f"Conflicts: {baseline['conflict_days']}→{best['conflict_days']} days")
    ax.text(0.05,0.95,summary,transform=ax.transAxes,fontsize=6.5,fontfamily="monospace",
            va="top",linespacing=1.3,bbox=dict(boxstyle="round,pad=0.4",facecolor="#FAFAFA",edgecolor="#DDD",linewidth=0.5))
    publish(figC,"fig_scheme_D.png")

    # ── Petri 网执行状态图 (全季, 7 步) ──
    cpn_spec = {
        "places": [("Field_A", 0.3, 4.5), ("Field_B", 0.3, 2.5), ("Field_C", 0.3, 0.5),
                   ("Pump", 3.8, 4.5), ("Drone", 3.8, 0.5), ("A_Done", 6.0, 4.5),
                   ("B_Done", 6.0, 2.5), ("C_Done", 6.0, 0.5)],
        "transitions": [("Irrigate", 2.0, 4.5, 0.7, 0.35), ("Fertilize", 2.0, 2.5, 0.7, 0.35),
                       ("ScoutSpray", 2.0, 0.5, 0.7, 0.45), ("Harvest", 5.0, 2.5, 0.7, 0.35)],
        "arcs": [("Field_A","Irrigate"),("Irrigate","A_Done"),("Pump","Irrigate"),("Irrigate","Pump"),
                 ("Field_B","Fertilize"),("Fertilize","B_Done"),("Pump","Fertilize"),("Fertilize","Pump"),
                 ("Field_C","ScoutSpray"),("ScoutSpray","C_Done"),("Drone","ScoutSpray"),("ScoutSpray","Drone"),
                 ("A_Done","Harvest"),("B_Done","Harvest"),("C_Done","Harvest"),
                 ("Drone","Harvest"),("Harvest","Drone")],
    }
    states = [
        {"desc":"S0: Apr 1 — 3 fields, GDD=0, pump+drone idle",
         "marking":{"Field_A":1,"Field_B":1,"Field_C":1,"Pump":1,"Drone":1,"A_Done":0,"B_Done":0,"C_Done":0},
         "firing":None,"enabled":["Irrigate","Fertilize","ScoutSpray"],
         "condition":"All enabled: each field has token, resources available"},
        {"desc":"S1: Apr 12 (GDD>200) — Irrigate A fires, pump locked",
         "marking":{"Field_A":0,"Field_B":1,"Field_C":1,"Pump":0,"Drone":1,"A_Done":0,"B_Done":0,"C_Done":0},
         "firing":"Irrigate","enabled":["Fertilize","ScoutSpray"],
         "condition":"Irrigate: Field_A≥1,Pump≥1. Fertilize: Field_B≥1,Pump≥1 but Pump=0 now"},
        {"desc":"S2: Apr 24 (GDD>400) — Fertilize B fires, ScoutSpray C fires",
         "marking":{"Field_A":0,"Field_B":0,"Field_C":0,"Pump":0,"Drone":0,"A_Done":1,"B_Done":0,"C_Done":0},
         "firing":"Fertilize","enabled":["ScoutSpray"],
         "condition":"Fertilize: Field_B≥1,Pump≥1. ScoutSpray: Field_C≥1,Drone≥1 (parallel!)"},
        {"desc":"S3: May 4 (GDD>600) — A,B,C all progressing, pump+drone busy",
         "marking":{"Field_A":0,"Field_B":0,"Field_C":0,"Pump":0,"Drone":0,"A_Done":1,"B_Done":1,"C_Done":0},
         "firing":None,"enabled":[],
         "condition":"No enabled: Pump=0,Drone=0 (both busy). Wait for resources."},
        {"desc":"S4: May 14 (GDD>800) — 2nd Irrigate A, pump reused",
         "marking":{"Field_A":0,"Field_B":0,"Field_C":0,"Pump":0,"Drone":1,"A_Done":1,"B_Done":1,"C_Done":1},
         "firing":None,"enabled":["Harvest"],
         "condition":"Harvest enabled: A_Done≥1,B_Done≥1,C_Done≥1,Drone≥1"},
        {"desc":"S5: May 23 (GDD>1000) — Harvest fires, drone collects all",
         "marking":{"Field_A":0,"Field_B":0,"Field_C":0,"Pump":1,"Drone":0,"A_Done":0,"B_Done":0,"C_Done":0},
         "firing":"Harvest","enabled":[],
         "condition":"Harvest: A_Done≥1,B_Done≥1,C_Done≥1,Drone≥1 (AND-join!)"},
        {"desc":"S6: Jun 1 — Season complete, all resources idle",
         "marking":{"Field_A":0,"Field_B":0,"Field_C":0,"Pump":1,"Drone":1,"A_Done":0,"B_Done":0,"C_Done":0},
         "firing":None,"enabled":[],
         "condition":"Deadlock: all GDD stages complete, season over"},
    ]
    draw_petri_execution_states(cpn_spec, states, "fig_D_petri.png", figsize=(12, 8))

    return {"saving_pct":round(saving/baseline["total"]*100,1)}


# ═══════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    print("="*65); print(f"  Nature CPN Figures v5 — {VERSION}"); print("="*65)
    rA=run_scheme_A(); rB=run_scheme_B(); rC=run_scheme_C(); rD=run_scheme_D()
    print(f"\n  Scheme A: {rA['saving_pct']}%  B: {rB['saving_pct']}%  C: {rC['saving_pct']}%  D: {rD['saving_pct']}%")
    print(f"\n  Published: {PAPER_FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)): print(f"    {f}")
    print(f"\n  Data: {DATA_DIR}/")
    for f in sorted(os.listdir(DATA_DIR)): print(f"    {f}")
    print(f"  Version: {VERSION_DIR}/")
    print(f"{'='*65}")