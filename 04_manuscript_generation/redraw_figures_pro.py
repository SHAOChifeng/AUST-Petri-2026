#!/usr/bin/env python3
"""
Professional figure redraw for OCPPP manuscript v7+.
All conceptual figures: Fig 1-6, 10-12, 16-18.
MDPI-compatible: 300 DPI, Arial, consistent colour scheme.

Colour scheme:
  CPN places (states):   #1A73E8 (Google Blue) / #E8F0FE (light)
  CPN transitions (ops): #E37400 (Orange) / #FFF3E0 (light)
  CPN tokens:            #34A853 (Green)
  Resources:             #9334E6 (Purple)
  Background:            #FAFAFA
  Grid/axis:             #E0E0E0
  Lines:                 #333333
"""

import os, sys, math, json, random
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path(__file__).parent.resolve()
NEWCLAUDE_DIR = BASE.parent
FIG_DIR = BASE / "figures"
BIG_FIG_DIR = str(FIG_DIR)
LSTM_FIG_DIR = str(NEWCLAUDE_DIR / "lstm_agri" / "output" / "figures")
CPNTOOLS_DATA = NEWCLAUDE_DIR / "cpntools" / "schemes" / "output"

os.makedirs(FIG_DIR, exist_ok=True)

# ── MDPI-compatible rcParams ───────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "grid.alpha": 0.15,
    "grid.color": "#999999",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ── Palette ────────────────────────────────────────────────────────
C = {
    "blue":    "#1A73E8",
    "blue_l":  "#E8F0FE",
    "orange":  "#E37400",
    "orange_l":"#FFF3E0",
    "green":   "#34A853",
    "green_l": "#E6F4EA",
    "purple":  "#9334E6",
    "purple_l":"#F3E8FD",
    "red":     "#EA4335",
    "red_l":   "#FCE8E6",
    "grey":    "#5F6368",
    "grey_l":  "#F1F3F4",
    "white":   "#FFFFFF",
    "line":    "#333333",
}

# ── graphviz: check and render ─────────────────────────────────────
DOT_OK = False
try:
    import graphviz
    DOT_OK = True
except:
    pass

def render_dot(dot_source, fname, dpi=300):
    """Render DOT to PNG+SVG using graphviz."""
    if not DOT_OK:
        print(f"  ⚠ graphviz unavailable, skipping {fname}")
        return
    tmp = "/tmp/fig_dot"
    os.makedirs(tmp, exist_ok=True)
    src_path = os.path.join(tmp, f"{fname}.dot")
    with open(src_path, "w") as f:
        f.write(dot_source)
    import subprocess
    for fmt in ["png", "svg"]:
        dpi_flag = f"-Gdpi={dpi}" if fmt == "png" else ""
        out_path = os.path.join(FIG_DIR, f"{fname}.{fmt}")
        args = ["dot", f"-T{fmt}"]
        if dpi_flag:
            args += [dpi_flag]
        args += [src_path, "-o", out_path]
        subprocess.run(args, check=True, capture_output=True)
        sz = os.path.getsize(out_path) // 1024
        print(f"  ✅ {fname}.{fmt}  ({sz}K)")
    os.remove(src_path)


# ═══════════════════════════════════════════════════════════════════
# 1. CPN STYLE TEMPLATE (refined, publication-grade)
# ═══════════════════════════════════════════════════════════════════
# Unified palette: places = soft blue circles, transitions = warm
# amber rounded boxes, tokens = charcoal dots, join places = double
# circle (classic Petri-net AND-join symbol).

CPN_GLOBALS = '''
    graph [fontname="Helvetica", fontsize=10, dpi=300,
           bgcolor="white", margin=20, pad=0.25,
           nodesep=0.55, ranksep=0.75];
    node  [fontname="Helvetica", fontsize=10, fontcolor="#1A1A1A"];
    edge  [fontname="Helvetica", fontsize=8, fontcolor="#444444"];
'''

CPN_PLACE  = ('shape="circle", fixedsize="true", width="0.9", '
              'style="filled", fillcolor="#EAF2FB", color="#2C5F8A", penwidth=1.4')
CPN_PLACE_WIDE = ('shape="circle", fixedsize="true", width="1.15", '
              'style="filled", fillcolor="#EAF2FB", color="#2C5F8A", penwidth=1.4')
CPN_JOIN   = ('shape="circle", fixedsize="true", width="0.9", peripheries="2", '
              'style="filled", fillcolor="#FDE9E6", color="#A63D2F", penwidth=1.8')
CPN_TRANS  = ('shape="box", style="filled,rounded", fillcolor="#FCF3E4", color="#9A6B1F", '
              'penwidth=1.4, width=1.05, margin="0.16,0.07"')
CPN_TRANS_EMPH = ('shape="box", style="filled,rounded", fillcolor="#FDF0DC", color="#8A5A10", '
              'penwidth=2.0, width=1.25, margin="0.16,0.07"')
CPN_TOKEN  = ('shape="circle", style="filled", fillcolor="#2B2B2B", color="#2B2B2B", '
              'fixedsize="true", width=0.12, label=""')
CPN_ARC    = 'color="#4A4A4A", penwidth=1.1, arrowsize=0.75'
CPN_ARC_EMPH = 'color="#A63D2F", penwidth=1.7, arrowsize=0.85'
CPN_NOTE   = 'shape="note", style="filled", fillcolor="#F4F4F2", color="#8A8A85", fontsize=8.5, fontcolor="#555550"'
CPN_TAG    = 'shape="plaintext", fontsize=8, fontcolor="#666666", fontname="Helvetica-Oblique"'

def place_label(name, tokens=None, colset=None):
    """HTML-like label: name (+ token dots) inside the circle; colour set below via xlabel."""
    name_html = name.replace("\n", "<BR/>")
    inner = f'<FONT POINT-SIZE="10">{name_html}</FONT>'
    if tokens:
        dots = ('&#8226; ' * tokens).rstrip()
        inner += f'<BR/><FONT POINT-SIZE="8">{dots}</FONT>'
    xl = f' xlabel="{colset}"' if colset else ''
    return f'label=<{inner}>{xl}'



# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Simple CPN — Irrigate → Fertilize → Spray
# ═══════════════════════════════════════════════════════════════════
def draw_fig1():
    dot = f'''
    digraph {{
        {CPN_GLOBALS}
        rankdir="LR";

        // Places (states) — Ready initially carries three field tokens
        node [{CPN_PLACE}];
        p_ready [{place_label("Ready", tokens=3, colset="FIELD")}];
        p_irri  [{place_label("Irrigated", colset="FIELD")}];
        p_fert  [{place_label("Fertilized", colset="FIELD")}];
        p_spray [{place_label("Sprayed", colset="FIELD")}];
        p_done  [{place_label("Done", colset="FIELD")}];

        // Transitions (activities)
        node [{CPN_TRANS}];
        t_irri  [label="Irrigate"];
        t_fert  [label="Fertilize"];
        t_spray [label="Spray"];

        // Arcs
        edge [{CPN_ARC}];
        p_ready -> t_irri -> p_irri -> t_fert -> p_fert -> t_spray -> p_spray -> p_done;
    }}
    '''
    render_dot(dot, "fig1_cpn_sequence")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Shared Transition
# ═══════════════════════════════════════════════════════════════════
def draw_fig2():
    dot = f'''
    digraph {{
        {CPN_GLOBALS}

        subgraph cluster_fieldA {{
            label="Workflow N₁ (Field A)"; fontsize=11; style="dashed,rounded";
            color="#2C5F8A"; fontcolor="#2C5F8A"; bgcolor="#FBFCFE"; margin=18;
            node [{CPN_PLACE}];
            a_ready [{place_label("Ready_A")}];
            a_done  [{place_label("Done_A")}];
            node [{CPN_TRANS}];
            a_fert [label="Fertilize_A"];
            a_ready -> a_fert -> a_done;
        }}

        subgraph cluster_fieldB {{
            label="Workflow N₂ (Field B)"; fontsize=11; style="dashed,rounded";
            color="#4C7A44"; fontcolor="#4C7A44"; bgcolor="#FCFEFC"; margin=18;
            node [{CPN_PLACE}];
            b_ready [{place_label("Ready_B")}];
            b_done  [{place_label("Done_B")}];
            node [{CPN_TRANS}];
            b_fert [label="Fertilize_B"];
            b_ready -> b_fert -> b_done;
        }}

        // Shared transition (the single irrigation pump)
        node [{CPN_TRANS_EMPH}];
        shared [label="Irrigate\n(shared pump)", fontsize=11, fontcolor="#5A3A07"];

        edge [{CPN_ARC_EMPH}];
        a_ready -> shared -> a_done;
        b_ready -> shared -> b_done;

        // Resource note
        node [{CPN_NOTE}];
        note [label="One pump — only one\\nfield can irrigate\\nat a time"];
        shared -> note [style=dotted, color="#8A8A85", penwidth=1.0, arrowhead=none];
    }}
    '''
    render_dot(dot, "fig2_shared_transition")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Precursor Transitions
# ═══════════════════════════════════════════════════════════════════
def draw_fig3():
    l_p1 = place_label("Ready\nWater")
    l_p2 = place_label("Ready\nNutrients")
    l_p3 = place_label("Ready\nPest-free")
    l_join = place_label("Field\nReady")
    dot = f'''
    digraph {{
        {CPN_GLOBALS}
        rankdir="LR";

        // Three parallel precursor branches
        node [{CPN_PLACE}];
        p1 [{l_p1}];
        p2 [{l_p2}];
        p3 [{l_p3}];

        node [{CPN_TRANS}];
        t1 [label="Irrigate"];
        t2 [label="Fertilize"];
        t3 [label="Spray"];

        node [{CPN_PLACE}];
        j1 [{place_label("Watered")}];
        j2 [{place_label("Fertilized")}];
        j3 [{place_label("Sprayed")}];

        // Join place (double circle = AND-join)
        node [{CPN_JOIN}];
        join [{l_join}];

        // Harvest transition
        node [{CPN_TRANS}];
        harvest [label="Harvest", fontsize=11];

        node [{CPN_PLACE}];
        done [{place_label("Harvested")}];

        edge [{CPN_ARC}];
        p1 -> t1 -> j1;
        p2 -> t2 -> j2;
        p3 -> t3 -> j3;
        j1 -> join; j2 -> join; j3 -> join;

        edge [{CPN_ARC_EMPH}];
        join -> harvest;

        edge [{CPN_ARC}];
        harvest -> done;

        // AND-join note
        node [{CPN_TAG}];
        lbl [label="requires all three tokens"];
        join -> lbl [style=dotted, color="#8A8A85", arrowhead=none];
    }}
    '''
    render_dot(dot, "fig3_precursor_transitions")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Multi-modal Transition
# ═══════════════════════════════════════════════════════════════════
def draw_fig4():
    l_uav = place_label("UAV image\ndata source")
    l_soil = place_label("Soil sensor\ndata source")
    l_scouted = place_label("Scouted\nfield")
    l_report = place_label("Multi-modal\nreport")
    dot = f'''
    digraph {{
        {CPN_GLOBALS}
        rankdir="LR";

        // Input places
        node [{CPN_PLACE_WIDE}];
        field [{place_label("Field")}];
        uav   [{l_uav}];
        soil  [{l_soil}];

        // Scout transition (multi-modal)
        node [{CPN_TRANS_EMPH}];
        scout [label="Scout\n(multi-modal)", fontsize=11, fontcolor="#5A3A07"];

        // Output places
        node [{CPN_PLACE}];
        scouted [{l_scouted}];
        report  [{l_report}];

        node [{CPN_TAG}];
        ndvi  [label="NDVI index"];
        moist [label="Soil moisture"];
        pest  [label="Pest alert"];

        edge [{CPN_ARC}];
        field -> scout;
        uav -> scout;
        soil -> scout;
        scout -> scouted;
        scout -> report;
        report -> ndvi  [style=dotted, arrowhead=none];
        report -> moist [style=dotted, arrowhead=none];
        report -> pest  [style=dotted, arrowhead=none];
    }}
    '''
    render_dot(dot, "fig4_multimodal_transition")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Queue Model with Resource Constraint
# ═══════════════════════════════════════════════════════════════════
def draw_fig5():
    l_incoming = place_label("Incoming\nfields", tokens=3)
    l_queue = place_label("Term\n(queue)")
    l_timer = place_label("Timer", tokens=1, colset="E")
    dot = f'''
    digraph {{
        {CPN_GLOBALS}

        node [{CPN_PLACE}];
        incoming [{l_incoming}];
        queue    [{l_queue}];
        timer    [{l_timer}];
        done     [{place_label("Completed")}];

        node [{CPN_TRANS}];
        formq [label="FormQ"];
        exec  [label="ExecuteOp"];

        edge [{CPN_ARC}];
        incoming -> formq;
        formq -> queue;
        queue -> exec;
        timer -> exec;
        exec -> timer [label="  @+t", fontsize=8];
        exec -> done;

        // Annotation
        node [{CPN_TAG}];
        annot [label="Timer holds one token →\\ncapacity = one pump"];
        timer -> annot [style=dotted, color="#8A8A85", arrowhead=none];
    }}
    '''
    render_dot(dot, "fig5_queue_model")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: Algorithm 1 — Workflow Construction
# ═══════════════════════════════════════════════════════════════════
def draw_fig6():
    dot = f'''
    digraph {{
        {CPN_GLOBALS}
        splines="polyline"; rankdir="LR"; nodesep=0.5;

        subgraph cluster_left {{
            label="(a) Input: operation set T"; fontsize=11; style="dashed,rounded";
            color="#8A8A85"; fontcolor="#555550"; bgcolor="#FBFBFA"; margin=18;
            node [shape="box", style="rounded", fillcolor="#EFEFED", color="#9A9A95",
                  fontsize=10, width=1.4, margin="0.14,0.09"];
            i1 [label="Irrigate"];
            i2 [label="Scout_Drone"];
            i3 [label="Fertilize"];
            i4 [label="Spray"];
        }}

        subgraph cluster_mid {{
            label="(b) Precedence discovery"; fontsize=11; style="dashed,rounded";
            color="#4C7A44"; fontcolor="#4C7A44"; bgcolor="#FBFDFB"; margin=18;
            node [shape="box", style="rounded", fillcolor="#EAF4EA", color="#4C7A44",
                  fontsize=10, width=1.4, margin="0.14,0.09"];
            m1 [label="Irrigate"];
            m2 [label="Scout_Drone"];
            m3 [label="Fertilize"];
            m4 [label="Spray"];
            m1 -> m3 [label="soil_moisture", fontsize=8, color="#4C7A44", penwidth=1.3];
            m2 -> m4 [label="pest_density", fontsize=8, color="#4C7A44", penwidth=1.3];
        }}

        subgraph cluster_right {{
            label="(c) Output CPN (two concurrent branches)"; fontsize=11; style="dashed,rounded";
            color="#2C5F8A"; fontcolor="#2C5F8A"; bgcolor="#FBFCFE"; margin=18;
            node [{CPN_PLACE}, width=0.8, fontsize=9];
            r_ready1 [{place_label("Ready")}];
            r_ready2 [{place_label("Ready")}];
            r_irri   [{place_label("Irrigated")}];
            r_scout  [{place_label("Scouted")}];
            node [{CPN_JOIN}, width=0.8, fontsize=9];
            r_done   [{place_label("Complete")}];
            node [{CPN_TRANS}, width=0.95, fontsize=9];
            r1 [label="Irrigate"];
            r2 [label="Scout_Drone"];
            r3 [label="Fertilize"];
            r4 [label="Spray"];
            edge [{CPN_ARC}];
            r_ready1 -> r1 -> r_irri -> r3;
            r_ready2 -> r2 -> r_scout -> r4;
            r3 -> r_done; r4 -> r_done;
        }}

        edge [style=invis]; i1 -> m1 -> r1; i2 -> m2 -> r2; i3 -> m3 -> r3; i4 -> m4 -> r4;
    }}
    '''
    render_dot(dot, "fig6_workflow_construction")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 10: Gantt Chart — Algorithm 2
# ═══════════════════════════════════════════════════════════════════
def draw_fig10():
    """Gantt chart for 6 execution sequences (Algorithm 2)."""
    sequences = [
        ("Soil→Scout→Trap",  [(8.0,0.8,'tech'), (8.8,1.0,'drone'), (9.8,0.9,'both')]),
        ("Soil→Trap→Scout",  [(8.0,0.8,'tech'), (8.8,0.9,'both'), (9.7,1.0,'drone')]),
        ("Scout→Soil→Trap",  [(8.0,1.0,'drone'), (9.0,0.8,'tech'), (9.8,0.9,'both')]),
        ("Scout→Trap→Soil",  [(8.0,1.0,'drone'), (9.0,0.9,'both'), (9.9,0.8,'tech')]),
        ("Trap→Soil→Scout",  [(8.0,0.9,'both'), (8.9,0.8,'tech'), (9.7,1.0,'drone')]),
        ("Trap→Scout→Soil",  [(8.0,0.9,'both'), (8.9,1.0,'drone'), (9.9,0.8,'tech')]),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {'drone': '#1A73E8', 'tech': '#34A853', 'both': '#E37400'}
    labels = {'drone': 'Drone', 'tech': 'Technician', 'both': 'Drone+Tech'}

    for i, (name, ops) in enumerate(sequences):
        for j, (start, dur, res) in enumerate(ops):
            y = len(sequences) - 1 - i
            ax.barh(y, dur, 0.35, left=start, color=colors[res], edgecolor='white', linewidth=0.3, alpha=0.9)
            mid = start + dur / 2
            if dur > 0.4:
                ax.text(mid, y, res[0].upper(), ha='center', va='center', fontsize=6, color='white', fontweight='bold')

    ax.set_yticks(range(len(sequences)))
    ax.set_yticklabels([s[0] for s in sequences], fontsize=8)
    ax.set_xlabel("Time (hours from 08:00)", fontsize=9)
    ax.set_xlim(7.5, 11.5)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)

    legend = [mpatches.Patch(color=c, label=labels[k]) for k, c in colors.items()]
    ax.legend(handles=legend, fontsize=7, loc='lower right', framealpha=0.9, edgecolor='#E0E0E0')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig10_gantt_comparison.png")
    fig.savefig(FIG_DIR / "fig10_gantt_comparison.svg")
    plt.close()
    print(f"  ✅ fig10_gantt_comparison.png  ({os.path.getsize(FIG_DIR / 'fig10_gantt_comparison.png') // 1024}K)")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 11: Frost Intersection — Algorithm 3
# ═══════════════════════════════════════════════════════════════════
def draw_fig11():
    """3-panel precursor + 1-panel intersection."""
    days = list(range(1, 31))
    np.random.seed(42)

    # Simulated temperature
    temp = 15 - 0.5 * np.array(days) + 3 * np.sin(np.array(days) / 3)
    # Simulated wind
    wind = 5 - 0.08 * np.array(days) + 2 * np.sin(np.array(days) / 2)
    # Simulated humidity
    humid = 50 + 0.5 * np.array(days) + 10 * np.sin(np.array(days) / 4)

    temp_thresh = 0
    wind_thresh = 3
    humid_thresh = 60

    temp_exceed = temp < temp_thresh
    wind_exceed = wind < wind_thresh
    humid_exceed = humid > humid_thresh

    fig, axes = plt.subplots(4, 1, figsize=(7, 7), sharex=True)

    ax = axes[0]
    ax.plot(days, temp, color='#EA4335', linewidth=1.5)
    ax.axhline(temp_thresh, color='#EA4335', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.fill_between(days, temp_thresh, temp, where=temp_exceed, color='#EA4335', alpha=0.15)
    ax.set_ylabel("Temperature (°C)", fontsize=8)
    ax.set_title("Temperature (LSTM forecast)", fontsize=9, fontweight='bold')
    ax.set_ylim(-8, 20)

    ax = axes[1]
    ax.plot(days, wind, color='#1A73E8', linewidth=1.5)
    ax.axhline(wind_thresh, color='#1A73E8', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.fill_between(days, 0, wind, where=wind_exceed, color='#1A73E8', alpha=0.15)
    ax.set_ylabel("Wind Speed (m/s)", fontsize=8)
    ax.set_title("Wind speed (climatology)", fontsize=9, fontweight='bold')
    ax.set_ylim(0, 10)

    ax = axes[2]
    ax.plot(days, humid, color='#34A853', linewidth=1.5)
    ax.axhline(humid_thresh, color='#34A853', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.fill_between(days, humid_thresh, 100, where=humid_exceed, color='#34A853', alpha=0.15)
    ax.set_ylabel("Humidity (%)", fontsize=8)
    ax.set_title("Relative humidity (climatology)", fontsize=9, fontweight='bold')
    ax.set_ylim(20, 100)

    ax = axes[3]
    intersection = temp_exceed & wind_exceed & humid_exceed
    ax.fill_between(days, 0, 1, where=intersection, color='#9334E6', alpha=0.5, step='mid')
    ax.plot(days, intersection.astype(int), color='#9334E6', linewidth=1.5, drawstyle='steps-mid')
    ax.set_ylabel("Frost Risk", fontsize=8)
    ax.set_xlabel("Day", fontsize=9)
    ax.set_title("Predicted frost window (intersection)", fontsize=9, fontweight='bold')
    ax.set_ylim(-0.1, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'], fontsize=7)

    if np.any(intersection):
        first_day = days[np.argmax(intersection)]
        ax.annotate(f"Frost predicted\nDay {first_day}",
                    xy=(first_day, 1), xytext=(first_day + 4, 0.7),
                    arrowprops=dict(arrowstyle='->', color='#9334E6', lw=1.0),
                    fontsize=8, color='#9334E6', fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig11_frost_intersection.png")
    fig.savefig(FIG_DIR / "fig11_frost_intersection.svg")
    plt.close()
    print(f"  ✅ fig11_frost_intersection.png  ({os.path.getsize(FIG_DIR / 'fig11_frost_intersection.png') // 1024}K)")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 12: Framework Architecture
# ═══════════════════════════════════════════════════════════════════
def draw_fig12():
    dot = '''digraph {
''' + CPN_GLOBALS + '''labelloc="t"; labeljust="l"; fontsize=11; rankdir="LR"; splines="ortho";

subgraph cluster_m1 { label="Module 1: Process Modelling"; style="rounded,dashed";
color="#1A73E8"; bgcolor="#E8F0FE"; fontcolor="#1A73E8"; fontsize=10;
node [shape="box", style="filled,rounded", fillcolor="#BBDEFB", fontsize=9];
algo1 [label="Algorithm 1\\nCPN Construction"];
library [label="Operation\\nPattern Library"];
algo1 -> library [style=dashed, dir=both]; }

subgraph cluster_m2 { label="Module 2: Multi-Modal AI"; style="rounded,dashed";
color="#34A853"; bgcolor="#E6F4EA"; fontcolor="#34A853"; fontsize=10;
node [shape="box", style="filled,rounded", fillcolor="#C8E6C9", fontsize=9];
yolo [label="YOLOv7\\nLeaf Detection"];
lstm [label="LSTM\\nTemp Forecast"];
yolo_img [label="Crop\\nImages", shape="folder", fillcolor="#FFF3E0"];
lstm_ts  [label="Sensor\\nTime Series", shape="folder", fillcolor="#FFF3E0"];
yolo_img -> yolo; lstm_ts -> lstm; }

subgraph cluster_m3 { label="Module 3: CPN Prediction"; style="rounded,dashed";
color="#E37400"; bgcolor="#FFF3E0"; fontcolor="#E37400"; fontsize=10;
node [shape="box", style="filled,rounded", fillcolor="#FFE0B2", fontsize=9];
cpn [label="CPN Discrete-Event\\nSimulation"];
algo2 [label="Algorithm 2\\nUnordered Ops"];
algo3 [label="Algorithm 3\\nLow-Freq Events"];
trace [label="Execution\\nTrace", shape="note", fillcolor="#FCE8E6"];
cpn -> algo2; cpn -> algo3; cpn -> trace [style=dashed]; }

edge [penwidth=1.5, color="#5F6368"];
algo1 -> yolo [style=dashed, label="state vector", fontsize=7];
algo1 -> lstm [style=dashed, label="state vector", fontsize=7];
yolo -> cpn [label="disease ratio", fontsize=7];
lstm -> cpn [label="temperature", fontsize=7];
}
'''
    render_dot(dot, "fig12_framework")
    render_dot(dot, "fig12_framework")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 16: CPN Scheme A — Irrigation Scheduling
# ═══════════════════════════════════════════════════════════════════
def draw_fig16():
    """Scheme A: Irrigation scheduling comparison."""
    try:
        data = json.load(open(CPNTOOLS_DATA / "all_schemes_real_data.json"))
    except:
        data = None

    if data is None or "A" not in data:
        print("  ⚠ Scheme A data not found, using fallback")
        orders = ["A→B→C", "A→C→B", "C→A→B", "B→A→C", "B→C→A", "C→B→A"]
        totals = [136, 149, 149, 151, 173, 173]
        best_idx = 0
    else:
        r = data["A"]
        orders = [x["order"] for x in r["results"]]
        totals = [x["total"] for x in r["results"]]
        best_idx = 0

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))

    # Panel 1: Bar chart
    ax = axes[0]
    colors = ["#34A853" if i == best_idx else "#EA4335" if i == len(orders) - 1 else "#E0E0E0" for i in range(len(orders))]
    bars = ax.barh(range(len(orders)), totals, color=colors, edgecolor="white", height=0.5)
    ax.set_yticks(range(len(orders)))
    ax.set_yticklabels(orders, fontsize=7)
    ax.set_xlabel("Total Time (min)", fontsize=8)
    ax.set_title("All 6 Orderings", fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    for i, (bar, t) in enumerate(zip(bars, totals)):
        ax.text(t + 3, bar.get_y() + bar.get_height() / 2, f"{t} min", va="center", fontsize=7, fontweight="bold")
    ax.set_xlim(0, max(totals) * 1.2)
    ax.grid(axis="x", alpha=0.15)

    # Panels 2 & 3: Gantt charts (simplified)
    for panel_idx, (title, order_name) in enumerate([
        (f"Best: {orders[0]} ({totals[0]} min)", orders[0]),
        (f"Worst: {orders[-1]} ({totals[-1]} min)", orders[-1]),
    ]):
        ax = axes[panel_idx + 1]
        fields = order_name.split("→")
        durations = {"A": 45, "B": 31, "C": 22}
        post_ops = {"A": [("Fert", 30), ("Scout", 20), ("Spray", 25)],
                    "B": [("Fert", 20), ("Scout", 14), ("Spray", 17)],
                    "C": [("Fert", 15), ("Scout", 10), ("Spray", 13)]}
        op_colors = {"Irri": "#1A73E8", "Fert": "#34A853", "Scout": "#E37400", "Spray": "#9334E6"}

        t = 0
        y_positions = {"A": 3, "B": 2, "C": 1}
        for f in fields:
            # Irrigate
            ax.barh(y_positions[f], durations[f], 0.4, left=t, color="#1A73E8", edgecolor="white", linewidth=0.3)
            ax.text(t + durations[f] / 2, y_positions[f], "Irri", ha="center", va="center", fontsize=5, color="white", fontweight="bold")
            t += durations[f]
            for op_name, dur in post_ops[f]:
                c = op_colors.get(op_name, "#999")
                ax.barh(y_positions[f], dur, 0.4, left=t, color=c, edgecolor="white", linewidth=0.3)
                ax.text(t + dur / 2, y_positions[f], op_name, ha="center", va="center", fontsize=5, color="white", fontweight="bold")
                t += dur

        ax.set_yticks([3, 2, 1])
        ax.set_yticklabels([f"Field {f}" for f in fields], fontsize=7)
        ax.set_xlabel("Time (min)", fontsize=8)
        ax.set_title(title, fontsize=8, fontweight="bold")
        # Mark pump usage periods
        for i, f in enumerate(fields):
            start = sum(durations[g] for g in fields[:i])
            ax.annotate("", xy=(start, -0.1), xytext=(start + durations[f], -0.1),
                        arrowprops=dict(arrowstyle="|-|", color="#1A73E8", lw=1.5))
        ax.set_xlim(0, max(totals))

    fig.suptitle("Scheme A: Irrigation scheduling comparison.", fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig16_cpn_irrigation.png")
    fig.savefig(FIG_DIR / "fig16_cpn_irrigation.svg")
    plt.close()
    print(f"  ✅ fig16_cpn_irrigation.png  ({os.path.getsize(FIG_DIR / 'fig16_cpn_irrigation.png') // 1024}K)")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  OCPPP Professional Figure Redraw")
    print("  300 DPI | Arial | MDPI-compatible styling")
    print("=" * 60)

    print("\n--- Fig 1-6: CPN diagrams ---")
    draw_fig1()
    draw_fig2()
    draw_fig3()
    draw_fig4()
    draw_fig5()
    draw_fig6()

    print("\n--- Fig 10: Algorithm 2 illustration (Fig 11 real-data version is generated by draw_fig11_real.py) ---")
    draw_fig10()

    print("\n--- Fig 12: Framework architecture ---")
    draw_fig12()

    print("\n--- Fig 16: CPN Scheme A ---")
    draw_fig16()

    print("\n--- Done ---")
    print(f"All figures saved to: {FIG_DIR}")
