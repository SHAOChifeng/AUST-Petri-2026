#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 两类别图重生成脚本(PlantVillage:healthy_leaf / diseased_leaf)
============================================================

用途:基于"重新标注后的 2 类数据集 + 2026-07-18 训练好的 50% 子集权重",
      在【测试集】上重新生成论文用图:

  v2.0/figures/S19_yolo_class_dist.png   类别分布(健康叶 vs 病叶)
  v2.0/figures/S20_yolo_conf_hist.png    置信度分布
  v2.0/figures/S21_yolo_det_count.png    每图检测数分布
  v2.0/figures/S22_yolo_health_ratio.png 每图病害比例(病害严重度估计依据)
  v2.0/figures/fig14_yolo_detection.png  Figure 14 检测样例(4x3 网格)

  并顺带输出:
  output/detection_stats_2class.csv      全量测试集逐图检测统计
  output/regenerate_2class_report.txt    汇总报告(含测试集图片数,
                                          用于核对论文 5.1.2 的 10,860)

注意:
  1) 本脚本【不做训练】,只做 detect/统计/绘图,论文中的验证集数字
     (mAP@0.5=0.986, mAP@0.5:0.95=0.965)不受影响。
  2) S17/S18/S23(PR、F1、混淆矩阵)已直接取自今天 val 运行的
     runs/test/exp,无需重跑;若该目录被清理,脚本会自动重跑 test.py。
  3) 运行环境:conda activate Agronomy,在 yolov7_agri 目录下执行:
       python regenerate_2class_figures.py
"""

import os, sys, csv, shutil, subprocess, random
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ===================== 路径 =====================
PROJECT_DIR = Path(__file__).parent.resolve()
YOLOV7_DIR = PROJECT_DIR / "yolov7_official"
DATASET_YAML = PROJECT_DIR / "cfg" / "plantvillage.yaml"
TEST_IMG_DIR = Path("C:/SCIData/Agronomy/yolo_dataset/images/test")
TEST_LBL_DIR = Path("C:/SCIData/Agronomy/yolo_dataset/labels/test")
OUTPUT_DIR = PROJECT_DIR / "output"
PAPER_FIG_DIR = PROJECT_DIR.parent / "v2.0" / "figures"

CLASS_NAMES = {0: "healthy_leaf", 1: "diseased_leaf"}
CLASS_COLORS = {"healthy_leaf": "#2ecc71", "diseased_leaf": "#e74c3c"}

IMG_SIZE = 256          # 与训练/验证一致
CONF_THRES = 0.25
IOU_THRES = 0.45
FIG14_COLS, FIG14_ROWS = 4, 3   # 4x3 网格:左2列健康、右2列病叶


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_weights():
    """优先 7 月 18 日 50% 子集训练出的权重,找不到则取最新的 best_*.pt"""
    cand = OUTPUT_DIR / "weights" / "best_0718_1508.pt"
    if cand.exists():
        return cand
    bests = sorted((OUTPUT_DIR / "weights").glob("best_*.pt"))
    if bests:
        return bests[-1]
    # 兜底:官方训练目录
    bests = sorted((YOLOV7_DIR / "runs" / "train").glob("*/weights/best.pt"))
    if bests:
        return bests[-1]
    raise FileNotFoundError("未找到权重,请确认 output/weights/ 下有 best_*.pt")


def run_detect(source_dir, out_name, save_img):
    """调用 YOLOv7 detect.py"""
    weights = find_weights()
    save_dir = YOLOV7_DIR / "runs" / "detect_2class" / out_name
    cmd = [
        sys.executable, str(YOLOV7_DIR / "detect.py"),
        "--weights", str(weights),
        "--source", str(source_dir),
        "--img-size", str(IMG_SIZE),
        "--conf-thres", str(CONF_THRES),
        "--iou-thres", str(IOU_THRES),
        "--device", "0",
        "--project", str(YOLOV7_DIR / "runs" / "detect_2class"),
        "--name", out_name,
        "--exist-ok",
        "--save-txt", "--save-conf",
    ]
    if not save_img:
        cmd.append("--nosave")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log(f"detect.py -> {save_dir}")
    r = subprocess.run(cmd, cwd=str(YOLOV7_DIR), capture_output=True,
                       text=True, errors="replace", env=env, timeout=3600)
    if r.returncode != 0:
        log("detect 失败:\n" + r.stderr[-800:])
        sys.exit(1)
    log("detect 完成")
    return save_dir


def collect_stats(detect_dir):
    """解析 detect 输出的 labels/*.txt,得到每图统计"""
    labels_dir = detect_dir / "labels"
    if not labels_dir.exists():
        log(f"未找到 labels 目录: {labels_dir}")
        sys.exit(1)
    stats = []
    per_class = {n: {"count": 0, "confs": []} for n in CLASS_NAMES.values()}
    for lf in sorted(labels_dir.glob("*.txt")):
        dets = []
        with open(lf) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 6:
                    cls_id = int(float(p[0]))
                    conf = float(p[5])
                    if cls_id in CLASS_NAMES:
                        name = CLASS_NAMES[cls_id]
                        dets.append((name, conf))
                        per_class[name]["count"] += 1
                        per_class[name]["confs"].append(conf)
        n_h = sum(1 for d in dets if d[0] == "healthy_leaf")
        n_d = sum(1 for d in dets if d[0] == "diseased_leaf")
        stats.append({
            "file": lf.stem,
            "total_dets": len(dets),
            "healthy": n_h,
            "diseased": n_d,
            "avg_conf": float(np.mean([d[1] for d in dets])) if dets else 0.0,
        })
    return stats, per_class


def save_report(stats, per_class, n_test_images):
    """写汇总报告"""
    total_det = sum(per_class[c]["count"] for c in CLASS_NAMES.values())
    lines = [
        "YOLOv7 两类别测试集检测统计报告",
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"测试集图片数(有标签文件数): {n_test_images}",
        f"总检测数: {total_det}",
    ]
    for name in CLASS_NAMES.values():
        c = per_class[name]["count"]
        confs = per_class[name]["confs"]
        avg = float(np.mean(confs)) if confs else 0.0
        lines.append(f"{name}: {c} ({c / max(total_det, 1) * 100:.1f}%), 平均置信度 {avg:.4f}")
    overall_confs = [c for name in CLASS_NAMES.values() for c in per_class[name]["confs"]]
    lines.append(f"总体平均置信度: {float(np.mean(overall_confs)):.4f}")
    lines.append("")
    lines.append("注: 论文 5.1.2 中测试集图片数写作 10,860,请用上面的实际图片数核对是否需要修改。")
    lines.append("注: Scheme B (scheme_B.py) 中写死的示意置信度 0.85/0.89,可与上面平均置信度对比,")
    lines.append("    如有明显出入可决定是否顺手更新(不影响 CPN 结论)。")
    report = OUTPUT_DIR / "regenerate_2class_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"报告: {report}")
    print("\n" + "\n".join(lines))


# ===================== 绘图: S19-S22 + Figure 14 =====================

def plot_S19_class_dist(per_class):
    names = list(CLASS_NAMES.values())
    counts = [per_class[n]["count"] for n in names]
    colors = [CLASS_COLORS[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", width=0.55)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.01,
                str(c), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of detections")
    ax.set_title("")
    ax.set_xticklabels(names, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    out = PAPER_FIG_DIR / "S19_yolo_class_dist.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    log(f"S19 -> {out}")


def plot_S20_conf_hist(per_class):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in CLASS_NAMES.values():
        confs = per_class[name]["confs"]
        if confs:
            ax.hist(confs, bins=40, range=(0, 1), alpha=0.55,
                    color=CLASS_COLORS[name], label=name)
    ax.set_xlabel("Detection confidence")
    ax.set_ylabel("Number of detections")
    ax.set_title("")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    out = PAPER_FIG_DIR / "S20_yolo_conf_hist.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    log(f"S20 -> {out}")


def plot_S21_det_count(stats):
    counts = [s["total_dets"] for s in stats]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(counts, bins=range(0, max(counts) + 2), color="steelblue",
            edgecolor="white", alpha=0.85, align="left")
    mean_c = float(np.mean(counts))
    ax.axvline(mean_c, color="red", ls="--", lw=1.5,
               label=f"Mean = {mean_c:.2f}")
    ax.set_xlabel("Detections per image")
    ax.set_ylabel("Number of images")
    ax.set_title("")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    out = PAPER_FIG_DIR / "S21_yolo_det_count.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    log(f"S21 -> {out}")


def plot_S22_health_ratio(stats):
    ratios = []
    for s in stats:
        tot = s["healthy"] + s["diseased"]
        if tot > 0:
            ratios.append(s["diseased"] / tot)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ratios, bins=20, range=(0, 1), color="#8e44ad",
            edgecolor="white", alpha=0.8)
    mean_r = float(np.mean(ratios))
    ax.axvline(mean_r, color="red", ls="--", lw=1.5,
               label=f"Mean diseased ratio = {mean_r:.2f}")
    ax.set_xlabel("Diseased leaf ratio per image (n_diseased / n_leaves)")
    ax.set_ylabel("Number of images")
    ax.set_title("")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    out = PAPER_FIG_DIR / "S22_yolo_health_ratio.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    log(f"S22 -> {out}")


def gt_class(stem):
    """读取测试集真实标签(每图一个框),返回 'healthy_leaf'/'diseased_leaf'/None"""
    lp = TEST_LBL_DIR / f"{stem}.txt"
    if lp.exists():
        try:
            first = lp.read_text().strip().split()[0]
            return CLASS_NAMES.get(int(float(first)))
        except Exception:
            return None
    return None


def plot_fig14(stats, detect_dir):
    """Figure 14: 4x3 检测样例网格,左2列健康、右2列病叶,按真实标签选图"""
    labels_dir = detect_dir / "labels"
    lab_map = {}
    for lf in labels_dir.glob("*.txt"):
        lab_map[lf.stem] = lf.read_text().splitlines()

    healthy_pool = [s for s in stats if gt_class(s["file"]) == "healthy_leaf"
                    and s["file"] in lab_map]
    diseased_pool = [s for s in stats if gt_class(s["file"]) == "diseased_leaf"
                     and s["file"] in lab_map]
    random.seed(42)
    random.shuffle(healthy_pool); random.shuffle(diseased_pool)
    n_each = (FIG14_ROWS * FIG14_COLS) // 2
    if len(healthy_pool) < n_each or len(diseased_pool) < n_each:
        log(f"警告: 按真实标签选图不足(healthy={len(healthy_pool)}, diseased={len(diseased_pool)}),"
            f" 改为按检测结果选图")
        healthy_pool = [s for s in stats if s["healthy"] > 0 and s["diseased"] == 0
                        and s["file"] in lab_map]
        diseased_pool = [s for s in stats if s["diseased"] > 0 and s["healthy"] == 0
                         and s["file"] in lab_map]
        random.shuffle(healthy_pool); random.shuffle(diseased_pool)
    sel_h = healthy_pool[:n_each]
    sel_d = diseased_pool[:n_each]
    if len(sel_h) < n_each or len(sel_d) < n_each:
        log("错误: 可用的检测样例不足,无法生成 Figure 14")
        return
    # 4列布局: 前2列健康、后2列病叶
    selected = []
    for r in range(FIG14_ROWS):
        selected += [sel_h[2 * r], sel_h[2 * r + 1],
                     sel_d[2 * r], sel_d[2 * r + 1]]

    fig, axes = plt.subplots(FIG14_ROWS, FIG14_COLS, figsize=(16, 12))
    for idx, s in enumerate(selected):
        r, c = idx // FIG14_COLS, idx % FIG14_COLS
        ax = axes[r, c]
        stem = s["file"]
        img_path = TEST_IMG_DIR / f"{stem}.jpg"
        if not img_path.exists():
            ax.axis("off"); continue
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        ax.imshow(im)
        for line in lab_map.get(stem, []):
            p = line.split()
            if len(p) < 6:
                continue
            cls_id = int(float(p[0]))
            if cls_id not in CLASS_NAMES:
                continue
            cx, cy, w, h = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
            conf = float(p[5])
            x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
            name = CLASS_NAMES[cls_id]
            ax.add_patch(mpatches.Rectangle((x1, y1), w * W, h * H, fill=False,
                                            ec=CLASS_COLORS[name], lw=2))
            ax.text(x1, y1 - 4, f"{name} {conf:.2f}", color=CLASS_COLORS[name],
                    fontsize=8, va="bottom",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(CLASS_COLORS[gt_class(stem) or "healthy_leaf"])
            sp.set_linewidth(3)
        ax.set_title(f"{gt_class(stem)}", fontsize=10, pad=4)
    out = PAPER_FIG_DIR / "fig14_yolo_detection.jpg"
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out, format="jpg", dpi=150, bbox_inches="tight",
                pil_kwargs={"quality": 90}); plt.close(fig)
    log(f"Figure 14 -> {out}")


# ===================== S17/S18/S23: 直接取自 val 运行 =====================

def copy_val_curves():
    """优先用今天(或最近)val 运行的结果;没有则自动重跑 test.py(val 划分)"""
    exp = YOLOV7_DIR / "runs" / "test" / "exp"
    src_map = {
        "PR_curve.png": PAPER_FIG_DIR / "S17_yolo_pr_curve.png",
        "F1_curve.png": PAPER_FIG_DIR / "S18_yolo_f1_curve.png",
        "confusion_matrix.png": PAPER_FIG_DIR / "S23_yolo_confusion_matrix.png",
    }
    if not exp.exists() or not (exp / "PR_curve.png").exists():
        log("未找到 val 运行结果,自动执行 test.py(验证集划分)...")
        weights = find_weights()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        cmd = [
            sys.executable, str(YOLOV7_DIR / "test.py"),
            "--data", str(DATASET_YAML),
            "--weights", str(weights),
            "--img-size", str(IMG_SIZE),
            "--batch-size", "32",
            "--device", "0",
            "--project", str(YOLOV7_DIR / "runs" / "test_2cls"),
            "--name", "exp",
            "--exist-ok",
        ]
        r = subprocess.run(cmd, cwd=str(YOLOV7_DIR), capture_output=True,
                           text=True, errors="replace", env=env, timeout=3600)
        if r.returncode != 0:
            log("test.py 失败:\n" + r.stderr[-800:])
            return
        exp = YOLOV7_DIR / "runs" / "test_2cls" / "exp"
        log("test.py 完成,验证集指标见上方输出(应与论文 mAP@0.5=0.986 一致)")
    for src_name, dst in src_map.items():
        src = exp / src_name
        if src.exists():
            shutil.copy(src, dst)
            log(f"{src_name} -> {dst.name}")
        else:
            log(f"缺少 {src_name},跳过")


# ===================== 主流程 =====================

def main():
    log("=" * 60)
    log("YOLOv7 两类别图重生成")
    log("=" * 60)

    # 0. 检查
    yaml_text = DATASET_YAML.read_text(encoding="utf-8")
    if "nc: 2" not in yaml_text:
        log(f"警告: {DATASET_YAML} 中 nc != 2,请确认为两类标注!")
        log("本脚本仍继续,但类别名可能不符。")
    if not TEST_IMG_DIR.exists():
        log(f"测试集目录不存在: {TEST_IMG_DIR}")
        sys.exit(1)
    weights = find_weights()
    log(f"权重: {weights}")

    # 1. 全量测试集 detect(只存 txt,不存图)
    detect_dir = run_detect(TEST_IMG_DIR, "exp_stats", save_img=False)
    stats, per_class = collect_stats(detect_dir)
    n_test_images = len(stats)
    log(f"测试集图片数: {n_test_images}")

    # 2. 统计 CSV
    csv_path = OUTPUT_DIR / "detection_stats_2class.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "total_dets", "healthy",
                                          "diseased", "avg_conf"])
        w.writeheader()
        w.writerows(stats)
    log(f"统计 CSV: {csv_path}")

    # 3. 报告
    save_report(stats, per_class, n_test_images)

    # 4. 论文图
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_S19_class_dist(per_class)
    plot_S20_conf_hist(per_class)
    plot_S21_det_count(stats)
    plot_S22_health_ratio(stats)
    plot_fig14(stats, detect_dir)
    copy_val_curves()

    log("=" * 60)
    log("全部完成。请把输出发给我(或直接说一声),我继续更新论文:")
    log("  - S19 图注去掉 background 表述")
    log("  - 新增 Figure S23(混淆矩阵)图注")
    log("  - 按报告核对 5.1.2 测试集图片数(10,860)")
    log("  - 核对 Scheme B 示意置信度是否需更新")


if __name__ == "__main__":
    main()
