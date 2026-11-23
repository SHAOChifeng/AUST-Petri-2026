# OCPPP — Object-Centric Process Prediction Framework

Code accompanying the manuscript *"Integrating Coloured Petri Nets with Multi-Modal Deep Learning for Interpretable Prediction in Precision Crop Management"* (submitted to *Agronomy*, MDPI).

The framework combines (1) coloured Petri net (CPN) process modelling, (2) YOLOv7 leaf-disease detection and LSTM temperature forecasting, and (3) CPN-based interpretable scheduling/frost prediction. This repository contains the simulation, training-support, figure-generation, and manuscript-generation scripts used in the experiments.

## Repository layout

| Folder | Content |
|---|---|
| `01_cpn_simulation/` | CPN scheduling scenarios A–D (`run_all_real.py` runs all four with real data; `scheme_A.py`–`scheme_D.py` are standalone versions), the real-data loader, and the simulation outputs used in the paper (`scheme_data/`). |
| `02_lstm_forecasting/` | NASA POWER downloader, LSTM training script, and the visualisation script that produces Figures 12 and S1–S15. Trained weights (~100 MB) are not included; they are reproducible with `train.py`. |
| `03_yolov7_detection/` | Two-class figure regeneration script and the PlantVillage data configuration (`cfg/plantvillage.yaml`). Detection relies on the official [YOLOv7](https://github.com/WongKinYiu/yolov7) codebase. |
| `04_manuscript_generation/` | The `python-docx` manuscript generator (`generate_final_manuscript_v24.py`), the reference renumbering post-processor, and the figure-drawing scripts (CPN diagrams, Gantt charts, frost figure). |

## Dependencies

Python 3.8+ with:

```
cpnpy>=0.5        # CPN simulation
numpy, pandas, matplotlib
torch>=1.13       # LSTM training
python-docx, Pillow, graphviz   # manuscript & figures
ultralytics>=8    # YOLOv8s baseline (optional)
```

## Reproducing the experiments

**1. CPN simulations (Schemes A–D).** Requires `data/raw/nasa_power_19810101_20241231.csv` (see step 2 to download it). From `01_cpn_simulation/`:

```bash
python download_data.py   # (in 02_lstm_forecasting; writes lstm_agri/data/raw/)
python run_all_real.py    # re-runs Schemes A–D and saves figures + JSON results
```

**2. LSTM temperature forecasting.** From `02_lstm_forecasting/`:

```bash
python download_data.py   # NASA POWER, Hefei 31.8°N 117.2°E, 1981–2024
python train.py           # 44-year model (config in config.json; ~20 min CPU/GPU)
python visualize.py       # Figure 12 + Supplementary Figures S1–S15
```

**3. YOLOv7 two-class detection.** Uses the official [YOLOv7](https://github.com/WongKinYiu/yolov7) repository and the PlantVillage dataset (public, [Hughes & Salathé 2015]). The PlantVillage images are converted to YOLO format with one full-image box per image, two classes (`healthy_leaf`, `diseased_leaf`), following `cfg/plantvillage.yaml`:

```bash
# in the official yolov7 repository:
python test.py --task test --weights <path>/best_0718_1508.pt --data cfg/plantvillage.yaml
python regenerate_2class_figures.py   # Figures 11, S19–S23 (requires the dataset paths in the script)
```

**4. Manuscript generation.** From `04_manuscript_generation/` (requires the figure PNGs from steps 1–3):

```bash
python generate_final_manuscript_v24.py   # builds the .docx manuscript + supplementary file
python renumber_refs.py output/OCPPP_Agronomy_v24.docx   # renumbers citations by first appearance
```

## Data availability

- PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset (open access)
- NASA POWER: https://power.larc.nasa.gov/ (via `download_data.py`; requires a free API access)
- Trained model weights and the full detection statistics CSV are available from the corresponding author on reasonable request.

## License

MIT — see LICENSE.
