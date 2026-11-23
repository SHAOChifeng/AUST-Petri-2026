#!/usr/bin/env python3
"""
OCPPP Agronomy Manuscript — Complete MDPI-compliant Word document generation.
Generates a fully formatted .docx with all sections, figures, tables, and
experimental results including LSTM and YOLOv7 data.
"""

import os, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.resolve()
NEWCLAUDE_DIR = BASE.parent  # newclaude/
FIG_DIR = str(BASE / "figures")
LSTM_FIG_DIR = str(NEWCLAUDE_DIR / "lstm_agri" / "output" / "figures")
OUT_PATH = str(BASE / "output" / "OCPPP_Agronomy_v24.docx")
SUPP_PATH = str(BASE / "output" / "OCPPP_Agronomy_v24_Supplementary.docx")

# ── Helper functions ──────────────────────────────────────────────────────

def _set_cell_shading(cell, color):
    """Apply background shading to a table cell."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color}" w:val="clear"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                    first_indent=Cm(0.75), space_after=Pt(3),
                    space_before=Pt(0), line_spacing=1.15):
    """Create a paragraph with standard formatting."""
    p = doc.add_paragraph()
    p.alignment = alignment
    pf = p.paragraph_format
    pf.first_line_indent = first_indent
    pf.space_after = space_after
    pf.space_before = space_before
    pf.line_spacing = line_spacing
    return p

def _run(p, text, bold=False, italic=False, size=Pt(10),
         font_name="Times New Roman", color=None):
    """Add a run to a paragraph."""
    r = p.add_run(text)
    r.font.name = font_name
    r.font.size = size
    r.font.bold = bold
    r.font.italic = italic
    if color:
        r.font.color.rgb = color
    # CJK font fallback: only set eastAsia when the run actually contains
    # CJK characters, so that Latin runs (incl. math symbols) keep a single
    # clean font and Word/WPS do not fall back to full-width glyphs.
    if any('一' <= ch <= '鿿' for ch in text):
        r_elem = r._element
        rPr = r_elem.find(qn('w:rPr'))
        if rPr is None:
            rPr = parse_xml(f'<w:rPr {nsdecls("w")}/>')
            r_elem.insert(0, rPr)
        rFonts = rPr.find(qn('w:rFonts'))
        if rFonts is None:
            rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:eastAsia="SimSun"/>')
            rPr.insert(0, rFonts)
        else:
            rFonts.set(qn('w:eastAsia'), 'SimSun')
    return r

def heading(text, level=1, sb=Pt(14)):
    """Section heading."""
    sz_map = {0: Pt(14), 1: Pt(12), 2: Pt(10), 3: Pt(10)}
    sz = sz_map.get(level, Pt(10))
    sb_map = {1: Pt(14), 2: Pt(10), 3: Pt(8)}
    sb_val = sb_map.get(level, sb)
    p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.LEFT,
                        first_indent=Cm(0), space_before=sb_val, space_after=Pt(4))
    r = _run(p, text, bold=True, size=sz)
    return p

def body(text, indent=True, bold_prefix=None):
    """Body paragraph with optional bold prefix (for definitions)."""
    p = _make_paragraph(first_indent=Cm(0.75) if indent else Cm(0))
    if bold_prefix:
        _run(p, bold_prefix, bold=True)
        _run(p, text)
    else:
        _run(p, text)
    return p

def body_para():
    """Return an empty formatted body paragraph for building manually."""
    return _make_paragraph()

def caption(text):
    """Figure/table caption — bold, centered, 9pt."""
    p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        first_indent=Cm(0), space_before=Pt(4), space_after=Pt(6))
    _run(p, text, bold=True, size=Pt(9))
    return p

def code_block(text):
    """Monospace code block."""
    p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.LEFT,
                        first_indent=Cm(0.5), space_before=Pt(2))
    _run(p, text, font_name="Courier New", size=Pt(8))
    return p

def insert_image(filename, width=Inches(6.2), caption_text=None):
    """Insert image from either main figures/ or LSTM output/figures/.
    Width defaults to 6.2 in; tall images are capped at 7.0 in height."""
    fp = os.path.join(FIG_DIR, filename)
    if not os.path.exists(fp):
        fp = os.path.join(LSTM_FIG_DIR, filename)
    if os.path.exists(fp):
        try:
            from PIL import Image as PILImage
            with PILImage.open(fp) as im:
                w_px, h_px = im.size
            aspect = h_px / max(w_px, 1)
            if width.inches * aspect > 7.0:
                width = Inches(7.0 / aspect)
        except Exception:
            pass
        p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                            first_indent=Cm(0), space_before=Pt(2))
        run = p.add_run()
        run.add_picture(fp, width=width)
    if caption_text:
        caption(caption_text)

def make_table(headers, rows, col_widths=None, caption_text=None):
    """Create a formatted table with header shading.
    MDPI convention: table caption is placed ABOVE the table."""
    # Caption ABOVE the table (MDPI requirement)
    if caption_text:
        cp = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.LEFT,
                             first_indent=Cm(0), space_before=Pt(6), space_after=Pt(2))
        _run(cp, caption_text, bold=True, size=Pt(9))

    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _run(p, h, bold=True, size=Pt(9))
        _set_cell_shading(cell, "D9E2F3")

    # Data rows
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.rows[ri + 1].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _run(p, str(val), size=Pt(9))

    # Column widths if specified
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = w

    # Spacer paragraph after table
    _make_paragraph(first_indent=Cm(0), space_before=Pt(0), space_after=Pt(3))
    return t

def algorithm_box(title, lines):
    """Display algorithm with a styled box heading."""
    heading(title, level=3, sb=Pt(10))
    for line in lines:
        code_block(line)
    _make_paragraph(first_indent=Cm(0), space_after=Pt(4))

def set_doc(target):
    """Switch the global document object (main manuscript vs supplementary file)."""
    global doc
    doc = target

def setup_document(d):
    """Apply A4 page setup and default style to a fresh Document."""
    sec = d.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    for margin in ["top_margin", "bottom_margin", "left_margin", "right_margin"]:
        setattr(sec, margin, Cm(2.5))
    sty = d.styles["Normal"]
    sty.font.name = "Times New Roman"
    sty.font.size = Pt(10)
    sty.paragraph_format.space_after = Pt(3)
    sty.paragraph_format.line_spacing = 1.15


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT CREATION
# ═══════════════════════════════════════════════════════════════════════════
doc = Document()

# ── Page setup (A4, 2.5cm margins) and default style ──
setup_document(doc)

# ==========================================================================
# TITLE PAGE
# ==========================================================================
p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                    first_indent=Cm(0), space_before=Pt(0), space_after=Pt(6))
_run(p, "Integrating Coloured Petri Nets with Multi-Modal Deep Learning\nfor Interpretable Prediction in Precision Crop Management",
     bold=True, size=Pt(14))

# Authors line
p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                    first_indent=Cm(0), space_before=Pt(4), space_after=Pt(2))
_run(p, "Chifeng Shao", size=Pt(10))
_run(p, " 1,2,*, Qianqian Wang 2", size=Pt(10))

# Affiliations
affiliations = [
    "1 School of Mathematics and Big Data, Anhui University of Science and Technology, Huainan 232001, China",
    "2 College of Information and Network Engineering, Anhui Science and Technology University, Fengyang 233100, China",
]
for aff in affiliations:
    p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        first_indent=Cm(0), space_after=Pt(1))
    _run(p, aff, size=Pt(9))

# Correspondence
p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                    first_indent=Cm(0), space_before=Pt(2), space_after=Pt(16))
_run(p, "* Correspondence: fc_shao@126.com", size=Pt(9))

# ==========================================================================
# ABSTRACT
# ==========================================================================
heading("Abstract", level=1, sb=Pt(0))

body("Precision agriculture generates multi-modal data, yet existing predictive approaches rely either on black-box deep learning models lacking process-level interpretability or on mechanistic crop models that lack flexible data integration. We propose the Object-Centric Process Prediction Framework (OCPPP), combining coloured Petri nets (CPNs) with multi-modal deep learning. OCPPP comprises three modules: an object-centric CPN construction module assembling prediction-ready workflows from agronomic requirements; a multi-modal activity analysis module applying YOLOv7 to crop leaf images for disease severity estimation and LSTM networks to sensor time series for temperature forecasting; and an interpretable prediction module simulating the CPN to optimise operation sequences and predict low-frequency events. Experiments on PlantVillage and NASA POWER data show that YOLOv7 attains mAP@0.5 of 0.986 (mAP@0.5:0.95 = 0.965) on validation and 0.985 (0.964) on the held-out test set, on par with a YOLOv8s baseline; the LSTM improves day-1 temperature-forecast R² from 0.884 (5-year) to 0.949 (44-year). CPN simulations with real weather data yield 21.4% irrigation savings, 33.3% pest-control reduction, 24.8% frost-protection gain, and 17.3% full-season reduction; low-frequency frost prediction attains 41.7% hit rate. An end-to-end walk-through demonstrates the integrated pipeline, and every prediction carries an inspectable CPN execution trace, offering structural transparency beyond post-hoc explanations.")

# Keywords
p = _make_paragraph(first_indent=Cm(0), space_before=Pt(4))
_run(p, "Keywords: ", bold=True)
_run(p, "precision agriculture; predictive process monitoring; coloured Petri nets; multi-modal deep learning; interpretable AI; low-frequency event prediction")


# ==========================================================================
# 1. INTRODUCTION
# ==========================================================================
heading("1. Introduction")

body("Modern precision agriculture is fundamentally a multi-system interactive process in which crop production is managed through a sequence of interconnected agronomic operations—tillage, sowing, irrigation, fertilisation, pest control, and harvesting [1,17]. Unlike isolated prediction tasks, effective crop management requires understanding the temporal and causal dependencies among these operations: irrigation timing depends on prior rainfall and soil moisture, fertilisation strategies must account for crop growth stages detected from remote sensing, and harvesting schedules are constrained by preceding pest management activities. Each operation generates heterogeneous data—multispectral UAV imagery for crop health assessment, in-situ soil sensor time series for nutrient and moisture monitoring, and meteorological records for environmental context [17]. This multi-modal, flow-oriented nature of agricultural production calls for a predictive framework that can jointly reason across modalities while respecting the process-level dependencies inherent in crop management.")

body("Existing predictive approaches in precision agriculture fall into two broad categories. The first comprises single-task deep learning models that have achieved remarkable performance in specific subtasks: convolutional neural networks (CNNs) for disease detection from leaf images [9], long short-term memory (LSTM) networks for yield forecasting from time-series data [2,7], and object detectors such as YOLO for real-time pest and weed identification [8]. However, these models operate as black boxes—they produce predictions without revealing the decision path, making it difficult for agronomists and farmers to trust, verify, or refine the results [11]. The second category includes process-based crop models such as DSSAT and APSIM, which simulate crop growth through mechanistic equations and offer high interpretability [10]. These models, however, require extensive parameter calibration, struggle to integrate multi-modal sensor data, and lack the flexibility to adapt to dynamically changing management practices [23]. A critical gap remains: no existing framework jointly supports multi-modal data fusion, process-level prediction across interdependent agronomic operations, and transparent interpretability of the predictive reasoning chain.")

body("Agricultural production is particularly vulnerable to low-frequency, high-impact events—early or late frosts, extreme drought episodes, and sudden pest outbreaks—that can decimate yields despite their rarity [20]. These events are inherently difficult to predict with data-driven methods alone because historical occurrences are sparse, providing statistically insufficient training signals for deep learning models [18,21]. However, such events seldom arise in isolation; they are typically preceded by an observable chain of precursor conditions. For example, a frost event is preceded by sustained temperature decline, decreasing wind speed, and dropping humidity—each of which can be predicted from environmental time series with reasonable accuracy. A process-level framework that captures these causal chains can infer the likelihood of the rare event from the joint state of its precursors, thereby compensating for the scarcity of direct training examples.")

body("Coloured Petri nets (CPNs) [22] provide a mathematically rigorous formalism for modelling concurrent, resource-constrained processes, in which typed data tokens flow through a network of places and transitions. In the business process management (BPM) literature, CPNs have been widely used for control-flow modelling, simulation, and verification of complex workflows [14,15]. Their key advantage for predictive tasks is structural transparency: the model explicitly encodes the causal and temporal relationships between activities, the resource constraints that govern their execution, and the data dependencies that propagate through the process. In agricultural engineering, Petri nets have been applied to greenhouse climate control [13], agricultural product traceability [24], and irrigation system coordination. However, existing agricultural applications of Petri nets have focused on process control and optimisation rather than prediction, and they have not been integrated with deep learning models for multi-modal data analysis. This represents a missed opportunity, as the structural interpretability of CPNs naturally complements the predictive power of modern AI models.")

body("In this paper, we propose the Object-Centric Process Prediction Framework (OCPPP), which integrates CPN-based process modelling with multi-modal deep learning for interpretable prediction in precision crop management. The framework consists of three interconnected modules:")

# Module list as body paragraphs
body("Module 1—Process Modelling (Section 4.1): Given a set of required agronomic operations for a target field (e.g., irrigate, fertilise, spray, harvest), we construct an object-centric CPN model that captures control-flow dependencies, data-flow constraints, and resource-sharing relationships among operations. The model is dynamically assembled using Algorithm 1, which discovers precedence constraints from a library of known agronomic workflows.")

body("Module 2—Individual Activity Analysis (Section 4.2): For each operation in the process model, we apply task-specific AI models to extract state information from multi-modal activity logs. Object detection (YOLOv7 [6]) processes crop leaf images to estimate disease severity and health status, while LSTM networks [19] forecast environmental variables such as temperature from historical time series.")

body("Module 3—Interpretable Prediction (Section 4.3): The outputs of the individual AI models are integrated as tokenised state information into the CPN model. The CPN simulates the execution of the full workflow under different operation sequences, producing interpretable predictions of total completion time, resource bottlenecks, and system behaviour. Algorithm 2 performs combinatorial remaining-time prediction for unordered operations, and Algorithm 3 predicts low-frequency events by reasoning over the predicted states of their precursor conditions.")

body("The main contributions of this work are fourfold:")

body("1. Activity-Level Multi-Modal Insight. By applying dedicated AI models (YOLOv7, LSTM) to analyse image and time-series modalities within individual agronomic activity logs, the framework extracts fine-grained operational state information that would otherwise be lost in aggregated log data.")

body("2. Process-Level Interpretability. Unlike black-box deep learning predictors, OCPPP leverages the structural transparency of CPN models to make the predictive reasoning process visible and auditable. Every prediction is accompanied by an inspectable CPN execution trace that records which operations fired, in what order, at what times, and which resources they consumed. Agronomists can inspect the simulated token flow, identify resource bottlenecks, and understand why a particular prediction was reached.")

body("3. High-to-Low Frequency Prediction Transfer. For low-frequency agricultural events (e.g., frost, pest outbreaks) where direct data-driven prediction is infeasible, OCPPP predicts the precursor conditions through individual AI models and infers the rare event from the joint state of its prerequisites (Algorithm 3), transforming a data-scarce problem into a data-sufficient one.")

body("4. Low-to-High Dimensionality Composition. Instead of learning implicit high-dimensional representations of the entire workflow, OCPPP predicts individual activity states from low-dimensional inputs and composes them through the CPN structure. This modular design supports real-time model updates—individual AI models can be retrained or replaced with state-of-the-art alternatives without rebuilding the process-level predictor.")

# ==========================================================================
# 2. RELATED WORK
# ==========================================================================
heading("2. Related Work")

heading("2.1 Machine Learning in Precision Agriculture", level=2)
body("The application of machine learning (ML) and deep learning (DL) in precision agriculture has grown rapidly over the past decade, driven by the increasing availability of remote sensing imagery, in-situ sensor data, and high-performance computing [1,17]. A comprehensive survey by Kamilaris and Prenafeta-Boldú [1] catalogued early DL applications across crop classification, disease detection, weed identification, and yield prediction, concluding that convolutional neural networks (CNNs) and recurrent neural networks (RNNs) had become the de facto standard architectures for agricultural image and time-series analysis.")

body("In crop yield prediction, deep neural networks have demonstrated significant advantages over traditional statistical models. Khaki and Wang [2] developed a deep neural network integrating genotype and environmental variables to predict corn yield across the US Corn Belt. You et al. [3] proposed a deep Gaussian process framework for yield prediction from remote sensing data, capturing spatial and temporal dependencies simultaneously. More recently, Kumar et al. [7] combined UAV multispectral data with machine learning models for multi-stage corn yield prediction, demonstrating the value of temporal feature aggregation across growth stages.")

body("In the domain of crop disease and pest detection, object detection models—particularly the YOLO family—have become the predominant approach. Yang et al. [4] proposed a lightweight YOLOv8-based algorithm for tomato detection that incorporates feature enhancement and attention mechanisms. Xiao et al. [5] provided a comprehensive review of deep learning-based fruit detection and recognition for automatic harvesting. Zhang et al. [8] developed DCF-Yolov8 for aggregating low-level features to detect agricultural pests and diseases, while Bi et al. [9] proposed an improved MobileNetV3 model for corn leaf disease identification. A recent review by Attri et al. [38] surveyed deep learning techniques across agricultural applications, covering image-based disease diagnosis, yield prediction, and resource optimisation.")

body("Despite these advances, existing ML and DL approaches in agriculture share a common limitation: they are designed as single-task predictors that operate on isolated activities. A disease detection model, for instance, predicts disease presence from an image but does not reason about how this detection affects downstream operations such as pesticide scheduling, irrigation adjustment, or harvest timing. The predictive logic remains confined to individual tasks, and the interdependencies between agronomic operations are not captured.")

heading("2.2 Interpretable AI in Agriculture", level=2)
body("The black-box nature of deep learning models has raised increasing concerns about trustworthiness and reliability in high-stakes agricultural decisions [11]. Explainable AI (XAI) methods aim to address this by providing post-hoc explanations for model predictions. Established techniques such as SHAP, LIME, and Grad-CAM have been applied to agricultural problems including plant disease diagnosis [12] and crop yield factor interpretation.")

body("Mostafa et al. [12] provided a comprehensive review of explainable deep learning in plant phenotyping, demonstrating how Grad-CAM can highlight the regions of plant images most influential for disease classification. Their study showed that while XAI techniques can visualise model attention, they provide model-level explanation (why a specific input leads to a specific output) rather than process-level explanation (how this prediction affects the broader agronomic workflow). Furthermore, post-hoc explanations are approximations that may not faithfully represent the model's internal reasoning [11].")

body("This distinction is critical for precision agriculture, where a single prediction—such as a disease alert or a yield forecast—triggers a cascade of subsequent management decisions. A process-level understanding of the agronomic workflow is essential for farmers and agronomists to trust and act upon AI-generated predictions. The present work differs from the XAI literature by providing structural interpretability through an explicit process model (Petri net) rather than post-hoc explanation of a black-box model.")

heading("2.3 Petri Nets in Agricultural Systems", level=2)
body("Petri nets, originally developed for modelling concurrent and distributed systems, have found application in agricultural engineering for process control, resource allocation, and system verification. Tovany et al. [13] proposed a continuous timed Petri net model for greenhouse climate control, capturing the dynamic interactions between temperature, humidity, and ventilation actuators. The model enabled formal verification of control strategies before deployment, demonstrating the value of Petri nets for agricultural system design.")

body("In the domain of agricultural traceability and supply chain management, Wang and Sun [24] applied coloured Petri nets to model the tracing system for agricultural products, capturing the flow of information and materials across multiple stages from farm to consumer. These applications share a common orientation: they use Petri nets for system modelling, control, and optimisation rather than for prediction.")

body("To the best of our knowledge, no existing study has integrated Petri nets with deep learning models for multi-modal process-level prediction in agriculture. The present work addresses this gap by using coloured Petri nets as the structural backbone that organises the outputs of individual AI models into a coherent, interpretable prediction framework.")

heading("2.4 Predictive Business Process Monitoring: A Cross-Domain Reference", level=2)
body("Predictive business process monitoring (PBPM) is a well-established research area in the business process management community that addresses the problem of predicting future states of running process instances from historical event logs [14,15]. Tax et al. [14] introduced LSTM-based next-event prediction for business processes, achieving state-of-the-art accuracy on multiple benchmark logs. Mehdiyev et al. [15] further developed deep learning architectures for business process prediction, incorporating attention mechanisms to capture long-range dependencies in process traces. Outcome-oriented benchmarks [36] and methodological surveys [37] have since consolidated the evaluation practice and scope of the field.")

body("A key development in PBPM is the shift from traditional case-centric to object-centric process mining [16,35]. Traditional process mining assumes a single case identifier (e.g., an order ID) that links all events in a process. However, real-world processes—whether in healthcare, logistics, or agriculture—involve multiple interacting objects: patients and hospital resources in healthcare, fields and farm equipment in agriculture. Van der Aalst [16,35] formalised the problem of divergence and convergence in object-centric event data, where multiple objects participate in a single event or a single object spans multiple process instances.")

body("The present work draws inspiration from these PBPM developments but differs in three important respects. First, PBPM methods operate on structured event logs with predefined taxonomies, whereas agricultural process data is inherently multi-modal (imagery, time series, text logs), requiring modality-specific AI models for information extraction. Second, PBPM focuses on prediction accuracy as the primary objective, whereas our framework places equal emphasis on interpretability—the ability for domain experts to inspect and understand the predictive reasoning chain. Third, the agricultural domain introduces unique challenges—seasonal variability, weather-dependent process dynamics, and the critical importance of low-frequency extreme events [20,21]—that are not addressed in existing PBPM methods.")

# ==========================================================================
# 3. PRELIMINARIES
# ==========================================================================
heading("3. Preliminaries")
body("This section introduces the formal definitions underlying the proposed framework. All concepts are adapted from the standard Petri net literature [22] and extended with agricultural semantics to support the modelling of agronomic workflows.")

heading("3.1 Coloured Petri Nets", level=2)
body("Coloured Petri nets (CPNs) extend ordinary Petri nets by attaching typed data values—called colors—to tokens, enabling compact representation of systems with similar but distinguishable entities [22].")

body("A coloured Petri net is a tuple Σ = (P, T, A, Σ_C, V, C, G, E, I), where: P is a finite set of places, representing system states (e.g., \"Ready\", \"Irrigated\", \"Fertilized\"); T is a finite set of transitions, representing activities (e.g., \"Irrigate\", \"Fertilize\"), with P ∩ T = ∅; A ⊆ (P × T) ∪ (T × P) is a set of directed arcs connecting places to transitions and vice versa; Σ_C is a finite set of color sets (data types), such as FIELD = {Field_A, Field_B, Field_C}; V is a finite set of typed variables; C: P → Σ_C is a color function assigning a color set to each place; G: T → Expression is a guard function mapping each transition to a Boolean expression over V; E: A → Expression is an arc expression function defining the tokens consumed from or produced to each place; I: P → Expression is an initialisation function assigning a multi-set of typed tokens to each place.",
    bold_prefix="Definition 1 (Coloured Petri Net). ")

body("A transition t ∈ T is enabled at marking M if, for each input place p ∈ •t, the tokens required by the arc expression E(p,t) are present with the correct colors, and the guard G(t) evaluates to true. When t fires, it consumes the specified tokens from its input places and produces tokens on its output places according to E(t,p), yielding a new marking M′. This formal semantics provides the mathematical foundation for simulation and verification.")

body("Figure 1 illustrates a simple CPN modelling three sequential agronomic operations on a field. The color set FIELD = {Field_A, Field_B, Field_C} distinguishes tokens representing different fields, allowing the same net structure to model multiple field instances simultaneously. Place Ready initially holds three tokens, one for each field. Transition Irrigate is enabled when a field token is present at Ready. Upon firing, it consumes one token from Ready and deposits one token in Irrigated, representing that the field has received irrigation. The token then enables Fertilize, which in turn enables Spray. This explicit token flow encodes the causal dependency that a field must be irrigated before it can be fertilised, and fertilised before it can be sprayed.",
    bold_prefix="Agricultural example. ")

insert_image("fig1_cpn_sequence.png", caption_text="Figure 1. A simple CPN for three sequential agronomic operations. Places (circles) represent system states; transitions (rectangles) represent activities. The color set FIELD = {Field_A, Field_B, Field_C} distinguishes tokens belonging to different fields. The sequence Irrigate → Fertilize → Spray captures the causal dependencies inherent in crop management.")

heading("3.2 Shared Transitions and Resource Constraints", level=2)
body("In multi-field agricultural systems, multiple processes often share the same physical resource—an irrigation pump, a drone sprayer, or a farm worker—creating dependencies across ostensibly independent workflows. We formalise this through the concept of shared transitions.")

body("Let N_1 = (P_1, T_1, A_1) and N_2 = (P_2, T_2, A_2) be two Petri nets modelling independent field workflows. If T_1 ∩ T_2 ≠ ∅, the transitions in T_1 ∩ T_2 are called shared transitions. A shared transition represents an activity that belongs to multiple process models simultaneously, typically because it consumes a shared resource that can only be used by one process instance at a time.",
    bold_prefix="Definition 2 (Shared Transition). ")

body("Figure 2 illustrates two nets: N₁ models the workflow for Field A, and N₂ models the workflow for Field B. Both nets include the same transition Irrigate, representing a single irrigation pump physically connected to both fields. When Irrigate fires for Field A, it locks the pump for the duration of the irrigation operation (modelled by the timed token), preventing Field B's irrigation until the pump becomes available.",
    bold_prefix="Agricultural example. ")

insert_image("fig2_shared_transition.png", caption_text="Figure 2. Shared transition between two field workflows. N₁ (Field A, left dashed box) and N₂ (Field B, right dashed box) both include transition Irrigate (center, highlighted), representing a shared irrigation pump.")

body("For a net N = (P, T, A) and a transition t ∈ T, let •t = {p ∈ P | (p,t) ∈ A} denote the preset of t, and let t• = {p ∈ P | (t,p) ∈ A} denote the postset of t. A transition t_j is a precursor of transition t_i, denoted t_j ≺ t_i, if there exists a place p ∈ P such that (t_j, p) ∈ A and (p, t_i) ∈ A. The set Pre(t_i) = {t_j ∈ T | t_j ≺ t_i} is called the precursor transition set of t_i.",
    bold_prefix="Definition 3 (Precursor Transition Set). ")

body("In a corn production workflow, the transition Harvest has Pre(Harvest) = {Irrigate, Fertilize, Spray} because irrigation, fertilisation, and pest control must all precede harvesting. Figure 3 shows this relationship: three parallel branches each produce a token on their respective intermediate places. These tokens flow into a join place (Field Ready), which accumulates all three. Harvest can fire only when all three tokens are present simultaneously—a synchronisation mechanism known as AND-join semantics that is native to Petri nets.",
    bold_prefix="Agricultural example. ")

insert_image("fig3_precursor_transitions.png", caption_text="Figure 3. Precursor transitions of Harvest. Pre(Harvest) = {Irrigate, Fertilize, Spray}. All three precursor transitions must complete and deposit tokens at the join place Field Ready before Harvest can be enabled.")

heading("3.3 Activity Logs in Precision Agriculture", level=2)
body("Process mining and prediction rely on event logs that record the execution of activities. We define three types of logs relevant to agricultural workflows, extending the standard definitions from the process mining literature [14,16] with modalities specific to precision agriculture.")

body("For a transition t_i ∈ T in a Petri net, each firing of t_i generates an activity record of the form ⟨ID, Field_ID, Operator_ID, Timestamp, Other_Inf⟩. The collection of all such records for t_i is called the activity log of t_i.",
    bold_prefix="Definition 4 (Activity Log). ")

body("For a net N = (P, T, A), the system log is the union of all activity logs across all transitions, recorded as ⟨CASE_ID, Event, Timestamp, Other_Inf⟩, where CASE_ID links events belonging to the same process instance.",
    bold_prefix="Definition 5 (System Log). ")

body("For a net N and a process instance, the ordered sequence of transition firings is called a trace. The collection of all traces across instances is the trace log of N.",
    bold_prefix="Definition 6 (Trace Log). ")

body("A transition t_i is called a multi-modal transition if its activity log includes data in two or more modalities beyond structured text—such as images (crop leaf photographs or UAV multispectral imagery), time series (sensor readings), or audio recordings—stored in the Other_Inf field of Definitions 4 and 5.",
    bold_prefix="Definition 7 (Multi-Modal Transition). ")

body("The Scout transition exemplifies a multi-modal activity. It consumes three types of input tokens: a field token, a UAV image data source, and a soil sensor data source. Its firing produces a scouted field token and a scout report containing NDVI, soil moisture, and pest alert information. Figure 4 illustrates this multi-modal transition.",
    bold_prefix="Agricultural example. ")

insert_image("fig4_multimodal_transition.png", caption_text="Figure 4. Multi-modal transition Scout. The transition consumes three input tokens: a field token, a UAV image data source, and a soil sensor time series. It produces a scouted field state and a multi-modal scout report.")

body("Given a trace log L with total transition occurrences |L|, and a threshold θ ∈ (0,1), a transition t_i with occurrence count f(t_i) is called a low-frequency transition if f(t_i) / |L| < θ.",
    bold_prefix="Definition 8 (Low-Frequency Transition). ")

body("An unexpected frost event or a rare pest outbreak that occurs in fewer than 1% of field seasons would be classified as a low-frequency transition (θ = 0.01). Because direct historical data for such events is sparse, prediction must rely on reasoning over precursor conditions rather than purely data-driven methods.",
    bold_prefix="Agricultural example. ")

# ==========================================================================
# 4. METHOD
# ==========================================================================
heading("4. Multi-Modal Interpretable Prediction Method")
body("This section presents the Object-Centric Process Prediction Framework (OCPPP), which integrates CPN-based process modelling with multi-modal deep learning for interpretable prediction in precision crop management. The framework comprises three interconnected modules, each addressing a distinct aspect of the prediction problem.")

heading("4.1 Process-Level Modelling (Module 1)", level=2)
body("Existing predictive approaches in precision agriculture typically assume a fixed, predefined workflow. In practice, however, the set of agronomic operations required for a given field varies with crop type, growth stage, soil conditions, and seasonal factors. A field planted with drought-tolerant maize, for instance, may require fewer irrigation events than one planted with a conventional variety. Similarly, a field with a known pest history may require additional scouting and spraying operations.")

body("To address this variability, we propose an object-centric process modelling approach. The \"object\" in our framework is a crop field (or a management unit within a field). Given its specific agronomic requirements, the framework dynamically constructs a CPN model that captures the relevant operations, their causal dependencies, the data flows between them, and the resource constraints under which they operate.")

heading("4.1.1 CPN Declaration and Resource Modelling", level=3)
body("The CPN model for a crop field is built from the following primitives, declared in CPN Tools syntax:")

code_block("colset FIELD  = with Field_A | Field_B | Field_C | Others timed;")
code_block("colset FIELDList = list FIELD;")
code_block("colset E = unit with e timed;")
code_block("var field : FIELD;")
code_block("var fieldList : FIELDList;")

body("The color set FIELD distinguishes tokens representing different fields. The timed keyword attaches a time stamp to each token, enabling temporal simulation of operation durations and queue wait times. The list type FIELDList models a queue of fields waiting for a shared resource.")

body("Resource constraints are modelled through a timer place that limits concurrent access. Figure 5 illustrates a CPN model with a queue and a resource limit. The place IncomingFields holds tokens for fields awaiting an operation. Transition FormQ moves tokens into the queue (Term), while transition ExecuteOp consumes one token from the queue and one token from Timer. The Timer place enforces a limit on parallel executions: it initially contains a fixed number of tokens (e.g., one token for a single irrigation pump), and each execution consumes one token, which is returned after the operation duration elapses (modelled by the timed return arc labeled @+t). The declarations above follow the CPN Tools notation for readability; the executable simulations in Section 5.4 are implemented with the cpnpy Python library [27], which follows the same semantics as Definition 1.")

insert_image("fig5_queue_model.png", caption_text="Figure 5. CPN model with queue and resource constraint. Place Timer limits parallel execution count (initially one token, representing a single irrigation pump). The queue (Term) buffers waiting fields.")

heading("4.1.2 Automated Process Model Construction (Algorithm 1)", level=3)
body("Given a set of required agronomic operations for a target field, the framework must assemble them into a connected CPN. The key challenge is discovering the precedence constraints between operations: some activities must occur in sequence (e.g., irrigation must precede fertilisation if the fertiliser requires moist soil), others may be concurrent or unordered (e.g., two independent scouting flights over different field sections).")

body("We formalise this through the concept of data-flow constraints between transitions. Two operations are data-dependent if the output data of one serves as the input data of the other. For example, the output of a soil moisture sensor (threshold: dry) is a precondition for triggering an irrigation event. Algorithm 1 discovers these constraints from a library of known agronomic workflow patterns.")

algorithm_box("Algorithm 1. Object-Centric Process Model Construction", [
    "Input: Required operation set T = {t₁, t₂, ..., tₙ},",
    "        existing net library N = {N₁, N₂, ..., Nₘ}",
    "Output: Connected CPN model N_pred = (P_pred, T_pred, A_pred)",
    "",
    " 1:  G ← empty directed graph with vertices T",
    " 2:  // Phase 1: Cross-net dependency discovery",
    " 3:  for each pair (tᵢ, tⱼ) ∈ T × T, tᵢ ≠ tⱼ do",
    " 4:    for each net Nₖ ∈ N do",
    " 5:      if tᵢ has input data d_in ∧ tⱼ has output data d_out",
    " 6:         ∧ d_in ∩ d_out ≠ ∅ within Nₖ then",
    " 7:        add edge tⱼ → tᵢ to G  // tⱼ is data-precursor of tᵢ",
    " 8:  // Phase 2: Intra-net dependency discovery",
    " 9:  for each tᵢ ∈ T do",
    "10:    for each Nₖ ∈ N where tᵢ ∈ Tₖ do",
    "11:      for each tⱼ ∈ Tₖ, tⱼ ≠ tᵢ do",
    "12:        if tᵢ has input data d_in ∧ tⱼ has output data d_out",
    "13:           ∧ d_in ∩ d_out ≠ ∅ then",
    "14:          add edge tⱼ → tᵢ to G",
    "15:  // Phase 3: Layered CPN construction",
    "16:  L₀ ← {t ∈ T | in-degree(t) = 0 in G}  // initial layer",
    "17:  for each remaining t ∈ T \ L₀, sorted by topological order do",
    "18:    Lᵢ ← Lᵢ ∪ {t}  // assign to layer by longest-path depth",
    "19:    for each precursor p ∈ Pre(t) do",
    "20:      add place p_mid with AND-join arc semantics",
    "21:  return N_pred = (P_pred, T_pred, A_pred) built from G and layers",
])

body("How Algorithm 1 works, illustrated with a concrete example. Consider a field requiring four operations: T = {Irrigate, Scout_Drone, Fertilize, Spray}. The existing net library contains three relevant nets: N₁ (a standard irrigation workflow), N₂ (a fertilisation workflow requiring soil moisture data from irrigation), and N₃ (a pest management workflow where Scout_Drone produces a pest density map consumed by Spray). Algorithm 1 discovers Irrigate ≺ Fertilize and Scout_Drone ≺ Spray. Irrigate and Scout_Drone have no precursors and are placed in Layer 0 as concurrent branches. The resulting CPN, shown in Figure 6, has two independent starting branches that converge independently to their dependent operations.",
    bold_prefix="How Algorithm 1 works. ")

insert_image("fig6_workflow_construction.png", caption_text="Figure 6. Workflow construction by Algorithm 1. (a) Input operation set. (b) Data-precursor dependencies discovered from the library nets. (c) Output CPN with two concurrent branches (Irrigate and Scout_Drone in Layer 0) and their dependent operations (Fertilize and Spray in Layer 1).")

body("Pattern library and complexity. The pattern library used in our implementation contains m = 9 operation patterns—irrigate, fertilise, scout, spray, harvest, cover, anti-frost application, soil sampling, and pest trapping—each annotated with typical precedence templates; additional patterns extend the library without changing Algorithm 1. For a request with k operations and a library of m patterns, Algorithm 1 performs at most O(k · m) precedence checks, so the construction cost is negligible relative to the subsequent simulation; when no library pattern matches an operation pair, the framework falls back to a manual precedence declaration, a limitation discussed in Section 6. Because every library pattern is a workflow-net fragment, the nets assembled by Algorithm 1 remain bounded and deadlock-free by construction; formal verification of these properties is left to future work.",
    bold_prefix="Pattern library and complexity. ")

heading("4.2 Multi-Modal Activity Analysis (Module 2)", level=2)
body("Once the process model is constructed, individual operations must be analysed to extract their current and predicted states. Agricultural activity logs are inherently multi-modal: a single operation may generate UAV imagery (requiring computer vision), soil sensor time series (requiring sequence modelling), and structured metadata. Module 2 applies task-specific AI models to each modality.")

heading("4.2.1 Disease Severity Estimation Using YOLOv7", level=3)
body("YOLOv7 [6] is a state-of-the-art real-time object detector that has been widely adopted in agricultural applications including fruit detection [5], pest and disease identification [8], and crop growth stage recognition [4].")

body("In our framework, YOLOv7 processes crop leaf images—such as those from the publicly available PlantVillage dataset [25]—to estimate the health status of plants before and after an operation. The specific detection tasks include disease and pest symptom detection (identifying leaf lesions, discoloration, or pest damage from crop photographs) and growth stage classification (detecting phenological markers such as tasseling in maize or flowering in soybean, based on visual features). Although each PlantVillage image contains a single leaf, so that localisation is trivially solved by the full-image boxes, we retain the detection formulation for three reasons. First, the framework targets UAV field imagery in which multiple leaves, background objects, and partial occlusions co-occur, and a detector extends to this regime without architectural change. Second, object-level outputs (bounding boxes with class probabilities) match the activity-log records of Definition 7, preserving the multi-modal transition semantics of the CPN. Third, detection yields a direct disease-severity estimate through the ratio of diseased to healthy detections (d_disease = n_diseased / n_leaves), which a single image-level classification score cannot express when multiple objects are present.")

body("The YOLOv7 architecture consists of a feature extraction backbone (Extended Efficient Layer Aggregation Network, E-ELAN), a neck for multi-scale feature fusion (SPPCSPC and PANet), and a detection head that outputs bounding boxes and class probabilities. The output of this module is a structured state vector:")

p = _make_paragraph(first_indent=Cm(1.5))
_run(p, "s_vision = ⟨n_leaves, d_disease⟩", italic=True)

body("where n_leaves is the number of detected leaf instances and d_disease = n_diseased / n_leaves is the disease severity ratio. The vision module is designed to be task-agnostic: the same interface supports leaf-level analysis, UAV-based field surveys, or greenhouse monitoring, depending on the available data source.")

heading("4.2.2 Temperature Forecasting Using LSTM", level=3)
body("Time-series data—temperature, precipitation, humidity, and solar radiation—are central to agricultural decision-making. LSTM networks [19] are well-suited for forecasting these variables because they capture long-range temporal dependencies through gated memory cells.")

body("In our framework, LSTM models are trained on historical meteorological time series from the NASA POWER dataset [26]. The LSTM architecture consists of an input layer, two LSTM hidden layers (128 units each) with dropout for regularisation, and a dense output layer producing the forecast. Each LSTM cell maintains a cell state and a hidden state, controlled by input, forget, and output gates.")

body("The output of this module is a predicted state vector:")

p = _make_paragraph(first_indent=Cm(1.5))
_run(p, "s_ts = ⟨T_temp, R_rainfall, H_humidity, W_wind⟩", italic=True)

body("where each component denotes a predicted value over the 7-day forecast horizon, with associated prediction intervals.")

heading("4.3 Interpretable Prediction via CPN Simulation (Module 3)", level=2)
body("The outputs of Modules 1 and 2 are combined in Module 3: the CPN process model from Module 1 provides the structural backbone, and the activity state predictions from Module 2 serve as token values that parameterise the simulation. The CPN is then executed as a discrete-event simulation, producing predictions of workflow-level outcomes.")

heading("4.3.1 Process Model Execution Semantics", level=3)
body("The CPN model is executed according to the standard CPN semantics formalised by Jensen and Kristensen [22]. We restate the execution semantics here in the context of agricultural process prediction.")

body("Let Σ = (P, T, A, Σ_C, V, C, G, E, I) be a coloured Petri net with global clock τ ∈ ℝ_≥0 and marking M : P → Bag(Σ_C × ℝ_≥0) assigning each place a multi-set of timed tokens. The execution proceeds as follows:",
    bold_prefix="Definition 9 (CPN Execution Semantics). ")

body("Marking initialisation: M₀(p) = I(p) for all p ∈ P, where I(p) is the initial token assignment. Resource places (e.g., Pump) are initialised with available resource tokens; field places (e.g., Ready) are initialised with field tokens to be processed.")

body("Enabling condition: A transition t ∈ T is enabled at marking M with binding b : V → Σ_C, denoted Enabled(t, M, b), if and only if: (a) for each input place p ∈ •t, the arc expression E(p, t)(b) evaluates to a multi-set of tokens present in M(p) with timestamps ≤ τ; and (b) the guard expression G(t)(b) evaluates to true. Formally:")

p = _make_paragraph(first_indent=Cm(1.5))
_run(p, "Enabled(t, M, b) ⟺ (∀p ∈ •t : E(p, t)(b) ≤ M(p)) ∧ G(t)(b) = true", italic=True)

body("Firing rule: When t fires with binding b at time τ, it produces a new marking M' defined as:")

p = _make_paragraph(first_indent=Cm(1.5))
_run(p, "M'(p) = M(p) − E(p, t)(b) + E(t, p)(b)   for all p ∈ P", italic=True)

body("where consumed tokens are removed from input places and produced tokens are placed on output places. For timed transitions, the produced tokens receive timestamp τ + delay(t), modelling operation duration.")

body("Time advancement: Between firings, the global clock advances to the earliest timestamp among all pending tokens:")

p = _make_paragraph(first_indent=Cm(1.5))
_run(p, "τ' = min{timestamp(tok) | tok ∈ M(p), p ∈ P, timestamp(tok) > τ}", italic=True)
body("If no tokens are pending, the simulation terminates. This discrete-event semantics ensures that the CPN execution trace records every operation's start time, duration, end time, and resource usage—providing the structural transparency that distinguishes OCPPP from black-box predictors.")

heading("4.3.2 Prediction for Unordered Operations (Algorithm 2)", level=3)
body("In many agricultural scenarios, a set of operations at the same growth stage can be performed in any order—for instance, scouting, soil sampling, and pest trapping are logically independent. However, their execution order affects total completion time due to resource contention and time-dependent conditions. Algorithm 2 addresses this by enumerating all feasible execution sequences and predicting the total duration for each.")

algorithm_box("Algorithm 2. Combined Process Remaining Time Prediction", [
    "Input:  Unordered transition set U = {t₁, t₂, ..., tₖ};",
    "        current time τ₀; duration predictor PredictDuration;",
    "        resource constraints R; concurrency model C",
    "Output: List L = {(seq, total_time) | seq ∈ Perm(U)} sorted by total_time",
    "",
    " 1:  Perm ← all permutations of U           // O(k!) sequences",
    " 2:  L ← ∅",
    " 3:  for each sequence seq ∈ Perm do",
    " 4:    τ ← τ₀",
    " 5:    for each transition t ∈ seq do",
    " 6:      d ← PredictDuration(t, τ)          // task duration",
    " 7:      τ_wait ← ResourceWaitTime(t, τ, R) // resource contention",
    " 8:      τ ← τ + τ_wait + d",
    " 9:      UpdateResourceState(t, R)           // lock/release",
    "10:    append (seq, τ − τ₀) to L",
    "11:  sort L by total_time ascending",
    "12:  return L",
])

body("Suppose a field requires three unordered operations: U = {Scout_Drone, Soil_Sample, Trap_Pests}. The farm has one drone and one field technician. Algorithm 2 generates 6 sequences and evaluates each. The optimal sequence (Soil_Sample → Scout_Drone → Trap_Pests) completes in 2.8 hours by parallelising the technician's soil sampling work with drone operations. Figure 7 compares the completion times of all six candidate sequences. At the field-operation granularity considered here, unordered sets are small (k ≤ 6–8), for which full enumeration is trivial; for larger k, branch-and-bound pruning by completion-time lower bounds or resource-release ordering could reduce the search, which we leave to future work. This worked example is illustrative; the quantitative evaluation of Algorithm 2 over realistic agronomic scenarios is reported in Section 5.4, and a proof-of-concept execution trace is provided in Appendix A.",
    bold_prefix="How Algorithm 2 works. ")

insert_image("fig10_gantt_comparison.png", caption_text="Figure 7. Comparison of total completion time for six execution sequences of unordered agronomic operations (Algorithm 2). Each row shows one sequence with operations coloured by type. The optimal sequence completes in 2.8 hours, a 22% improvement over the worst sequence (3.6 hours).")

heading("4.3.3 Prediction for Low-Frequency Events (Algorithm 3)", level=3)
body("Low-frequency events—such as an early frost or a sudden pest outbreak—are characterised by sparse historical data that makes direct deep learning prediction infeasible. However, these events are typically preceded by a chain of observable precursor conditions. The key insight of Algorithm 3 is that the low-frequency event can be inferred from the joint state of its precursors, transforming a data-scarce problem into a data-sufficient one.")

algorithm_box("Algorithm 3. Low-Frequency Event Prediction via Precursor Reasoning", [
    "Input:  Low-frequency transition t_low;",
    "        precursor set Pre(t_low) = {p₁, p₂, ..., pₘ};",
    "        threshold functions θₚ for each precursor;",
    "        prediction horizon H = [τ₀, τ₀ + Δ].",
    "Output: Predicted occurrence time τ_low, or ∅ if not predicted.",
    "",
    " 1:  for each precursor p ∈ Pre(t_low) do",
    " 2:    PredictState(p, H)            // using Module 2 AI models",
    " 3:    Θ'_p ← {τ ∈ H | predicted_state(p, τ) > θₚ}",
    " 4:  // Intersection: AND-join semantics",
    " 5:  τ_candidates ← Θ'_p₁",
    " 6:  for each remaining Θ'_pᵢ, i ≥ 2 do",
    " 7:    τ_candidates ← τ_candidates ∩ Θ'_pᵢ",
    " 8:    if τ_candidates = ∅ then return ∅",
    " 9:  τ_low ← min(τ_candidates)       // earliest joint occurrence",
    "10:  return τ_low",
])

body("The algorithm exploits the AND-join semantics of Petri nets: a transition with multiple input places is enabled only when all input tokens are simultaneously present. The intersection of precursor time windows corresponds to this condition. Figure 8 illustrates the intersection procedure on a real worked example (forecast origin 2023-12-22).")

insert_image("fig11_frost_intersection.png", caption_text="Figure 8. Low-frequency frost prediction via precursor condition intersection (Algorithm 3), shown on a real worked example (forecast origin 2023-12-22). (a) The LSTM 7-day temperature forecast (all days below 0 °C); (b, c) climatological probabilities P(WS < 3 m/s) and P(RH > 60%) for the corresponding calendar days, computed from the 44-year training record; (d) the Boolean intersection, whose earliest satisfied day (Day 2, 2023-12-23) is the frost time returned by Algorithm 3.")

heading("4.3.4 Interpretability Through Simulation", level=3)
body("A distinguishing feature of OCPPP is that its predictions are accompanied by an execution trace generated by the CPN simulation. This trace records: (1) which operations fired, in what order, and at what times; (2) which resource was used for each operation and for how long—making resource bottlenecks visible; (3) which precursor conditions triggered the prediction of a low-frequency event; and (4) how changes in resource availability or operation timing affect the predicted outcome—supporting \"what-if\" analysis. Figure 9 summarises the overall framework architecture. To make the interpretive value of the trace concrete, consider Scheme B (Section 5.4.2): the simulation shows that all four single-drone orderings complete in 90 minutes, and the trace attributes this invariance to the drone being continuously occupied—the resource bottleneck—rather than to any ordering choice, an insight that a black-box completion-time predictor cannot provide.")

insert_image("fig12_framework.png", caption_text="Figure 9. The OCPPP framework architecture. The three modules correspond to Sections 4.1 (process modelling, left), 4.2 (multi-modal AI analysis, center), and 4.3 (CPN-based interpretable prediction, right).")

# ==========================================================================
# 5. EXPERIMENTS AND RESULTS
# ==========================================================================
heading("5. Experiments and Results")
body("This section presents the experimental evaluation of the OCPPP framework across four dimensions: object detection on the PlantVillage dataset, temperature forecasting on NASA POWER data, interpretable CPN simulation demonstrating resource contention tracing, and low-frequency frost prediction using a hybrid LSTM-climatology approach. Section 5.6 then chains the individually validated modules into a single end-to-end prediction pipeline on a walk-through example.")

heading("5.1 Experimental Setup", level=2)

heading("5.1.1 Platform Configuration", level=3)
body("All experiments were conducted on a workstation running Windows 11 with an Intel Core i7-12700 CPU, an NVIDIA GeForce RTX 3090 GPU (24 GB VRAM), and 64 GB of RAM. The deep learning models were implemented in Python 3.8 with PyTorch 1.13 and CUDA 11.7. The CPN models were implemented and simulated using the cpnpy library [27], an open-source Python framework for constructing and executing coloured Petri net models.")

heading("5.1.2 Datasets", level=3)
body("Two datasets and one scenario configuration were used in the experiments.")

body("The PlantVillage dataset [25] consists of 54,306 images of healthy and diseased crop leaves across 14 crop species and 26 disease categories. We selected this dataset because it is one of the largest publicly available agricultural image datasets, providing sufficient training diversity for a deep detection model, and its controlled laboratory imaging conditions allow us to isolate disease recognition capability without confounding factors such as variable field illumination. The original PlantVillage images were converted to YOLO format with automated bounding box annotation, one full-image box per image, for two classes: healthy_leaf and diseased_leaf. An earlier conversion declared a third background class; inspection showed that no annotation was ever assigned to it, so the scheme was consolidated to the two leaf classes. The YOLO-converted dataset contains 86,876 training images, 10,849 validation images, and 10,885 test images. For the training experiment, a 50% stratified subset (~43,438 training images) was used to balance model accuracy against computational cost, with the full 86,876-image set reserved for future large-scale deployment. The 10% subset (~8,687 images) used in the training-data scale analysis of Section 5.2 was drawn from the full training set with the same class-stratified procedure.",
    bold_prefix="Dataset 1—PlantVillage (for YOLOv7). ")

body("The NASA Prediction of Worldwide Energy Resources (POWER) project [26] provides global daily meteorological reanalysis data at a 0.5 degree grid resolution. We downloaded daily time series for the Hefei/Anhui region (31.8°N, 117.2°E) from January 1981 to December 2024—a 44-year record covering approximately 16,000 days—including six parameters: temperature at 2 meters (T2M, °C), precipitation (PRECTOTCORR, mm/day), relative humidity (RH2M, %), wind speed (WS2M, m/s), surface pressure (PS, kPa), and solar radiation (ALLSKY_SFC_SW_DWN, kW·h/m²/day). The 44-year span captures multi-decadal climate variability including multiple El Niño-Southern Oscillation cycles, which a shorter record cannot provide. The data was split chronologically: 80% training (January 1981–March 2016), 10% validation (March 2016–August 2020), and 10% test (August 2020–December 2024). A chronological split was chosen over a random split to simulate the realistic deployment scenario where a model trained on past data predicts future conditions. For the ablation analysis, a 5-year subset (January 2020–December 2024) was also prepared to quantify the benefit of longer training records.",
    bold_prefix="Dataset 2—NASA POWER (for LSTM). ")

body("The CPN scheduling scenarios are parameterised using real meteorological data from NASA POWER and the growing degree day (GDD) model [29]. Four agricultural scheduling scenarios are implemented: irrigation scheduling using the Hargreaves ET₀ equation [28] with 2024 temperature data, pest control scheduling using YOLOv7 PlantVillage detection statistics, frost emergency response using 44-year frost climatology, and full-season resource configuration for winter wheat using GDD-driven crop growth stage detection. All scenarios are executed as discrete-event simulations within the cpnpy framework [27], with operation durations and resource constraints grounded in the real data.",
    bold_prefix="Scenario Configuration 3—NASA POWER + GDD (for CPN simulation). ")

heading("5.1.3 Evaluation Metrics", level=3)
body("For object detection (YOLOv7), we used mean average precision at IoU threshold 0.5 (mAP@0.5), precision, recall, and F1-score, which are standard metrics allowing direct comparison with published YOLOv7 results on agricultural datasets [4,6,8]. Detection metrics were computed with the YOLOv7 evaluation defaults—confidence threshold 0.001 and IoU threshold 0.65 for non-maximum suppression [6]. For time-series forecasting (LSTM), we used mean absolute error (MAE), root mean squared error (RMSE), and the coefficient of determination (R²). For CPN scheduling simulation, we measured total completion time (minutes) and relative time savings across scheduling strategies and resource configurations.")

heading("5.1.4 Model Configuration and Hyperparameter Justification", level=3)

body("This subsection provides a detailed justification of the hyperparameter choices for both deep learning models, explaining the rationale behind each design decision.")

# YOLOv7 Hyperparameters
p = _make_paragraph(first_indent=Cm(0))
_run(p, "YOLOv7 Configuration and Rationale.", bold=True)

body("The YOLOv7 model was initialised with COCO pre-trained weights (yolov7.pt) and fine-tuned on the PlantVillage YOLO-converted dataset at 256 × 256 pixel resolution with a batch size of 128. The reduced resolution (84% fewer pixels than the 640 × 640 default) is justified by the dataset characteristics: PlantVillage images are close-up leaf photographs where key disease features remain clearly distinguishable at 256 × 256, and the GPU memory savings enable a batch size of 128 that provides more stable gradient estimates. Training was conducted on a 50% stratified subset (~43,438 images) for 150 epochs using the hyp.fast.yaml augmentation schedule (mosaic p = 0.3, mixup p = 0.05). The SGD optimiser with momentum 0.937 and initial learning rate 0.01 with cosine annealing follows the standard YOLOv7 recipe [6]; detailed hyperparameter comparisons between the 10% and 50% configurations are provided in Appendix B (Table A2).")

body("The LSTM model uses 2 hidden layers of 128 units each (dropout 0.3), a 14-day input window, and a 7-day forecast horizon. The 128-unit hidden size was selected after preliminary experiments showed that 64 units underfit seasonal patterns (validation R² < 0.85), while 256 units provided marginal improvement at double the cost. The 14-day input captures synoptic-scale weather patterns, and the 7-day output matches standard medium-range forecasting horizons. The 44-year training set (~12,500 sequences) enables robust learning across multiple climate regimes. The Adam optimiser (lr = 0.001, weight decay = 1 × 10⁻⁵) follows standard LSTM practice [19]; Adam's per-parameter adaptive rates are effective for weather series where variables have vastly different scales. Training runs for a maximum of 200 epochs with early stopping patience of 30. Table 1 summarises the hyperparameter configurations of both models.")

# Table 1: Hyperparameters
make_table(
    ["Parameter", "YOLOv7", "LSTM"],
    [
        ["Input size", "256 × 256 px", "14-day × 6-variable sequence"],
        ["Batch size", "128", "64"],
        ["Optimiser", "SGD (momentum = 0.937)", "Adam (β₁ = 0.9, β₂ = 0.999)"],
        ["Learning rate", "0.01 (cosine annealing)", "0.001"],
        ["Weight decay", "5 × 10⁻⁴", "1 × 10⁻⁵"],
        ["Epochs", "150 (warmup = 3)", "200 (early stop patience = 30)"],
        ["Workers", "4", "0"],
        ["Loss function", "CIoU + BCE", "MSE"],
        ["Hidden units", "—", "2 layers × 128 units"],
        ["Dropout", "—", "0.3"],
        ["Pre-trained weights", "COCO (yolov7.pt)", "None"],
        ["Augmentation", "mosaic (0.3), mixup (0.05), HSV, flip", "—"],
        ["Training samples", "~43,438 (50% subset)", "~12,500 sequences"],
        ["GPU memory usage", "~20.4 GB", "~2.3 GB"],
        ["Per-epoch time", "~2.6 min (50% subset)", "~1 sec"],
    ],
    caption_text="Table 1. Training hyperparameters and hardware configuration for YOLOv7 and LSTM models."
)

heading("5.2 Experiment 1: YOLOv7 Object Detection on PlantVillage", level=2)
body("The YOLOv7 model was fine-tuned on the PlantVillage YOLO-converted 50% stratified subset (~43,438 training images) for 150 epochs. This configuration balances model accuracy against training cost while maintaining sufficient class diversity.")

body("Figure 10 presents the training curves over 150 epochs. All losses (box, objectness, classification) decrease rapidly within the first 25 epochs and converge gradually thereafter. Detection metrics (precision, recall, mAP@0.5) exhibit oscillatory behaviour during later epochs, which is characteristic of YOLOv7's bag-of-freebies training dynamics. The model peaks at mAP@0.5 of 0.992 around epoch 25, while mAP@0.5:0.95 continues improving to 0.966 by epoch 148, indicating that stricter IoU thresholds benefit from extended training. The final model (epoch 149), re-validated on the re-annotated two-class validation set, achieves precision = 0.91, recall = 0.95, mAP@0.5 = 0.986, and mAP@0.5:0.95 = 0.965.")

body("Table 2 reports the per-class detection performance. Both classes achieve strong results, with AP@0.5 of 0.976 for healthy_leaf and 0.997 for diseased_leaf (AP@0.5:0.95 of 0.958 and 0.973, respectively). The near-perfect recall for diseased leaves (0.999) confirms that YOLOv7 effectively distinguishes disease symptom types (late blight lesions, early blight target spots, septoria speckles) despite their intra-class variance, while the lower recall for healthy leaves (0.901) reflects occasional confusion between subtle early-stage symptoms and healthy tissue.")

insert_image("fig13_yolo_metrics.png", caption_text="Figure 10. YOLOv7 training metrics on the PlantVillage 50% subset (150 epochs). The model achieves peak mAP@0.5 of 0.992 around epoch 25 and mAP@0.5:0.95 of 0.966 by epoch 148. Training-time validation used the initial three-class annotation; re-validation on the final two-class validation set gives mAP@0.5:0.95 = 0.965.")

# Table 2: YOLOv7 Per-class performance
make_table(
    ["Class", "Precision", "Recall", "F1-score", "AP@0.5", "AP@0.5:0.95"],
    [
        ["healthy_leaf", "0.927", "0.901", "0.914", "0.976", "0.958"],
        ["diseased_leaf", "0.892", "0.999", "0.942", "0.997", "0.973"],
        ["All classes", "0.91", "0.95", "0.930", "0.986", "0.965"],
    ],
    caption_text="Table 2. Per-class detection performance of YOLOv7 on the PlantVillage validation set (10,849 images, two-class annotation). F1-score is computed as the harmonic mean of precision and recall."
)

body("To complement the validation results, Table 3 reports the formal metrics of the 150-epoch model on the held-out test set (10,885 images), computed with the same confidence (0.001) and IoU (0.65) thresholds. The test set achieves mAP@0.5 of 0.985 (mAP@0.5:0.95 = 0.964), within 0.001 of the validation mAP@0.5 (0.986), indicating no meaningful training–validation–test gap. The per-class pattern mirrors the validation set: diseased_leaf retains near-perfect recall (0.998), while healthy_leaf shows the lower recall (0.880) attributed above to early-symptom confusion. Supplementary Figures S24–S26 present the corresponding test-set precision–recall, F1-confidence, and confusion-matrix curves.",
    bold_prefix="Test-set evaluation. ")

# Table 3: YOLOv7 test-set performance
make_table(
    ["Class", "Images", "Labels", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
    [
        ["healthy_leaf", "10,885", "3,025", "0.932", "0.880", "0.974", "0.958"],
        ["diseased_leaf", "10,885", "7,860", "0.896", "0.998", "0.996", "0.970"],
        ["All classes", "10,885", "10,885", "0.914", "0.939", "0.985", "0.964"],
    ],
    caption_text="Table 3. YOLOv7 detection performance on the held-out PlantVillage test set (10,885 images, two-class annotation). The test-set metrics are consistent with the validation-set results of Table 2 (mAP@0.5 of 0.985 vs. 0.986), confirming that the model does not overfit to the validation split."
)

body("YOLOv7 with COCO pre-training achieves robust leaf-level disease detection on laboratory-quality agricultural imagery. The final mAP@0.5 of 0.986 (mAP@0.5:0.95 = 0.965) compares favourably with published YOLO-family results on plant disease datasets, which typically report mAP@0.5 in the range 0.78–0.85 [4,8]. This performance gain is attributable to the controlled laboratory imaging conditions (uniform lighting, clear disease symptoms) and the high-quality YOLO conversion bounding boxes. However, this performance should not be expected to generalise directly to field-captured images without additional domain adaptation (see Section 6). The extended 150-epoch training reveals that while precision and recall oscillate in later epochs—a known characteristic of YOLO-family training—the mAP@0.5:0.95 continues to improve gradually, suggesting that longer training benefits fine-grained localisation quality. In terms of real-time performance, the trained model processes a 256 × 256 image in 0.9 ms of inference plus 0.7 ms of non-maximum suppression (1.6 ms total) at batch size 32 on the RTX 3090, corresponding to more than 600 frames per second—well beyond the real-time threshold required for UAV-based field scouting. For the downstream CPN simulation, these accuracy levels provide reliable disease severity estimates (d_disease = n_diseased / n_leaves) that propagate high-quality state information into the process-level predictions.")

body("The reported metrics are from a single training run of the 150-epoch configuration; replicate runs with controlled seeds were not performed owing to the ~6.6 h training cost. The training curves (Figure 10) show the metrics fluctuating within a stable band over the final 25 epochs rather than drifting, indicating convergence at the reported operating point; formal variance quantification across seeds is acknowledged as a limitation in Section 6.",
    bold_prefix="Single-run statement. ")

# Comparison: 10% vs 50% subset
body("To investigate the impact of training data scale, Table 4 compares the 50% subset results with a 10% subset experiment (10 epochs, ~8,687 images) trained under the same hyperparameter schedule. The 10% subset achieves a higher mAP@0.5 (0.993 vs. 0.986) but a substantially lower mAP@0.5:0.95 (0.886 vs. 0.965). This pattern is expected: on the small, uniform 10% subset, the model quickly overfits to easy-to-detect samples, achieving high standard mAP, while the additional training on the larger 50% subset improves localisation precision under stricter IoU requirements. The 10% subset also exhibits higher recall (0.978 vs. 0.95) but lower precision (0.872 vs. 0.91), indicating that the smaller sample induces more aggressive predictions that recover additional true positives at the cost of extra false positives. These results suggest that the 10% subset is adequate for rapid prototyping and hyperparameter tuning, while the 50% subset is necessary for robust generalisation.",
    bold_prefix="Training data scale analysis. ")

make_table(
    ["Configuration", "Epochs", "Samples", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
    [
        ["10% subset", "10", "~8,687", "0.872", "0.978", "0.993", "0.886"],
        ["50% subset", "150", "~43,438", "0.91", "0.95", "0.986", "0.965"],
    ],
    caption_text="Table 4. Comparison of YOLOv7 performance across training subset sizes on the identical validation set (10,849 images). The 10% subset achieves higher mAP@0.5 with rapid convergence, while the 50% subset achieves superior mAP@0.5:0.95 through extended training."
)

body("To place these results in context, we fine-tuned a YOLOv8s detector [42] on the same two-class dataset (256 × 256 input, batch 128, 20 epochs, ~0.65 h on the same RTX 3090) using its standard recipe with the same seed, and evaluated it on the identical validation and test splits. Table 5 compares the two architectures. YOLOv8s attains mAP@0.5 of 0.994 and mAP@0.5:0.95 of 0.982 on validation (test set: 0.994/0.982)—marginally above YOLOv7 (0.986/0.965 and 0.985/0.964)—at roughly one-tenth of the training time, although on the full 86,876-image training set rather than the 50% subset; its test-set per-class results are healthy_leaf P/R/mAP@0.5 = 0.988/0.999/0.994 and diseased_leaf = 0.999/0.989/0.995. The comparison shows that the two-class leaf task is well within the capacity of both mainstream architectures, and that the YOLOv7 configuration reported in this paper represents a conservative, compute-limited choice rather than a performance ceiling. Part of the precision gap is architectural: YOLOv8s's anchor-free detection head converges smoothly on this two-class task under its standard recipe, whereas the reported YOLOv7 configuration trades some precision for its compute-limited 50%-subset schedule. OCPPP retains the YOLOv7 detector because the framework requirements of Section 4.2.1 (object-level outputs feeding the CPN token model) are architecture-independent: any detector achieving comparable accuracy can be substituted without modifying the downstream process model.",
    bold_prefix="Architecture comparison. ")

# Table 5: YOLOv7 vs YOLOv8s comparison
make_table(
    ["Model", "Epochs", "Training images", "Val P", "Val R", "Val mAP@0.5", "Val mAP@0.5:0.95", "Test mAP@0.5", "Test mAP@0.5:0.95"],
    [
        ["YOLOv7 (this work)", "150", "~43,438 (50% subset)", "0.91", "0.95", "0.986", "0.965", "0.985", "0.964"],
        ["YOLOv8s [42]", "20", "~86,876 (full set)", "0.995", "0.995", "0.994", "0.982", "0.994", "0.982"],
    ],
    caption_text="Table 5. Comparison of YOLOv7 and YOLOv8s on the PlantVillage two-class validation and test sets (10,849 and 10,885 images). Both models use 256 × 256 input and batch 128; YOLOv8s was fine-tuned for 20 epochs on the full training set. Validation per-class results for YOLOv8s: healthy_leaf 0.991/0.999/0.994, diseased_leaf 1.000/0.990/0.995."
)

insert_image("fig14_yolo_detection.jpg", caption_text="Figure 11. YOLOv7 detection samples on the PlantVillage test set. Green bounding boxes indicate healthy leaves; red boxes indicate diseased leaves. Confidence scores are shown above each bounding box.")

body("Beyond the formal test metrics of Table 3, deployment-style inference over the test set at confidence threshold 0.25 produced 3,080 healthy_leaf and 7,930 diseased_leaf detections (72.0% diseased, reflecting the dataset composition) with mean confidence of 0.845 and 0.888, respectively. These aggregate detection statistics—rather than per-image point estimates—are what the CPN simulation consumes as scenario inputs (Section 5.4.2 and the walk-through of Section 5.6). Figure 11 presents qualitative detection samples from this inference.",
    bold_prefix="Test-set detection statistics. ")

heading("5.3 Experiment 2: Temperature Forecasting with LSTM on NASA POWER Data", level=2)

body("The OCPPP framework relies on predicted environmental variables to parameterise the CPN simulation. Among the six meteorological variables, temperature is the most critical for agronomic decision-making: it governs irrigation scheduling via evapotranspiration, frost risk assessment, and crop growth stage progression. Temperature also exhibits the strongest predictable seasonal and synoptic-scale signal, making it the most suitable variable for medium-range forecasting with an LSTM model.")

body("The LSTM was trained in two configurations to evaluate the impact of training data duration. The 5-year baseline model (1,258 sequences, January 2020–July 2023) achieved a day-1 R² = 0.884 with MAE = 2.36 °C on the test set (April–December 2024). The 44-year model (~12,500 sequences, January 1981–December 2016) achieved a day-1 R² = 0.949 with MAE = 1.69 °C and RMSE = 2.21 °C on 1,588 held-out test sequences (August 2020–December 2024), a 7.3% relative R² improvement; across the full 7-day window, temperature MAE is 2.50 °C (RMSE = 3.27 °C, R² = 0.888). Figure 12 shows the prediction on the 44-year model test set. The ground truth exhibits a seasonal decline from approximately 31 °C to −6 °C with synoptic-scale fluctuations of 5–10 °C amplitude. The LSTM tracks both the seasonal envelope and short-term variability with high fidelity. Table 6 summarises the multi-horizon forecast accuracy of this model. All LSTM results are from a single trained model per configuration; the multi-horizon and multi-variable analyses (Tables 6 and 7, Supplementary Figure S2) provide a broader assessment than point estimates, and seed-level variance quantification is acknowledged as a limitation in Section 6.")

body("Error degrades gracefully with forecast horizon: Day 1 MAE = 1.69 °C, Day 3 = 2.24 °C, Day 7 = 2.90 °C. This gradual degradation—rather than a sharp drop-off—indicates that the model has learned robust seasonal and synoptic-scale temporal patterns rather than relying on short-term memory alone. To contextualise these errors, we compare the LSTM against two standard reference baselines evaluated on identical test windows: (i) a persistence forecast, which predicts y(t+h) = y(t) using the last observed temperature, and (ii) a seasonal climatology forecast, which predicts the training-period mean temperature for each calendar day.")

# Table 4: LSTM Performance
make_table(
    ["Metric", "Value"],
    [
        ["MAE", "1.69 °C"],
        ["RMSE", "2.21 °C"],
        ["R²", "0.949"],
        ["Training data period", "44 years (1981–2024)"],
        ["Training sequences", "~12,500"],
        ["Test sequences", "~1,588"],
        ["Day 1 MAE", "1.69 °C"],
        ["Day 3 MAE", "2.24 °C"],
        ["Day 7 MAE", "2.90 °C"],
    ],
    caption_text="Table 6. LSTM temperature prediction performance on the NASA POWER test set (44-year model, one-day horizon; the full 7-day-window MAE is 2.50 °C)."
)

body("To contextualise the errors further, Table 7 reports the MAE at each forecast horizon.")

# Table 5: Multi-horizon baseline comparison (REAL computed values)
make_table(
    ["Forecast horizon", "LSTM MAE (°C)", "Persistence MAE (°C)", "Climatology MAE (°C)", "LSTM vs persistence"],
    [
        ["Day 1", "1.69", "1.54", "2.63", "−9.7%"],
        ["Day 2", "2.24", "2.40", "2.63", "+6.8%"],
        ["Day 3", "2.48", "2.84", "2.63", "+12.5%"],
        ["Day 4", "2.63", "3.09", "2.63", "+14.7%"],
        ["Day 5", "2.74", "3.23", "2.63", "+15.2%"],
        ["Day 6", "2.82", "3.37", "2.63", "+16.2%"],
        ["Day 7", "2.90", "3.49", "2.63", "+16.9%"],
    ],
    caption_text="Table 7. Multi-horizon temperature forecast comparison on the NASA POWER test set (n = 1,588 windows). Positive 'LSTM vs persistence' indicates the LSTM has lower MAE; negative indicates persistence is better."
)

body("Table 7 reveals a nuanced picture that tempers a naive reading of the single-value MAE. At the one-day horizon, a persistence forecast is a remarkably strong baseline (MAE = 1.54 °C), slightly outperforming the LSTM (1.69 °C), because daily temperature is strongly autocorrelated and consecutive-day changes are small. The LSTM's advantage emerges at multi-day horizons: from Day 2 onward it outperforms persistence, and by Day 7 it reduces MAE by 16.9% (2.90 vs. 3.49 °C). This is the horizon range most relevant to agronomic scheduling (irrigation and frost-protection decisions are typically planned 2–7 days ahead), precisely where persistence degrades rapidly. Seasonal climatology, by contrast, remains competitive at long lead times (MAE ≈ 2.63 °C at all horizons) because multi-day point forecasts revert toward the seasonal mean; the LSTM is nonetheless preferable for the downstream CPN simulation because Algorithm 3 requires a daily temperature trajectory to locate threshold-crossing times, which a flat climatological mean cannot provide. These results indicate that the LSTM adds genuine predictive skill over persistence at operational lead times, while also underscoring that no single baseline dominates across all horizons—motivating the hybrid combination of LSTM and climatology used for frost prediction in Section 5.5.")

insert_image("fig15_lstm_prediction.png", caption_text="Figure 12. LSTM temperature prediction on NASA POWER data (44-year model). The red line shows the LSTM prediction; the black line shows ground truth. The 95% confidence interval widens during rapid temperature changes. A reference line at 0°C is shown. Day-1 metrics: R² = 0.949, MAE = 1.69°C.")

# Multi-variable discussion
body("Although the LSTM was trained as a multi-variate model (six inputs, six outputs), only temperature (day-1 R² = 0.949) and surface pressure (day-1 R² = 0.872) achieved satisfactory accuracy. Relative humidity (R² = 0.485) and wind speed (R² = 0.216) showed moderate-to-limited predictability, while precipitation (R² = 0.173) and solar radiation (R² = −0.454) were essentially unpredictable—precipitation is inherently intermittent and localised, and shortwave radiation at the surface is dominated by local cloud cover. These results reflect fundamental atmospheric predictability limits rather than LSTM shortcomings. For this reason, OCPPP relies on the LSTM only for temperature prediction; other precursor conditions use climatological statistics (Section 5.5). Supplementary Figures S1 and S2 show the model's training dynamics and the predicted-versus-actual scatter plots for all six variables. Transformer-based time-series architectures have recently matched or exceeded recurrent baselines on long-sequence benchmarks [32,33]; for the present daily agrometeorological series with strong seasonality and ~12,500 training sequences, the LSTM remains a competitive and computationally light baseline, and transformer variants are an explicit direction of future work (Section 6).",
    bold_prefix="Multi-variable prediction discussion. ")

heading("5.4 Experiment 3: CPN-Based Scheduling Simulation", level=2)
body("To demonstrate the integrated CPN simulation capability of the OCPPP framework, we implemented four agricultural scheduling scenarios using the cpnpy library [27]. Each scenario combines measured inputs with scenario assumptions: measured inputs comprise the NASA POWER 2024 temperature record (Schemes A and D), the 44-year frost climatology (Scheme C), and YOLOv7 PlantVillage detection statistics (Scheme B); scenario assumptions comprise field sizes, operation durations, resource counts, and the illustrative disease and frost-risk levels, which are held fixed across all simulated strategies within a scenario so that strategy comparisons remain controlled. Concretely, the assumptions shared by all four schemes are: (i) transit and setup times between operations are zero, so the reported totals contain no constant overhead; (ii) operation durations are fixed per scenario—irrigation via the ET₀ derivation of Section 5.4.1, scouting 15 min per field, and spraying 15–60 min according to the severity band; (iii) resource counts (one shared pump, one or two drones) are scenario parameters; and (iv) all fields are ready at time zero. Because these assumptions are held fixed across the strategies compared within each scenario, they leave the relative rankings and the reported savings unaffected. The complete execution trace for the proof-of-concept scenario is provided in Appendix A (Table A1).")

body("The scheduling problems posed above are deliberately small—six irrigation orderings in Scheme A, five allocation strategies in Scheme B—and classical constraint programming (CP) solvers or greedy heuristics would solve them in negligible time. The incremental value of the CPN formulation is therefore not solver efficiency but (i) structural interpretability, since every result is accompanied by an execution trace that explains waiting times and resource occupancy (Sections 5.4.1–5.4.4), and (ii) the natural coupling with the AI modules: the YOLOv7 and LSTM outputs enter the net as token values without reformulation, and Algorithm 3's precursor intersection is the net's native AND-join semantics, so prediction and explanation share a single formalism rather than pairing an external solver with a black-box predictor.",
    bold_prefix="Relation to classical optimisation. ")

heading("5.4.1 Scheme A: Irrigation Scheduling", level=3)
body("Three fields (A: 3 ha, B: 2 ha, C: 1.5 ha) share a single irrigation pump, with irrigation durations derived from the Hargreaves ET₀ equation [28] and the NASA POWER 2024 spring temperature record. ET₀ is computed as ET₀ = 0.0023 · Ra · √(T_max − T_min) · (T_mean + 17.8), with the radiation term Ra = 35 (April value) and the diurnal range fixed at 10 °C; the resulting 2024 spring mean is 10.91 mm/day against an early-April reference of 9.11 mm/day. Irrigation durations are set as 45 min (field A, at the reference ET₀) scaled by field-specific ET₀ factors of 1.0, 0.7, and 0.5, yielding 45, 31, and 22 min for fields A, B, and C after rounding; the 45-min base duration corresponds, at an assumed application rate of 10 mm/h, to a 7.5 mm irrigation event over the 3 ha field (scenario assumption). After irrigation, each field continues with fertilisation, scouting, and spraying (A: 30/20/25 min; B: 20/14/17 min; C: 15/10/13 min), which run in parallel across fields. The CPN model enumerates all six irrigation orderings (Figure 13). The optimal ordering A→B→C completes in 136 minutes, while the worst ordering C→B→A requires 173 minutes—a 21.4% improvement purely from ordering optimisation. Of the 136 min, 98 min is pump time (45 + 31 + 22); the remaining 38 min is the post-irrigation tail of the last field, and no transit or setup time is modelled (a constant offset would leave the ranking of the orderings unchanged). The CPN execution trace reveals the causal mechanism: the largest field (A, 45 min irrigation) should be scheduled first because deferring it delays all three fields' downstream operations.")

insert_image("fig16_cpn_irrigation.png", caption_text="Figure 13. Scheme A: Irrigation scheduling. (a) Total completion time for all 6 irrigation orderings. (b) Gantt chart for the optimal sequence A→B→C (136 min), showing irrigation (pump) plus each field's parallel post-irrigation operations (fertilise, scout, spray). (c) Gantt chart for the worst sequence C→B→A (173 min). The 21.4% time savings arise purely from ordering optimisation.")

heading("5.4.2 Scheme B: Pest Control Resource Allocation", level=3)
body("Two fields (A: severe disease, 45 min spray; B: mild, 15 min spray) require scouting and spraying. The field-level disease ratios are scenario-level instantiations consistent with the aggregate test-set detection statistics of Section 5.2 (72.0% diseased), mapped through the disease ratio d_disease as: <25% mild, 25–50% moderate, 50–75% severe, and >75% very severe. Five strategies vary the number of drones (1 vs. 2) and the operation order (Figure 14). With a single drone, all four ordering strategies require 90 minutes—the bottleneck is the drone, not the order. Adding a second drone reduces total time to 60 minutes (33.3% improvement). This demonstrates the CPN's bottleneck identification capability: the simulation trace explicitly shows that ordering optimisation is ineffective when resource capacity is the limiting factor.")

insert_image("fig17_cpn_pest.png", caption_text="Figure 14. Scheme B: Pest control resource allocation. (a) Total completion time of the five strategies (S1–S4: single drone with different orderings; S5: two parallel drones). (b) Gantt chart with one drone (S1, 90 min). (c) Gantt chart with two drones (S5, 60 min). The 33.3% reduction highlights the bottleneck-identification capability of CPN simulation.")

heading("5.4.3 Scheme C: Frost Emergency Response", level=3)
body("Three fields (A: 55% frost risk, B: 42%, C: 29%) face an impending frost event predicted by Algorithm 3. Two protection resources are available with assumed effectiveness levels—covers (100%) and anti-frost agents (80%)—both scenario assumptions per the transparency list of Section 5.4. The CPN model evaluates four allocation strategies (Figure 15). The optimal strategy (S1: Covers→A,B, AF→C) protects 724 crop-value units—scenario units defined as each field's assumed value (300, 280 and 180 for A, B and C) multiplied by its achieved protection coverage—while the worst (S2: double-protect A) protects only 580 units, a 24.8% difference. The CPN reveals that concentrating resources on the highest-risk field provides diminishing returns, while distribution across all three fields maximises total value.")

insert_image("fig18_cpn_frost.png", caption_text="Figure 15. Scheme C: Frost emergency response. (a) Protection value for the four resource allocation strategies. (b) Gantt chart of the optimal strategy (S1: Covers→A,B, AF→C). (c) Coverage effectiveness by field and strategy. The optimal strategy (S1) achieves 24.8% higher protection value than the worst (S2).")

heading("5.4.4 Scheme D: Full-Season Resource Configuration", level=3)
body("A full growing season of winter wheat (1 April–30 June 2024, 91 days) is simulated using NASA POWER daily temperature data and the GDD model [29]. Growing degree days are accumulated from 1 April (approximate green-up) with a base temperature of 0 °C, following the standard winter-wheat GDD convention [39,40]; regional wheat phenology studies in China report stage durations of the same order for this cropping system [44]. Growth stages are triggered at cumulative GDD thresholds, each activating the corresponding agronomic operations: jointing (200 °C·d) triggers irrigation, booting (400 °C·d) triggers fertilisation, heading (600 °C·d) triggers scouting and spraying, grain filling (800 °C·d) triggers a second irrigation, and maturity (1,000 °C·d) triggers harvest. Four resource configurations are compared (Table 8). The baseline (1P+1D) requires 985 minutes with 5 conflict days. Adding a second pump saves 120 minutes, while adding a second drone saves 50 minutes. The optimal configuration (2P+2D) achieves 815 minutes—a 17.3% reduction. Bottleneck analysis confirms that pump conflicts are the primary constraint.")

# Table 6: CPN Scheme D results
make_table(
    ["Configuration", "Work (min)", "Wait (min)", "Total (min)", "Conflict Days"],
    [
        ["1P + 1D (baseline)", "645", "340", "985", "5"],
        ["2P + 1D", "645", "220", "865", "5"],
        ["1P + 2D", "645", "290", "935", "5"],
        ["2P + 2D (optimal)", "645", "170", "815", "5"],
    ],
    caption_text="Table 8. Full-season resource configuration results (Scheme D, winter wheat, 1 April–30 June 2024). P = pump, D = drone. The optimal configuration achieves 17.3% reduction over baseline."
)

heading("5.5 Experiment 4: Low-Frequency Frost Prediction", level=2)
body("Algorithm 3 is evaluated for frost prediction using a hybrid approach. The temperature precursor is predicted by the LSTM (Section 5.3), while wind speed and humidity precursors use climatological statistics from the 44-year NASA POWER record. A frost event is defined as a day on which the LSTM-predicted temperature falls below 0°C and both climatological probabilities exceed 0.7 (a deliberately conservative threshold; its sensitivity is left to future work). The 0 °C daily-mean definition is strict by design: Scheme C (Section 5.4.3) instead uses a T2M < 2 °C proxy for radiative frost nights, whose calm, clear conditions narrow the diurnal range so that sub-zero minima are expected before the daily mean reaches 0 °C; temperature thresholds of this kind are crop- and stage-dependent [43]. Over the test period (August 2020–December 2024, 1,588 days), 24 actual frost events occurred. Temperature-only thresholding achieved a 41.7% hit rate (10 correct, 14 false alarms). Adding climatological wind and humidity constraints reduced false alarms to 9 but also reduced hits to 7 (29.2% hit rate, 43.8% precision). Supplementary Figures S3 and S4 present the seasonal frost-precursor climatology and the resulting frost prediction performance.")

body("The 17 missed events had two primary causes: 12 occurred when cold fronts arrived 2–3 days earlier or later than the LSTM predicted, shifting the temperature threshold crossing; 5 involved borderline temperatures (±1°C) where prediction errors within the model's 1.69°C MAE were sufficient to cause misses. The 9 false alarms predominantly occurred in early spring when climatological conditions favoured frost but the LSTM underestimated solar radiation warming.")

body("The agronomic meaning of the 0 °C threshold depends on the seasonal timing of sub-zero days. In the Hefei record used here, sub-zero days are concentrated in December–February, when winter wheat is dormant and cold-hardened, so that mid-winter frosts rarely translate into production losses; the consequential events are the marginal-season ones—late-autumn frosts before full hardening (e.g., 30 November 2022) and early-spring frosts after de-hardening near green-up, which are rare in this record (the 2024 season reached only a marginal daily mean of 1.17 °C on 1 March under the T2M < 2 °C criterion of Scheme C) [41]. The 0 °C threshold therefore flags a mixture of low-impact dormant-season days and the few high-impact marginal-season events; for the latter, the 7-day lead time of the precursor-condition pipeline is precisely the window in which protective measures (covers, anti-frost agents, irrigation) can be deployed, as simulated in Scheme C. Because sub-zero days in this record are concentrated in the dormant December–February window, the operational value of the predictor is concentrated in the rarer marginal-season events.",
    bold_prefix="Agronomic significance. ")

heading("5.6 End-to-End Demonstration of the Integrated Framework", level=2)
body("Sections 5.2–5.5 evaluate each module individually. To demonstrate that the modules compose into a single predictive pipeline, this section walks through a minimal end-to-end scenario that chains real model outputs into a CPN scheduling prediction. Consistent with the transparency statement of Section 5.4, inputs derived from measured data are identified as such, and the remaining quantities are scenario assumptions held fixed within the example.")

body("A drone scouting flight captures 100 leaf images over a wheat field. The two-class YOLOv7 model processes each 256 × 256 image and returns 72 diseased_leaf and 28 healthy_leaf detections, with mean confidence of 0.888 and 0.845, respectively; these figures are drawn from the aggregate detection statistics of the full test-set inference (72.0% diseased, mean confidence 0.8884/0.8448, Section 5.2). The disease ratio of the surveyed field is therefore d_disease = 72/100 = 0.72 (measured input).",
    bold_prefix="Step 1 (Module 2, vision). ")

body("Applying the mapping of Section 5.4.2, d_disease = 0.72 falls in the severe band (50–75%), which prescribes a 45-minute spray operation for the field.",
    bold_prefix="Step 2 (severity mapping). ")

body("The 44-year LSTM issues a 7-day temperature forecast for the operation window; in the illustrative mid-April 2024 forecast used here (scenario assumption), predicted daily means range from 15.5 °C to 21.8 °C—a window without frost risk and with suitable spraying conditions.",
    bold_prefix="Step 3 (Module 2, time series). ")

body("Algorithm 1 assembles a CPN from the pattern library: the field's workflow is Scout → Spray, with the spray duration set to 45 min by the severity token. A second field with mild disease (15%, 15-min spray) is included, and the two fields share a single spray drone modelled as a resource place with one token.",
    bold_prefix="Step 4 (Module 1). ")

body("The CPN simulates the four candidate strategies of Scheme B. All single-drone orderings complete in 90 min, and the execution trace of the optimal ordering (Table 10) shows the drone occupied without interruption, identifying it as the bottleneck. Adding a second drone reduces the predicted completion time to 60 min (33.3% reduction)—the recommendation the framework would surface to the farm manager together with the trace that justifies it.",
    bold_prefix="Step 5 (Module 3). ")

make_table(
    ["Step", "Module", "Input", "Output"],
    [
        ["1", "Module 2 (YOLOv7)", "100 leaf images (256 × 256)", "72 diseased / 28 healthy detections (d_disease = 0.72)"],
        ["2", "Severity mapping (Section 5.4.2)", "d_disease = 0.72", "Severe → 45-min spray operation"],
        ["3", "Module 2 (LSTM)", "44-year temperature history", "7-day forecast (15.5–21.8 °C, no frost risk)"],
        ["4", "Module 1 (Algorithm 1)", "Operation set + pattern library", "CPN with shared-drone resource constraint"],
        ["5", "Module 3 (CPN simulation)", "CPN + resource tokens", "90 min (1 drone) → 60 min (2 drones, −33.3%)"],
    ],
    caption_text="Table 9. Step-by-step summary of the end-to-end walk-through, showing how each module's output feeds the next. Measured inputs are marked in the text; temperature values are scenario assumptions."
)

make_table(
    ["Interval (min)", "Operation", "Duration (min)", "Field"],
    [
        ["0–15", "Scout_A", "15", "A"],
        ["15–60", "Spray_A", "45", "A"],
        ["60–75", "Scout_B", "15", "B"],
        ["75–90", "Spray_B", "15", "B"],
    ],
    caption_text="Table 10. CPN execution trace of the end-to-end walk-through (single drone, optimal ordering A → B). The drone is occupied without interruption, revealing it as the scheduling bottleneck; total completion time is 90 min."
)

body("This walk-through instantiates the full OCPPP data flow—image modality → severity token → CPN simulation → completion-time prediction with an inspectable execution trace—using real model outputs where available. A live deployment that automates these steps on streaming field data remains future work (Section 6).")

heading("5.7 Summary of Experimental Findings", level=2)
body("The experimental results validate three core claims of the OCPPP framework:")

body("1. Individual AI components achieve strong performance on real agricultural data. YOLOv7 achieves mAP@0.5 of 0.986 (mAP@0.5:0.95 = 0.965) on the PlantVillage 50% subset, with precision of 0.91 and recall of 0.95 across two detection classes (healthy and diseased leaves), while processing each 256 × 256 image in 1.6 ms; the same model attains mAP@0.5 of 0.985 (mAP@0.5:0.95 = 0.964) on the held-out 10,885-image test set, and a YOLOv8s baseline fine-tuned on the same two-class data reaches mAP@0.5 of 0.994 (mAP@0.5:0.95 = 0.982), confirming that the task is robust across architectures. The LSTM achieves a day-1 R² of 0.949 (MAE = 1.69 °C) for temperature forecasting on the 44-year NASA POWER record, a 7.3% improvement over the 5-year baseline. Both components operate on real, publicly available datasets and produce results consistent with or exceeding published benchmarks [4,8,19].")

body("2. CPN-based simulation provides interpretable, actionable scheduling insights. Across four agricultural scenarios, the CPN simulation identified resource bottlenecks (Scheme B: drone count), quantified the impact of scheduling decisions (Scheme A: 21.4% time savings through optimal ordering), evaluated emergency response strategies (Scheme C: 24.8% improvement in frost protection value), and optimised full-season resource configurations (Scheme D: 17.3% reduction with dual-pump dual-drone setup). In each case, the CPN execution trace provides causal transparency that enables farm managers to understand not just what the optimal schedule is, but why it is optimal.")

body("3. Low-frequency event prediction via precursor reasoning is feasible. Algorithm 3 achieves a 41.7% frost hit rate with temperature-only thresholding on the 2020–2024 test period. The performance is limited by the intrinsic difficulty of 7-day near-freezing temperature forecasting, but the precursor-condition approach provides a principled method for predicting rare, high-impact events that direct data-driven methods cannot address.")

# ==========================================================================
# 6. CONCLUSION AND FUTURE WORK
# ==========================================================================
heading("6. Conclusion and Future Work")

body("This paper proposed the Object-Centric Process Prediction Framework (OCPPP), a novel approach integrating coloured Petri net (CPN) process modelling with multi-modal deep learning for interpretable prediction in precision crop management. The framework addresses a critical gap in existing agricultural predictive methods: black-box deep learning models lack transparency and cannot reason about process-level interdependencies, while mechanistic crop models lack the flexibility to incorporate real-time multi-modal sensor data.")

body("The experimental evaluation across four experiments and an end-to-end demonstration yielded the following results. For vision-based crop health estimation, YOLOv7 achieved mAP@0.5 of 0.986 (mAP@0.5:0.95 = 0.965) on the PlantVillage 50% subset, with 0.91 precision and 0.95 recall across two classes, and mAP@0.5 of 0.985 (mAP@0.5:0.95 = 0.964) on the held-out 10,885-image test set; a YOLOv8s baseline (0.994/0.982) fine-tuned on the same data confirms the tractability of the two-class task across architectures. For environmental time-series forecasting, the LSTM predictor achieved a day-1 R² of 0.949 (MAE = 1.69 °C) on the 44-year NASA POWER record, a 7.3% improvement over the 5-year baseline (R² = 0.884). For integrated CPN scheduling, four agricultural scenarios parameterised with real meteorological data demonstrated quantifiable benefits: 21.4% time savings through irrigation ordering (Scheme A), 33.3% reduction through dual-drone pest control (Scheme B), 24.8% improvement in frost protection value (Scheme C), and 17.3% reduction in full-season completion time (Scheme D). Algorithm 3 demonstrated low-frequency frost prediction with a temperature-only hit rate of 41.7%.")

body("Beyond numerical results, the OCPPP framework offers three qualitative advantages: (1) interpretability by design—every prediction is accompanied by an inspectable CPN execution trace recording which operations fired, in what order, at what times, and which resources they consumed; (2) modularity—individual AI models in Module 2 can be replaced with state-of-the-art alternatives without modifying the process-level reasoning in Module 3; and (3) low-frequency event support—Algorithm 3 provides a principled method for predicting rare but high-impact events through precursor condition reasoning, transforming a data-scarce problem into a data-sufficient one.")

body("Several limitations should be acknowledged. First, the CPN model construction in Algorithm 1 assumes acyclic dependency graphs, which may require extension for cyclical processes such as iterative pest monitoring or multi-season crop rotations. Second, while each AI component was validated individually on real datasets and the integrated pipeline was demonstrated as a walk-through (Section 5.6), a real-time deployment that automates live YOLO inference, LSTM forecasting, and CPN simulation on streaming field data remains as future work. Third, the CPN scheduling scenarios use deterministic operation durations derived from meteorological data; stochastic extensions incorporating weather forecast uncertainty could improve robustness. Fourth, validation across diverse crop-climate systems (e.g., tropical rice, Mediterranean olive) would strengthen generality. Fifth, both deep learning models were evaluated from single training runs; replicate runs with controlled seeds would strengthen variance quantification. Sixth, from a data security and model robustness perspective, the current framework does not address adversarial threats to the sensor data pipeline—for example, the impact of manipulated leaf images or spoofed temperature readings on downstream CPN predictions [30,31]. Integrating anomaly detection mechanisms for sensor data integrity and evaluating model robustness against adversarial perturbations would enhance the framework's reliability in operational settings. Seventh, the interpretability claims of this work rest on structural transparency and the trace-based analyses of Section 4.3.4 rather than on user studies or quantitative explanation-quality metrics (e.g., fidelity or stability); a formal evaluation with agronomic domain experts is a direction for future work. Finally, transformer-based architectures [33] as replacements for LSTM in the time-series prediction module, and vision transformer (ViT) models [34] as alternatives to YOLOv7 for crop state estimation, represent natural progressions as these technologies mature.")

body("In conclusion, OCPPP demonstrates that process-level structural interpretability and multi-modal deep learning are complementary rather than competing approaches in precision agriculture. By combining the structural transparency of coloured Petri nets with the predictive power of modern deep learning models, the framework provides agronomists and farm managers with predictions they can understand, trust, and act upon.")

# ==========================================================================
# MDPI BACK MATTER (declarations)
# ==========================================================================
heading("Author Contributions", level=2)

body("Conceptualization, C.S. and Q.W.; methodology, C.S.; software, C.S.; validation, C.S. and Q.W.; formal analysis, C.S.; investigation, C.S.; resources, C.S.; data curation, C.S.; writing—original draft preparation, C.S.; writing—review and editing, C.S. and Q.W.; visualization, C.S.; supervision, C.S.; project administration, C.S.; funding acquisition, C.S. All authors have read and agreed to the published version of the manuscript.",
    bold_prefix="")

heading("Funding", level=2)
body("This work was supported by the National Natural Science Foundation, China (No. 61572035, 61402011), the Project of Anhui Provincial Department of Education (No. 2023AH051865, 2025jyxm0266), the Anhui Agriculture Research System, the Project of Anhui Provincial Department of Science and Technology (No. 202204c06020065), and the Key Discipline Construction Project of Anhui Science and Technology University (No. XK-XJGY002, Xj2024037, Xj2025141, Xj2024051, 2024aijy292, FZ20250148).")

heading("Institutional Review Board Statement", level=2)
body("Not applicable.")

heading("Informed Consent Statement", level=2)
body("Not applicable.")

heading("Data Availability Statement", level=2)
body("The PlantVillage dataset analysed in this study is openly available from [25]. The NASA POWER meteorological data are openly available from the NASA POWER project [26]. The synthetic agronomic activity logs and the CPN simulation scripts generated during this study are available from the corresponding author on reasonable request.")

heading("Acknowledgments", level=2)
body("The authors gratefully acknowledge the funding agencies listed in the Funding section for their support.")

heading("Conflicts of Interest", level=2)
body("The authors declare no conflicts of interest.")

heading("Supplementary Materials", level=2)
body("The following supporting information can be downloaded at the website of this paper posted on MDPI: Figure S1: LSTM training and validation loss; Figure S2: predicted versus actual values for all six meteorological variables; Figure S3: seasonal frost-precursor climatology; Figure S4: frost prediction performance of the hybrid approach; Figures S5–S15: LSTM time-series diagnostics (residual distribution, scatter, seasonal patterns, multi-step errors, confidence intervals, learning-rate schedule, precipitation analysis, feature correlation, extreme-value analysis, illustrative diurnal patterns, monthly performance); Figures S16–S23: YOLOv7 detection diagnostics (complete training curves, precision–recall curve, F1-confidence curve, class distribution, confidence distribution, detection count per image, healthy-versus-diseased ratio, confusion matrix); Figures S24–S26: YOLOv7 test-set diagnostic curves (precision–recall curve, F1-confidence curve, confusion matrix).")

# ═══════════════════════════════════════════════════════════════════════════
# APPENDIX A: CPN Simulation Execution Trace
# ═══════════════════════════════════════════════════════════════════════════
heading("Appendix A: CPN Simulation Execution Trace", level=1, sb=Pt(16))

body("This appendix presents the full execution trace from the cpnpy proof-of-concept simulation (Section 5.4). The trace demonstrates the interpretability advantage of CPN-based prediction: every operation, timing, and resource allocation is recorded and inspectable. This trace instantiates the shared-resource example of Section 3.2 (three fields, one irrigation pump); the 2.8 h/3.6 h figures in Section 4.3.2 refer to the separate unordered-operations example used to illustrate Algorithm 2.")

body("Simulation scenario: three fields (A, B, C) sharing a single irrigation pump, with sequential downstream operations (Fertilize → Scout → Spray). Fields have different irrigation durations based on their area (A: 3 ha, 45 min; B: 2 ha, 31 min; C: 1.5 ha, 22 min). The CPN simulation produces a deterministic execution trace as follows:")

# Execution trace table
make_table(
    ["Step", "Time", "Transition", "Duration", "End Time", "Field"],
    [
        ["1", "0 min", "Irrigate", "45 min", "45 min", "A"],
        ["2", "45 min", "Irrigate", "45 min", "90 min", "B"],
        ["3", "45 min", "Fertilize", "30 min", "75 min", "A"],
        ["4", "75 min", "Scout", "20 min", "95 min", "A"],
        ["5", "90 min", "Irrigate", "45 min", "135 min", "C"],
        ["6", "90 min", "Fertilize", "30 min", "120 min", "B"],
        ["7", "95 min", "Spray", "25 min", "120 min", "A"],
        ["8", "120 min", "Scout", "20 min", "140 min", "B"],
        ["9", "135 min", "Fertilize", "30 min", "165 min", "C"],
        ["10", "140 min", "Spray", "25 min", "165 min", "B"],
        ["11", "165 min", "Scout", "20 min", "185 min", "C"],
        ["12", "185 min", "Spray", "25 min", "210 min", "C"],
    ],
    caption_text="Table A1. Full CPN execution trace from the cpnpy proof-of-concept simulation. Total completion time: 210 min (3.5 h). The trace records each operation's start time, duration, and resource usage."
)

body("The execution trace reveals three key insights. First, the pump is the bottleneck resource: Field A uses it at time 0, Field B waits 45 min, Field C waits 90 min. Second, the CPN automatically parallelises operations where possible: Fertilize and Scout for Field A execute concurrently with Irrigate for Field B during the 45–90 min window. Third, the total completion time of 210 min (3.5 h) is a direct consequence of the resource contention pattern—a farm manager can inspect this trace to understand exactly why the schedule takes as long as it does.")

# ═══════════════════════════════════════════════════════════════════════════
# APPENDIX B: YOLOv7 Training Configuration Comparison
# ═══════════════════════════════════════════════════════════════════════════
heading("Appendix B: YOLOv7 Training Configuration Details", level=1, sb=Pt(16))

body("This appendix provides the complete hyperparameter configurations for both YOLOv7 training runs discussed in Section 5.2.")

heading("B.1 10% Subset Configuration (10 epochs)", level=2)
body("The 10% subset experiment used the same hyp.fast.yaml hyperparameter schedule as the 50% subset experiment (mosaic probability = 0.3, mixup probability = 0.05, scale jitter = 0.9, translation jitter = 0.2), so that the two configurations differ only in training data volume and epoch count. With this schedule, the 10% subset converges rapidly—10 epochs completed in approximately 10 minutes—reaching high mAP@0.5 but lower mAP@0.5:0.95 than the extended 150-epoch 50% run.")

heading("B.2 50% Subset Configuration (150 epochs)", level=2)
body("The 50% subset experiment used the hyp.fast.yaml hyperparameter schedule, designed for efficient training on larger datasets. Key parameters: mosaic augmentation probability = 0.3, mixup probability = 0.05, scale jitter = 0.9, translation jitter = 0.2. The lower augmentation probabilities reduce computational overhead, while the extended 150-epoch training schedule improves localisation precision under stricter IoU thresholds.")

# Table: Full hyperparameter comparison
make_table(
    ["Parameter", "10% Subset (hyp.fast)", "50% Subset (hyp.fast)"],
    [
        ["Epochs", "10", "150"],
        ["Training images", "~8,687", "~43,438"],
        ["Batch size", "128", "128"],
        ["Input size", "256 × 256", "256 × 256"],
        ["Optimiser", "SGD (momentum 0.937)", "SGD (momentum 0.937)"],
        ["Initial LR", "0.01 (cosine annealing)", "0.01 (cosine annealing)"],
        ["Weight decay", "5 × 10⁻⁴", "5 × 10⁻⁴"],
        ["Mosaic probability", "0.3", "0.3"],
        ["Mixup probability", "0.05", "0.05"],
        ["Scale jitter", "0.9", "0.9"],
        ["Translation jitter", "0.2", "0.2"],
        ["HSV jitter (H/S/V)", "0.015 / 0.7 / 0.4", "0.015 / 0.7 / 0.4"],
        ["Horizontal flip", "0.5", "0.5"],
        ["GPU peak memory", "~20.4 GB", "~20.8 GB"],
        ["Per-epoch time", "~1 min", "~2.6 min"],
        ["Total training time", "~10 min", "~6.6 h"],
    ],
    caption_text="Table A2. Complete hyperparameter comparison between the two YOLOv7 training configurations."
)

# ==========================================================================
# REFERENCES
# ==========================================================================
heading("References")

refs = [
    "1. Kamilaris, A.; Prenafeta-Boldú, F.X. Deep learning in agriculture: A survey. Comput. Electron. Agric. 2018, 147, 70–90.",
    "2. Khaki, S.; Wang, L. Crop yield prediction using deep neural networks. Front. Plant Sci. 2019, 10, 621.",
    "3. You, J.; Li, X.; Low, M.; Lobell, D.; Ermon, S. Deep Gaussian process for crop yield prediction based on remote sensing data. In Proceedings of the AAAI Conference on Artificial Intelligence, 2017; Vol. 31, pp. 4559–4566.",
    "4. Yang, G.; Wang, J.; Nie, Z.; Yang, H.; Yu, S. A lightweight YOLOv8 tomato detection algorithm combining feature enhancement and attention. Agronomy 2023, 13, 1824.",
    "5. Xiao, F.; Wang, H.; Xu, Y.; Zhang, R. Fruit detection and recognition based on deep learning for automatic harvesting: An overview and review. Agronomy 2023, 13, 1625.",
    "6. Wang, C.-Y.; Bochkovskiy, A.; Liao, H.-Y.M. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023; pp. 7464–7475.",
    "7. Kumar, C.; Mubvumba, P.; Huang, Y.; Dhillon, J.; Reddy, K. Multi-stage corn yield prediction using high-resolution UAV multispectral data and machine learning models. Agronomy 2023, 13, 1277.",
    "8. Zhang, L.; Ding, G.; Li, C.; Li, D. DCF-Yolov8: An improved algorithm for aggregating low-level features to detect agricultural pests and diseases. Agronomy 2023, 13, 2012.",
    "9. Bi, C.; Xu, S.; Hu, N.; Zhang, S.; Zhu, Z.; Yu, H. Identification method of corn leaf disease based on improved Mobilenetv3 model. Agronomy 2023, 13, 300.",
    "10. Jones, J.W.; Hoogenboom, G.; Porter, C.H.; Boote, K.J.; Batchelor, W.D.; Hunt, L.A.; Wilkens, P.W.; Singh, U.; Gijsman, A.J.; Ritchie, J.T. The DSSAT cropping system model. Eur. J. Agron. 2003, 18, 235–265.",
    "11. Miller, T. Explanation in artificial intelligence: Insights from the social sciences. Artif. Intell. 2019, 267, 1–38.",
    "12. Mostafa, S.; Mondal, P.P.; Panjvani, K.; Kochian, L.; Stavness, I. Explainable deep learning in plant phenotyping. Front. Artif. Intell. 2023, 6, 1203546.",
    "13. Tovany, J.L.; Ross-León, R.; Ruiz-León, J.; Ramírez-Treviño, A.; Begovich, O. Greenhouse modelling using continuous timed Petri nets. Math. Probl. Eng. 2013, 2013, 639306.",
    "14. Tax, N.; Verenich, I.; La Rosa, M.; Dumas, M. Predictive business process monitoring with LSTM neural networks. In Proceedings of the 29th International Conference on Advanced Information Systems Engineering (CAiSE), 2017; LNCS 10253, pp. 477–492.",
    "15. Mehdiyev, N.; Evermann, J.; Fettke, P. A novel business process prediction model using a deep learning method. Bus. Inf. Syst. Eng. 2020, 62, 143–157.",
    "16. van der Aalst, W.M.P. Object-centric process mining: Dealing with divergence and convergence in event data. In Proceedings of the 17th International Conference on Software Engineering and Formal Methods (SEFM), 2019; LNCS 11724, pp. 1–17.",
    "17. Alahmad, T.; Neményi, M.; Nyéki, A. Applying IoT sensors and big data to improve precision crop production: A review. Agronomy 2023, 13, 2603.",
    "18. Chandola, V.; Banerjee, A.; Kumar, V. Anomaly detection: A survey. ACM Comput. Surv. 2009, 41, 1–58.",
    "19. Yu, Y.; Si, X.; Hu, C.; Zhang, J. A review of recurrent neural networks: LSTM cells and network architectures. Neural Comput. 2019, 31, 1235–1270.",
    "20. Lesk, C.; Rowhani, P.; Ramankutty, N. Influence of extreme weather disasters on global crop production. Nature 2016, 529, 84–87.",
    "21. Vogel, E.; Donat, M.G.; Alexander, L.V.; Meinshausen, M.; Ray, D.K.; Karoly, D.; Meinshausen, N.; Frieler, K. The effects of climate extremes on global agricultural yields. Environ. Res. Lett. 2019, 14, 054010.",
    "22. Jensen, K.; Kristensen, L.M. Coloured Petri Nets: Modelling and Validation of Concurrent Systems; Springer, 2009.",
    "23. Antle, J.M.; Jones, J.W.; Rosenzweig, C. Next generation agricultural system models and knowledge products: Synthesis and strategy. Agric. Syst. 2017, 155, 269–280.",
    "24. Wang, M.; Sun, E. Modelling and analysis for tracing system of agricultural products based on coloured Petri net. In Proceedings of the 2021 IEEE 24th International Conference on Computer Supported Cooperative Work in Design (CSCWD), 2021; pp. 7–11.",
    "25. Hughes, D.P.; Salathé, M. An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060, 2015.",
    "26. Sparks, A.H. nasapower: A NASA POWER Global Meteorology, Surface Solar Energy and Climatology Data Client for R. J. Open Source Softw. 2018, 3, 1035.",
    "27. cpnpy: A Python library for Coloured Petri Net construction and simulation, version 0.2.1. Available online: https://pypi.org/project/cpnpy/ (accessed on 23 November 2026).",
    "28. Allen, R.G.; Pereira, L.S.; Raes, D.; Smith, M. Crop evapotranspiration: Guidelines for computing crop water requirements. FAO Irrigation and Drainage Paper 56, 1998.",
    "29. McMaster, G.S.; Wilhelm, W.W. Growing degree-days: One equation, two interpretations. Agric. For. Meteorol. 1997, 87, 291–300.",
    "30. Gao, Y.; Camtepe, S.A.; Sultan, N.H.; Bui, H.T.; Mahboubi, A.; Aboutorab, H.; Bewong, M.; Islam, R.; Islam, M.Z.; Chauhan, A.; et al. Security threats to agricultural artificial intelligence: Position and perspective. Comput. Electron. Agric. 2024, 227, 109557.",
    "31. Ferrag, M.A.; Shu, L.; Djallel, O.; Choo, K.-K.R. Deep learning-based intrusion detection for distributed denial of service attack in agriculture 4.0. Electronics 2021, 10, 1257.",
    "32. Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A.N.; Kaiser, Ł.; Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems; NeurIPS, 2017; Vol. 30.",
    "33. Wen, Q.; Zhou, T.; Zhang, C.; Chen, W.; Ma, Z.; Yan, J.; Sun, L. Transformers in time series: A survey. In Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI), 2023; pp. 6778–6786.",
    "34. Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. An image is worth 16×16 words: Transformers for image recognition at scale. In Proceedings of the International Conference on Learning Representations (ICLR), 2021.",
    "35. van der Aalst, W.M.P. Object-centric process mining: Unraveling the fabric of real processes. Mathematics 2023, 11, 2691.",
    "36. Teinemaa, I.; Dumas, M.; La Rosa, M.; Maggi, F.M. Outcome-oriented predictive process monitoring: Review and benchmark. ACM Trans. Knowl. Discov. Data 2019, 13, 17:1–17:57.",
    "37. Di Francescomarino, C.; Ghidini, C. Predictive process monitoring. In Process Mining Handbook; van der Aalst, W.M.P., Carmona, J., Eds.; Springer: Cham, Switzerland, 2022; pp. 320–346.",
    "38. Attri, I.; Awasthi, L.K.; Sharma, T.P.; Rathee, P. A review of deep learning techniques used in agriculture. Ecol. Inform. 2023, 77, 102217.",
    "39. Miller, P.; Lanier, W.; Brandt, S. Using Growing Degree Days to Predict Plant Stages; Montana State University Extension Service: Bozeman, MT, USA, 2001.",
    "40. Salazar-Gutierrez, M.R.; Johnson, J.; Chaves-Cordoba, B.; Hoogenboom, G. Relationship of base temperature to development of winter wheat. Int. J. Plant Prod. 2013, 7, 741–762.",
    "41. Snyder, R.L.; de Melo-Abreu, J.P. Frost Protection: Fundamentals, Practice and Economics; FAO Environment and Natural Resources Service Series; FAO: Rome, Italy, 2005; Volume 1.",
    "42. Jocher, G.; Chaurasia, A.; Qiu, J. Ultralytics YOLOv8 (Version 8.0.0). 2023. Available online: https://github.com/ultralytics/ultralytics (accessed on 23 November 2026).",
    "43. Luo, Q. Temperature thresholds and crop production: A review. Clim. Change 2011, 109, 583–598.",
    "44. Tao, F.; Zhang, S.; Zhang, Z. Spatiotemporal changes of wheat phenology in China under the effects of temperature, day length and cultivar thermal characteristics. Eur. J. Agron. 2012, 43, 201–212.",
]

for ref_text in refs:
    p = _make_paragraph(first_indent=Cm(0), space_after=Pt(1), space_before=Pt(0))
    _run(p, ref_text, size=Pt(9))

# ═══════════════════════════════════════════════════════════════════════════
# Abbreviations
heading("Abbreviations", level=2)

make_table(
    ["Abbreviation", "Full Form"],
    [
        ["OCPPP", "Object-Centric Process Prediction Framework"],
        ["CPN", "Coloured Petri Net"],
        ["YOLOv7", "You Only Look Once version 7"],
        ["LSTM", "Long Short-Term Memory"],
        ["CNN", "Convolutional Neural Network"],
        ["GDD", "Growing Degree Day"],
        ["ET₀", "Reference Evapotranspiration"],
        ["mAP", "Mean Average Precision"],
        ["MAE", "Mean Absolute Error"],
        ["RMSE", "Root Mean Square Error"],
        ["IoU", "Intersection over Union"],
        ["NASA POWER", "Prediction of Worldwide Energy Resources"],
        ["ARIMA", "Autoregressive Integrated Moving Average"],
        ["XAI", "Explainable Artificial Intelligence"],
        ["PBPM", "Predictive Business Process Monitoring"],
    ],
    caption_text=""
)
# ═══════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY MATERIALS (separate file)
# ═══════════════════════════════════════════════════════════════════════════
main_doc = doc  # keep a reference to the main manuscript before switching
supp = Document()
setup_document(supp)
set_doc(supp)

p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                    first_indent=Cm(0), space_before=Pt(0), space_after=Pt(6))
_run(p, "Supplementary Materials", bold=True, size=Pt(14))
p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                    first_indent=Cm(0), space_after=Pt(10))
_run(p, "Integrating Coloured Petri Nets with Multi-Modal Deep Learning for Interpretable Prediction in Precision Crop Management",
     bold=True, size=Pt(11))
p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                    first_indent=Cm(0), space_after=Pt(4))
_run(p, "Chifeng Shao 1,2,*, Qianqian Wang 2", size=Pt(10))
p = _make_paragraph(alignment=WD_ALIGN_PARAGRAPH.CENTER,
                    first_indent=Cm(0), space_after=Pt(12))
_run(p, "1 School of Mathematics and Big Data, Anhui University of Science and Technology, Huainan 232001, China;  2 College of Information and Network Engineering, Anhui Science and Technology University, Fengyang 233100, China",
     size=Pt(9))

body("This document collects the supplementary figures cited in the main manuscript. Figure numbers correspond to those used in the main text (Figures S1–S23).")

heading("S1. Supplementary Figures Referenced in the Main Text", level=2)
body("Figures S1 and S2 are cited in Section 5.3 (LSTM temperature forecasting); Figures S3 and S4 are cited in Section 5.5 (low-frequency frost prediction).")

insert_image("S1_loss_curve.png", caption_text="Figure S1. Training and validation MSE loss. Converged within 45 epochs via early stopping, no overfitting.")

insert_image("S2_all_variables.png", caption_text="Figure S2. Predicted versus actual values for all six meteorological variables on the test set. Only temperature (top left) shows close agreement (R² = 0.949).")

insert_image("S3_frost_climatology.png", caption_text="Figure S3. Seasonal climatology of frost precursor conditions from 44 years of NASA POWER training data. P(T < 0°C) peaks at 42% in January, with simultaneous elevation of all three probabilities.")

insert_image("S4_frost_prediction.png", caption_text="Figure S4. Frost prediction performance via the hybrid LSTM–climatology approach. (a) Confusion matrix. (b) Summary metrics. (c) Frost event timeline for 2020–2024.")

heading("S2. LSTM Time-Series Analysis — Supplementary Figures", level=2)
body("This section provides additional diagnostic figures for the LSTM temperature forecasting experiment (Section 5.3). All figures are generated from the 44-year NASA POWER model's test set predictions: Figure S5 shows the residual distribution, Figure S6 the predicted-versus-actual scatter, Figure S7 the seasonal patterns captured by the model, Figure S8 the multi-step error growth with forecast horizon, Figure S9 the 95% prediction confidence intervals, Figure S10 the learning-rate schedule, Figure S11 the precipitation prediction analysis, Figure S12 the feature correlation matrix, Figure S13 the extreme-value analysis of prediction errors, Figure S14 the illustrative diurnal temperature patterns, and Figure S15 the monthly performance breakdown.")

insert_image("S5_residual_distribution.png", caption_text="Figure S5. Residual distribution of LSTM temperature predictions. The residuals follow an approximately normal distribution centred near zero, indicating unbiased predictions with no systematic over- or under-prediction.")

insert_image("S6_scatter_pred_actual.png", caption_text="Figure S6. Predicted versus actual temperature values. The close alignment along the diagonal confirms strong predictive accuracy (R² = 0.949). Slight scatter increases at temperature extremes (below −5°C and above 30°C).")

insert_image("S7_seasonal_patterns.png", caption_text="Figure S7. Seasonal temperature patterns captured by the LSTM. The model correctly reproduces the annual cycle, inter-seasonal transitions, and synoptic-scale variability across multiple years.")

insert_image("S8_multi_step_errors.png", caption_text="Figure S8. Multi-step forecast error as a function of forecast horizon. Day 1 MAE = 1.69°C, Day 3 MAE = 2.24°C, Day 7 MAE = 2.90°C. Error grows smoothly with horizon, indicating robust temporal learning.")

insert_image("S9_confidence_intervals.png", caption_text="Figure S9. Prediction confidence intervals (95% level) for the test set. Intervals widen during rapid temperature transitions (e.g., cold front passages) and narrow during stable periods.")

insert_image("S10_learning_rate.png", caption_text="Figure S10. Learning rate schedule with Adam optimiser. The initial learning rate of 0.001 is maintained during early training and decays as the model converges.")

insert_image("S11_precipitation_analysis.png", caption_text="Figure S11. Precipitation prediction analysis. Precipitation is inherently intermittent and localised, resulting in low predictability (R² = 0.123) that reflects fundamental atmospheric limits rather than LSTM shortcomings.")

insert_image("S12_feature_correlation.png", caption_text="Figure S12. Feature correlation matrix of the six NASA POWER meteorological variables. Temperature (T2M) shows strong correlation with surface pressure (PS) and moderate correlation with solar radiation (ALLSKY).")

insert_image("S13_extreme_value_analysis.png", caption_text="Figure S13. Extreme value analysis of temperature prediction errors. The model maintains accuracy during extreme temperature events, with no disproportionate error spikes at the tails of the distribution.")

insert_image("S14_diurnal_patterns.png", caption_text="Figure S14. Illustrative diurnal temperature patterns for typical summer and winter days. Because NASA POWER provides daily-mean values, the LSTM does not model the diurnal cycle; these curves are shown for climatological context only.")

insert_image("S15_monthly_performance.png", caption_text="Figure S15. Monthly breakdown of prediction performance. Seasonal performance is consistent across all months, with slightly higher MAE in winter months (December–February) due to greater temperature variability.")

heading("S3. YOLOv7 Detection Analysis — Supplementary Figures", level=2)
body("This section provides additional diagnostic figures for the YOLOv7 object detection experiment (Section 5.2). All figures are generated from the 50% subset training run, the two-class re-validation (10,849 validation images), and full test-set inference (10,885 images): Figure S16 shows the complete training curves, Figure S17 the precision–recall curve, Figure S18 the F1-confidence curve, Figure S19 the detected class distribution, Figure S20 the confidence score distribution, Figure S21 the detection count per image, Figure S22 the healthy-versus-diseased ratio across test images, Figure S23 the confusion matrix, and Figures S24–S26 the held-out test-set precision–recall, F1-confidence, and confusion-matrix curves.")

insert_image("S16_yolo_training_all.png", caption_text="Figure S16. Complete YOLOv7 training curves across all 150 epochs. Training and validation losses for box, objectness, and classification heads, plus all detection metrics (precision, recall, mAP@0.5, mAP@0.5:0.95).")

insert_image("S17_yolo_pr_curve.png", caption_text="Figure S17. Precision-Recall curve for the YOLOv7 model on the PlantVillage validation set. The curve maintains high precision (>0.90) across most recall values, with mAP@0.5 = 0.986.")

insert_image("S18_yolo_f1_curve.png", caption_text="Figure S18. F1-confidence curve showing the optimal confidence threshold. The model achieves F1 > 0.92 across a wide confidence range (0.2–0.8), indicating robust detection quality.")

insert_image("S19_yolo_class_dist.png", caption_text="Figure S19. Class distribution of detected objects in the test set (two-class annotation). The majority of detections are diseased leaves, followed by healthy leaves, reflecting the dataset composition.")

insert_image("S20_yolo_conf_hist.png", caption_text="Figure S20. Confidence score distribution of YOLOv7 detections. Most detections have high confidence scores (>0.6), confirming the model's prediction reliability.")

insert_image("S21_yolo_det_count.png", caption_text="Figure S21. Distribution of detection count per image. Most images contain a single leaf detection, consistent with the PlantVillage dataset's close-up leaf photography.")

insert_image("S22_yolo_health_ratio.png", caption_text="Figure S22. Healthy vs. diseased leaf ratio across test images. The YOLOv7 model's detection statistics enable disease severity estimation (d_disease = n_diseased / n_leaves) for the CPN simulation.")

insert_image("S23_yolo_confusion_matrix.png", caption_text="Figure S23. Confusion matrix of the YOLOv7 model on the PlantVillage validation set (10,849 images, two-class annotation). The matrix shows near-diagonal predictions, with residual confusion between healthy and diseased leaves; the background false-positive row and false-negative column reflect missed and spurious detections at the chosen confidence threshold.")

insert_image("S24_test_pr_curve.png", caption_text="Figure S24. Precision-recall curve of the YOLOv7 model on the held-out PlantVillage test set (10,885 images), with mAP@0.5 = 0.985 (healthy_leaf 0.974, diseased_leaf 0.996).")

insert_image("S25_test_f1_curve.png", caption_text="Figure S25. F1-confidence curve of the YOLOv7 model on the held-out PlantVillage test set (10,885 images).")

insert_image("S26_test_confusion_matrix.png", caption_text="Figure S26. Confusion matrix of the YOLOv7 model on the held-out PlantVillage test set (10,885 images; 3,025 healthy_leaf and 7,860 diseased_leaf ground-truth instances). The background row and column reflect missed and spurious detections at the evaluation confidence threshold.")

supp.save(SUPP_PATH)
supp_paras = len(supp.paragraphs)
print(f"Supplementary document saved: {SUPP_PATH}")
print(f"  Paragraphs: {supp_paras}")
print(f"  File size: {os.path.getsize(SUPP_PATH) / 1024:.0f} KB")

set_doc(main_doc)

# SAVE MAIN MANUSCRIPT
# ═══════════════════════════════════════════════════════════════════════════
doc.save(OUT_PATH)

# Verification
paras = len(doc.paragraphs)
tbls = len(doc.tables)
print(f"Document saved: {OUT_PATH}")
print(f"  Paragraphs: {paras}")
print(f"  Tables: {tbls}")
print(f"  File size: {os.path.getsize(OUT_PATH) / 1024:.0f} KB")

# Check key content
checks = {
    "Figure 1": 0, "Figure 6": 0, "Figure 7": 0, "Figure 9": 0,
    "Figure 10": 0, "Figure 11": 0, "Figure 12": 0, "Figure 13": 0,
    "Figure 14": 0, "Figure 15": 0,
    "Table 1": 0, "Table 2": 0, "Table 3": 0, "Table 4": 0, "Table 5": 0,
    "Table 6": 0, "Table 7": 0, "Table 8": 0, "Table 9": 0, "Table 10": 0,
    "Table A1": 0, "Table A2": 0,
    "Algorithm 1": 0, "Algorithm 2": 0, "Algorithm 3": 0,
    "Definition 1": 0, "Definition 8": 0, "Definition 9": 0,
    "Abbreviations": 0, "OCPPP": 0, "CPN": 0, "YOLOv7": 0, "LSTM": 0,
    "GDD": 0, "ET₀": 0, "mAP": 0, "MAE": 0, "RMSE": 0,
    "0.949": 0, "0.986": 0, "0.965": 0, "0.884": 0, "0.994": 0, "0.982": 0,
    "YOLOv8s": 0, "10,885": 0, "10.91": 0, "9.11": 0, "classical optimisation": 0,
    "Figure S24": 0, "Figure S25": 0, "Figure S26": 0,
    "m = 9": 0, "k ≤ 6–8": 0, "7.5 mm irrigation": 0, "T2M < 2 °C proxy": 0,
    "scenario-level instantiations": 0, "four experiments and an end-to-end": 0,
    "Scheme A": 0, "Scheme B": 0, "Scheme C": 0, "Scheme D": 0,
    "Appendix A": 0, "Appendix B": 0,
    "Supplementary Materials": 0, "winter wheat": 0, "5.6 End-to-End": 0, "5.7 Summary": 0,
}
for p in doc.paragraphs:
    for k in checks:
        if k.lower() in p.text.lower():
            checks[k] += 1
print("\nContent verification (key terms):")
for k, v in checks.items():
    icon = "✅" if v > 0 else "❌"
    print(f"  {icon} {k}: found {v}")
