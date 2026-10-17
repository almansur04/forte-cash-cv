"""ForteBank Vision: pilot study under production constraints.

12 slides for a research/engineering audience. Statistical honesty over product claims.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ============ COLORS ============
MAGENTA = RGBColor(0xA3, 0x15, 0x51)
MAGENTA_SOFT = RGBColor(0xFD, 0xF2, 0xF8)
BLACK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xF3, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x00, 0x57, 0x3F)
GREEN_SOFT = RGBColor(0xEC, 0xFD, 0xF5)
RED = RGBColor(0xDC, 0x26, 0x26)
RED_SOFT = RGBColor(0xFE, 0xF2, 0xF2)
AMBER = RGBColor(0xD9, 0x77, 0x06)
AMBER_SOFT = RGBColor(0xFE, 0xF3, 0xC7)

HM_MAG = "#A31551"
HM_GREEN = "#00573F"
HM_RED = "#DC2626"
HM_GRAY = "#666666"
HM_AMBER = "#D97706"

CHARTS = Path("presentation/charts")
CHARTS.mkdir(parents=True, exist_ok=True)


def setup_matplotlib():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


# ============ CHARTS ============
def chart_stat_ci():
    fig, ax = plt.subplots(figsize=(14, 3.0))

    configs = ["Trivial baseline\n(all 'нарушение')",
               "Our model\n(measured)",
               "SOTA\n(100k labeled videos)"]
    points = [52.4, 61.22, 95.0]
    lo_err = [0, 16.22, 0]
    hi_err = [0, 14.78, 0]
    colors = ["#94A3B8", HM_MAG, "#CBD5E1"]
    x_pos = [1, 2, 3]

    for i, (x, y, c) in enumerate(zip(x_pos, points, colors)):
        if lo_err[i] > 0 or hi_err[i] > 0:
            ax.errorbar(x, y, yerr=[[lo_err[i]], [hi_err[i]]],
                        fmt="o", color=c, markersize=18, capsize=14,
                        capthick=3, linewidth=3, zorder=3)
        else:
            ax.plot(x, y, "o", color=c, markersize=18, zorder=3)
        label_y = y + (hi_err[i] + 6 if hi_err[i] > 0 else 6)
        ax.text(x, label_y, f"{y:.1f}%", ha="center",
                fontsize=13, fontweight="bold", color=c)

    ax.axhspan(45, 76, xmin=0.44, xmax=0.56, alpha=0.12, color=HM_MAG)
    ax.text(2.55, 60, "95% CI\n45–76%", fontsize=10, color=HM_MAG,
            fontweight="bold", ha="left", va="center")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("F1 score (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(CHARTS / "stat_ci.png", dpi=200, bbox_inches="tight")
    plt.close()


def chart_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.0))

    configs = ["V9 baseline\n416 PyTorch MPS",
               "ONNX export\n416 FP32",
               "INT8 quant\n416",
               "Input 320\nINT8",
               "+ cv2.grab\n320 INT8",
               "+ early-stop\n320 INT8"]
    f1 = [61.22, 61.22, 60.1, 58.7, 58.7, 58.7]
    latency = [2.0, 2.5, 1.0, 0.9, 0.9, 0.5]
    measured = [True, True, False, False, False, False]
    colors1 = [HM_GREEN if m else HM_AMBER for m in measured]

    bars1 = ax1.bar(range(len(configs)), f1, color=colors1, edgecolor="white", linewidth=2)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, fontsize=7.5, rotation=20, ha="right")
    ax1.set_ylabel("F1 (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(50, 70)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.set_title("F1 across pipeline configurations", fontsize=10,
                  fontweight="bold", color=HM_MAG, pad=6)
    for bar, val, m in zip(bars1, f1, measured):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                 f"{val:.1f}" + ("" if m else "*"),
                 ha="center", fontsize=8.5, fontweight="bold")

    bars2 = ax2.bar(range(len(configs)), latency, color=colors1, edgecolor="white", linewidth=2)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, fontsize=7.5, rotation=20, ha="right")
    ax2.set_ylabel("Latency (s)", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.set_title("Latency on 25s video", fontsize=10,
                  fontweight="bold", color=HM_MAG, pad=6)
    for bar, val, m in zip(bars2, latency, measured):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f"{val:.1f}s" + ("" if m else "*"),
                 ha="center", fontsize=8.5, fontweight="bold")

    fig.text(0.5, -0.02, "Green = measured on 42-clip benchmark.  Amber (*) = estimated, not re-run.",
             ha="center", fontsize=8.5, color=HM_GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(CHARTS / "ablation.png", dpi=200, bbox_inches="tight")
    plt.close()


def chart_failure_taxonomy():
    fig, ax = plt.subplots(figsize=(14, 2.8))

    categories = [
        ("FP: cashier leans, torso exits rect", 8, HM_RED),
        ("FP: YOLO detection dropout > 1.2s", 4, HM_RED),
        ("FN: sitting violation (phone, reaching)", 6, HM_AMBER),
        ("FN: violation shorter than 1.2s", 1, HM_AMBER),
    ]
    labels = [c[0] for c in categories]
    counts = [c[1] for c in categories]
    colors = [c[2] for c in categories]

    bars = ax.barh(labels, counts, color=colors, edgecolor="white", linewidth=2)
    ax.set_xlabel("Error count (19 total on 42 clips)", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)

    for bar, val in zip(bars, counts):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, str(val),
                va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(CHARTS / "failure_taxonomy.png", dpi=200, bbox_inches="tight")
    plt.close()


def chart_generalization():
    """Two camera angles. NO aspect='equal' -- use plain xlim/ylim."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.2))

    for ax, status in [(ax1, "training"), (ax2, "test")]:
        # Frame
        ax.add_patch(plt.Rectangle((0, 0), 4.64, 8.48,
                                    facecolor="#F1F5F9", edgecolor="#94A3B8", lw=1.5))
        # Desk zone
        dx1, dy1 = 0.15 * 4.64, 0.40 * 8.48
        dx2, dy2 = 0.85 * 4.64, 0.95 * 8.48
        ax.add_patch(plt.Rectangle((dx1, dy1), dx2-dx1, dy2-dy1,
                                    facecolor="none", edgecolor="#10B981", lw=3, ls="--"))
        ax.text(dx1+0.15, dy1+0.35, "DESK_RECT", color="#10B981",
                fontsize=9, fontweight="bold")

        if status == "training":
            ax.plot(2.3, 5.5, "o", color="#10B981", markersize=22)
            ax.text(2.3, 5.5, "cashier", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=8)
            ax.set_title("TRAINING ANGLE  (top-down, cashier inside)",
                         fontsize=10, fontweight="bold", color=HM_GREEN, pad=8)
        else:
            ax.plot(4.2, 1.5, "o", color=HM_RED, markersize=22)
            ax.text(4.2, 1.5, "cashier", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=8)
            ax.annotate("", xy=(4.2, 1.5), xytext=(2.3, 5.5),
                        arrowprops=dict(arrowstyle="->", color=HM_RED, lw=2, ls="--"))
            ax.text(2.8, 3.3, "position\nshifted", fontsize=8, color=HM_RED,
                    style="italic", ha="center")
            ax.set_title("TEST ANGLE  (per brief, cashier shifted)",
                         fontsize=10, fontweight="bold", color=HM_RED, pad=8)

        ax.set_xlim(-0.2, 4.84)
        ax.set_ylim(8.68, -0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(CHARTS / "generalization.png", dpi=200, bbox_inches="tight")
    plt.close()


def chart_iteration():
    fig, ax = plt.subplots(figsize=(14, 2.8))

    versions = ["V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18"]
    f1 = [61.22, 58.33, 66.67, 60.38, 8.70, 60.38, 61.90, 0.0, 0.0, 0.0]
    colors = []
    for v in f1:
        if v == 61.22:
            colors.append(HM_MAG)
        elif v == 0:
            colors.append(HM_RED)
        else:
            colors.append("#94A3B8")

    bars = ax.bar(versions, f1, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("F1 score (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 78)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    for bar, val in zip(bars, f1):
        # Annotate every bar, including zeros (so 0% is visible)
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height() + 1.5, 2),
                f"{val:.1f}", ha="center", fontsize=9, fontweight="bold",
                color=HM_RED if val == 0 else "black")

    ax.axhline(61.22, color=HM_MAG, linestyle=":", linewidth=1.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(CHARTS / "iteration.png", dpi=200, bbox_inches="tight")
    plt.close()


def generate_all_charts():
    setup_matplotlib()
    print("Generating charts...")
    chart_stat_ci(); print("  stat_ci.png")
    chart_ablation(); print("  ablation.png")
    chart_failure_taxonomy(); print("  failure_taxonomy.png")
    chart_generalization(); print("  generalization.png")
    chart_iteration(); print("  iteration.png")
    print("Charts done.\n")


# ============ DECK HELPERS ============
def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_top_bar(slide, color=MAGENTA):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(0.1))
    bar.fill.solid(); bar.fill.fore_color.rgb = color
    bar.line.fill.background()


def add_slide_number(slide, num, total=12):
    box = slide.shapes.add_textbox(Inches(9.0), Inches(5.28), Inches(0.85), Inches(0.25))
    p = box.text_frame.paragraphs[0]
    p.text = f"{num} / {total}"
    p.alignment = PP_ALIGN.RIGHT
    p.font.size = Pt(9); p.font.color.rgb = GRAY; p.font.name = "Arial"


def add_title(slide, text, num=None):
    add_top_bar(slide)
    box = slide.shapes.add_textbox(Inches(0.4), Inches(0.3), Inches(9.2), Inches(0.65))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(22); p.font.bold = True
    p.font.color.rgb = MAGENTA; p.font.name = "Arial"
    if num is not None:
        add_slide_number(slide, num)


def add_subtitle(slide, text, top=0.92):
    box = slide.shapes.add_textbox(Inches(0.4), Inches(top), Inches(9.2), Inches(0.32))
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(11); p.font.color.rgb = GRAY
    p.font.italic = True; p.font.name = "Arial"


def add_section(slide, title, bullets, top, left=0.4, width=9.2, accent=MAGENTA, size=10, bullet_gap=3):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.3))
    p = box.text_frame.paragraphs[0]
    p.text = title.upper()
    p.font.size = Pt(10); p.font.bold = True
    p.font.color.rgb = accent; p.font.name = "Arial"
    pPr = p._p.get_or_add_pPr(); pPr.set("spc", "150")

    box = slide.shapes.add_textbox(Inches(left), Inches(top + 0.32), Inches(width), Inches(1.5))
    tf = box.text_frame; tf.word_wrap = True
    first = True
    for b in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = "•  " + b
        p.font.size = Pt(size); p.font.color.rgb = BLACK; p.font.name = "Arial"
        p.space_after = Pt(bullet_gap)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", "180000"); pPr.set("indent", "-180000")


def add_image(slide, path, left, top, width=None, height=None):
    if Path(path).exists():
        kw = {}
        if width: kw["width"] = Inches(width)
        if height: kw["height"] = Inches(height)
        slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kw)


def card(slide, left, top, width, height, bg_color=LIGHT_GRAY, border_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid(); shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.color.rgb = bg_color
    shape.adjustments[0] = 0.05
    return shape


def add_table(slide, headers, rows, left, top, width, height, col_widths=None):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols,
                                         Inches(left), Inches(top),
                                         Inches(width), Inches(height))
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(10); p.font.bold = True
        p.font.color.rgb = WHITE; p.font.name = "Arial"
        cell.fill.solid(); cell.fill.fore_color.rgb = MAGENTA
        cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.05); cell.margin_bottom = Inches(0.05)
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9); p.font.color.rgb = BLACK; p.font.name = "Arial"
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r % 2 == 0 else LIGHT_GRAY
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04); cell.margin_bottom = Inches(0.04)


# ============ BUILD DECK ============
def build():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.63)

    # ===== SLIDE 1: TITLE =====
    slide = new_slide(prs)
    bg = slide.background; bg.fill.solid(); bg.fill.fore_color.rgb = MAGENTA
    box = slide.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(8.6), Inches(1.0))
    p = box.text_frame.paragraphs[0]
    p.text = "ForteBank Vision"
    p.font.size = Pt(52); p.font.bold = True
    p.font.color.rgb = WHITE; p.font.name = "Arial"
    box = slide.shapes.add_textbox(Inches(0.7), Inches(2.4), Inches(8.6), Inches(1.6))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "What is achievable in 48 hours with 42 labeled examples, no GPU, no cloud,"
    p.font.size = Pt(15); p.font.color.rgb = WHITE; p.font.name = "Arial"
    p = tf.add_paragraph()
    p.text = "and a legal constraint that the model must be explainable."
    p.font.size = Pt(15); p.font.color.rgb = WHITE; p.font.name = "Arial"
    badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(0.7), Inches(4.4), Inches(3.6), Inches(0.4))
    badge.fill.solid(); badge.fill.fore_color.rgb = WHITE
    badge.line.fill.background()
    p = slide.shapes.add_textbox(Inches(0.7), Inches(4.42), Inches(3.6), Inches(0.38)).text_frame.paragraphs[0]
    p.text = "Nazarbayev University Datathon 2026"
    p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = MAGENTA
    p.font.name = "Arial"; p.alignment = PP_ALIGN.CENTER
    box = slide.shapes.add_textbox(Inches(0.7), Inches(5.05), Inches(8.6), Inches(0.35))
    p = box.text_frame.paragraphs[0]
    p.text = "almansur.kakimov@nu.edu.kz  |  github.com/almansur04/forte-cash-cv"
    p.font.size = Pt(10); p.font.color.rgb = WHITE; p.font.name = "Arial"

    # ===== SLIDE 2: CONSTRAINTS =====
    slide = new_slide(prs)
    add_title(slide, "The constraints define the problem", num=2)
    add_subtitle(slide, "This is not a paper. It is a feasibility study under four simultaneous constraints.")
    quadrants = [
        (0.4, 1.35, "DATA", [
            "42 labeled clips (22 positive, 20 negative)",
            "10 to 73 seconds per clip",
            "No bounding boxes, one timestamp per violation",
        ]),
        (5.1, 1.35, "COMPUTE", [
            "48 hours, one M1 laptop",
            "No GPU at inference (branch server target)",
            "Deployment target: $5/month VPS",
        ]),
        (0.4, 3.3, "LEGAL", [
            "No cloud inference (data residency)",
            "Every output cites an RK normative clause",
            "No hallucinated citations",
        ]),
        (5.1, 3.3, "EVALUATION", [
            "Test set filmed from a different camera angle",
            "No published baseline for this task",
            "F1 is the target metric",
        ]),
    ]
    for left, top, title, bullets in quadrants:
        card(slide, left, top, 4.5, 1.8, LIGHT_GRAY)
        add_section(slide, title, bullets, top=top + 0.15, left=left + 0.2, width=4.1, accent=MAGENTA, size=9)

    # ===== SLIDE 3: STATISTICAL VALIDITY =====
    slide = new_slide(prs)
    add_title(slide, "Statistical validity: n=42 spans 30 F1 points", num=3)
    add_subtitle(slide, "Every number on subsequent slides inherits this uncertainty.")
    add_image(slide, CHARTS / "stat_ci.png", 0.4, 1.35, width=9.2)
    add_section(slide, "What this means", [
        "95% CI on F1 is 45–76%. The lower bound overlaps the trivial baseline.",
        "Required sample size for ±5% CI at 95% confidence is ~400 clips. We had 42.",
        "Treat every number on subsequent slides as an existence proof, not a performance claim.",
    ], top=3.95, width=9.2, accent=MAGENTA, size=10)

    # ===== SLIDE 4: WHAT WE SHIPPED =====
    slide = new_slide(prs)
    add_title(slide, "What we shipped: the verifiable subset", num=4)
    add_subtitle(slide, "One rule. Deterministic. Cited. Runs on a branch server, offline.")
    add_section(slide, "The rule", [
        "Detects: cashier leaves workstation zone for more than 1.2s",
        "Cites: NBRK Resolution 231, Article 22, verified on adilet.zan.kz",
        "Output: timestamp, evidence JPEG, article number, direct URL",
    ], top=1.4, width=5.6, size=10)
    add_section(slide, "The system", [
        "1.0s per 25s video on M1 CPU, 179 MB peak RAM",
        "Docker image, fully offline after pull",
        "Live demo, working API, deterministic output",
    ], top=2.95, width=5.6, accent=GREEN, size=10)
    # Fallback if screenshot missing: draw a placeholder card on the right
    if Path("presentation/demo_screenshot.png").exists():
        add_image(slide, "presentation/demo_screenshot.png", 5.9, 1.4, width=3.9)
    else:
        card(slide, 5.9, 1.4, 3.9, 3.8, MAGENTA_SOFT, border_color=MAGENTA)
        box = slide.shapes.add_textbox(Inches(6.0), Inches(2.9), Inches(3.7), Inches(0.8))
        p = box.text_frame.paragraphs[0]
        p.text = "Dashboard screenshot\n(see live demo)"
        p.font.size = Pt(12); p.font.italic = True
        p.font.color.rgb = MAGENTA; p.font.name = "Arial"
        p.alignment = PP_ALIGN.CENTER

    # ===== SLIDE 5: ABLATION =====
    slide = new_slide(prs)
    add_title(slide, "Ablation: what we measured vs. what we estimated", num=5)
    add_subtitle(slide, "The shipped configuration has NOT been re-benchmarked on all 42 clips.")
    add_image(slide, CHARTS / "ablation.png", 0.3, 1.3, width=9.4)
    add_section(slide, "Honest reading", [
        "Full F1 benchmark was run on V9 PyTorch MPS only. Post-ONNX numbers are estimated.",
        "The shipped INT8 320x320 model was spot-checked on X-11 (correct) and X-48 (correct).",
        "Re-running the benchmark on the shipped model is the first task for Phase 2.",
    ], top=4.05, width=9.2, accent=AMBER, size=10)

    # ===== SLIDE 6: FAILURE TAXONOMY =====
    slide = new_slide(prs)
    add_title(slide, "Failure taxonomy: 19 errors, 4 root causes", num=6)
    add_subtitle(slide, "Not 'the model is bad'. Specific failure modes with specific fixes.")
    add_image(slide, CHARTS / "failure_taxonomy.png", 0.3, 1.3, width=9.4)
    add_section(slide, "What is fixable vs. what is not", [
        "8 FPs from lean: tighten the vertical rect boundary. One line. May cost a few TPs.",
        "4 FPs from detection dropout: require 2 consecutive frames of no-detection before firing.",
        "6 FNs from sitting violations: needs an action classifier. Not fixable in 48h.",
        "1 FN from short violation: lower streak threshold from 6 to 4. Risk: more FPs.",
    ], top=3.85, width=9.2, accent=MAGENTA, size=9)

    # ===== SLIDE 7: GENERALIZATION (FIXED) =====
    slide = new_slide(prs)
    add_title(slide, "Generalization: the biggest honest limitation", num=7)
    add_subtitle(slide, "The dataset brief says test is filmed from a different angle. We did not solve this.")
    add_image(slide, CHARTS / "generalization.png", 0.3, 1.3, width=9.4)
    add_section(slide, "Two viable paths, both out of 48h scope", [
        "Approach A: per-video desk detection using YOLO's 'dining table' class + homography",
        "Approach B: manual desk-corner annotation at inference time (UI prompt before analysis)",
        "Both require 1–2 days of implementation and validation. Deferred to Phase 2.",
    ], top=4.0, width=9.2, accent=RED, size=9)

    # ===== SLIDE 8: WHAT WE REJECTED =====
    slide = new_slide(prs)
    add_title(slide, "What we rejected: with proper caveats", num=8)
    add_subtitle(slide, "Two single tests. We do not claim these generalize to the categories.")
    card(slide, 0.4, 1.4, 4.5, 2.95, RED_SOFT, border_color=RED)
    add_section(slide, "Moondream2 (VLM)", [
        "Prompt: 'Is there a mobile phone visible?' (single-sentence, no chain-of-thought)",
        "n=4 clean videos. All 4 produced false positives.",
        "Root cause: IP desk phone on every desk. VLM answered 'yes'.",
        "n=2 real violations. Both answered 'no'. Phone too small.",
    ], top=1.5, left=0.6, width=4.1, accent=RED, size=9)
    card(slide, 5.1, 1.4, 4.5, 2.95, RED_SOFT, border_color=RED)
    add_section(slide, "YOLOv8s-Pose", [
        "Head/wrist keypoints as proxy for presence and phone use",
        "Failure 1: keypoints drop for 1-3 frames on lean-forward (12/12 FPs)",
        "Failure 2: chin-rest and phone-call are geometrically identical in 2D",
        "No threshold separates them at this resolution",
    ], top=1.5, left=5.3, width=4.1, accent=RED, size=9)
    card(slide, 0.4, 4.5, 9.2, 1.0, AMBER_SOFT, border_color=AMBER)
    add_section(slide, "Caveat", [
        "Each test was a single experiment. n=4 and n=12 respectively. Single prompt, single model.",
        "Not tried: alternative prompts, chain-of-thought, or other VLMs. That is the study a real researcher would run.",
        "We do not claim 'VLMs fail'. We claim 'this VLM with this prompt failed'.",
    ], top=4.58, left=0.6, width=8.8, accent=AMBER, size=9)

    # ===== SLIDE 9: ITERATION STORY =====
    slide = new_slide(prs)
    add_title(slide, "The iteration story: measurement as ground truth", num=9)
    add_subtitle(slide, "9 versions. 3 produced 0% F1. Each looked correct before measurement.")
    add_image(slide, CHARTS / "iteration.png", 0.3, 1.3, width=9.4)
    add_section(slide, "What broke and why", [
        "V13 (central column): box expanded to 70% of frame → everything counted as 'at desk' → 8.7% F1",
        "V16 (adaptive home base): cashier never exceeded threshold → streak never fired → 0% F1",
        "V17 (no-null-detection guard): guard prevented streak from building → 0% F1",
        "V18 (y2=1.0): box covered bottom of frame, same failure as V17 → 0% F1",
    ], top=3.85, width=9.2, accent=MAGENTA, size=9)

    # ===== SLIDE 10: RELATED WORK =====
    slide = new_slide(prs)
    add_title(slide, "Related work and the honest gap", num=10)
    add_subtitle(slide, "We do not beat SOTA. We characterize the achievable floor under constraint.")
    rows = [
        ("AVA, MOMA, SlowFast", "100k+ labeled videos, 8-128 GPUs",
         "Not applicable: 42 videos, 1 M1"),
        ("Retail fraud (Chen et al. 2019)", "10k+ labeled transactions",
         "Not public, not obtainable in 48h"),
        ("Pose-based action recognition", "Requires stable keypoints",
         "Fails at this resolution (slide 8)"),
        ("VLM zero-shot (GPT-4V, Llama-Vision)", "Cloud APIs or 7 GB local models",
         "Prohibited by data residency"),
        ("This work", "42 clips, 48h, CPU-only, offline",
         "61.22% F1, verifiable, deployable"),
    ]
    add_table(slide,
              ["Approach", "Resource requirement", "Why not used here"],
              rows, left=0.4, top=1.4, width=9.2, height=2.75,
              col_widths=[2.5, 3.0, 3.7])
    add_section(slide, "Positioning", [
        "This is not SOTA. It is 'what can you ship today under the constraints above'.",
        "The gap to SOTA is order-of-magnitude on data and compute. Not closeable in 48h.",
        "The contribution is a feasibility bound and a legal-anchoring pattern, not a new model.",
    ], top=4.35, width=9.2, accent=MAGENTA, size=9)

    # ===== SLIDE 11: THE ASK =====
    slide = new_slide(prs)
    add_title(slide, "The ask: statistical power, not features", num=11)
    add_subtitle(slide, "What would move this from pilot study to validated claim.")
    rows = [
        ("WEEKS 1 TO 6", "Label 400 clips (from 42). Two annotators.",
         "Moves CI from ±15% to ±5%"),
        ("WEEKS 7 TO 10", "Fine-tune 3D action classifier for sitting violations",
         "Target: +5 F1 on phone/reaching"),
        ("WEEKS 11 TO 12", "Per-video desk detection for angle invariance",
         "Target: generalizes to test angle"),
        ("WEEKS 13 TO 16", "One pilot branch, 30 days live footage",
         "Success: 80% catch, <10% FP"),
    ]
    y = 1.35
    for label, task, outcome in rows:
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                       Inches(0.4), Inches(y), Inches(1.9), Inches(0.58))
        badge.fill.solid(); badge.fill.fore_color.rgb = MAGENTA
        badge.line.fill.background(); badge.adjustments[0] = 0.15
        p = slide.shapes.add_textbox(Inches(0.4), Inches(y + 0.14), Inches(1.9), Inches(0.4)).text_frame.paragraphs[0]
        p.text = label
        p.font.size = Pt(9); p.font.bold = True; p.font.color.rgb = WHITE
        p.font.name = "Arial"; p.alignment = PP_ALIGN.CENTER
        p = slide.shapes.add_textbox(Inches(2.5), Inches(y + 0.06), Inches(4.8), Inches(0.55)).text_frame.paragraphs[0]
        p.text = task
        p.font.size = Pt(10); p.font.color.rgb = BLACK; p.font.name = "Arial"
        p = slide.shapes.add_textbox(Inches(7.4), Inches(y + 0.06), Inches(2.3), Inches(0.55)).text_frame.paragraphs[0]
        p.text = outcome
        p.font.size = Pt(9); p.font.color.rgb = GREEN; p.font.name = "Arial"; p.font.italic = True
        y += 0.72
    card(slide, 0.4, 4.45, 9.2, 0.85, GREEN_SOFT, border_color=GREEN)
    add_section(slide, "What we need from you", [
        "400 historical clips for labeling  ·  one pilot branch contact  ·  nothing else",
    ], top=4.53, left=0.6, width=8.8, accent=GREEN, size=10)

    # ===== SLIDE 12: CLOSING =====
    slide = new_slide(prs)
    bg = slide.background; bg.fill.solid(); bg.fill.fore_color.rgb = MAGENTA
    box = slide.shapes.add_textbox(Inches(0.7), Inches(1.15), Inches(8.6), Inches(0.9))
    p = box.text_frame.paragraphs[0]
    p.text = "This is a pilot study."
    p.font.size = Pt(44); p.font.bold = True
    p.font.color.rgb = WHITE; p.font.name = "Arial"
    box = slide.shapes.add_textbox(Inches(0.7), Inches(2.15), Inches(8.6), Inches(1.4))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "We proved: deterministic, legally-anchored, CPU-deployable CV is achievable in 48h."
    p.font.size = Pt(14); p.font.color.rgb = WHITE; p.font.name = "Arial"
    p = tf.add_paragraph()
    p.text = "We did not prove: generalization, statistical significance, or action recognition."
    p.font.size = Pt(14); p.font.color.rgb = WHITE; p.font.name = "Arial"
    p = tf.add_paragraph()
    p.text = "The gap is measurable, not hidden."
    p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = WHITE; p.font.name = "Arial"
    card_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                      Inches(0.7), Inches(4.0), Inches(8.6), Inches(1.15))
    card_box.fill.solid(); card_box.fill.fore_color.rgb = WHITE
    card_box.line.fill.background(); card_box.adjustments[0] = 0.1
    box = slide.shapes.add_textbox(Inches(0.9), Inches(4.1), Inches(8.2), Inches(0.95))
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "github.com/almansur04/forte-cash-cv"
    p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = MAGENTA; p.font.name = "Arial"
    p = tf.add_paragraph()
    p.text = "docker run -p 8000:8000 almansur04/forte-cash-api:latest"
    p.font.size = Pt(10); p.font.color.rgb = BLACK; p.font.name = "Courier New"
    p = tf.add_paragraph()
    p.text = "API key: forte-hack-2026"
    p.font.size = Pt(10); p.font.color.rgb = BLACK; p.font.name = "Arial"

    out = Path("presentation/ForteBank_Vision.pptx")
    prs.save(out)
    print(f"\nSaved: {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    generate_all_charts()
    build()