from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.paths import PROCESSED_DIR, REPORTS_DIR  # noqa: E402


PAPER_DIR = REPORTS_DIR / "paper_q4_prep"
FIG_DIR = PAPER_DIR / "ieee_figures"
YOLO_MULTI_DIR = REPORTS_DIR / "yolo_multiseed" / "yolo_multiseed_strengthening"

DPI = 600
SINGLE_COL = (3.5, 2.45)
DOUBLE_COL = (7.16, 3.9)
WIDE = (7.16, 4.7)

INK = "#1f2933"
MUTED = "#64748b"
GRID = "#d9e2ec"
BG = "#ffffff"
ACCENT = "#007c89"
ACCENT_2 = "#b26a00"
ACCENT_3 = "#6b7280"
GOOD = "#198754"
WARN = "#b26a00"
BAD = "#b3261e"


METHOD_LABELS = {
    "extra_trees_count_regression": "Extra Trees",
    "random_forest_count_regression": "Random Forest",
    "hist_gradient_boosting_count_regression": "HistGradientBoosting",
    "knn_count_regression": "KNN",
    "watershed_counting": "Watershed",
    "otsu_connected_components": "Otsu CC",
    "distance_peak_counting": "Distance peaks",
    "adaptive_connected_components": "Adaptive CC",
    "blob_detector_counting": "Blob detector",
    "yolo_v8n_smoke": "YOLOv8n smoke",
    "yolov8n_gpu_20ep": "YOLOv8n",
    "yolo11n_gpu_20ep": "YOLO11n",
    "yolo11s_gpu_20ep": "YOLO11s",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def ieee_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str, pdf: bool = True, png: bool = True) -> list[Path]:
    ensure_dir()
    outputs: list[Path] = []
    if pdf:
        path = FIG_DIR / f"{stem}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        outputs.append(path)
    if png:
        path = FIG_DIR / f"{stem}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        outputs.append(path)
    plt.close(fig)
    return outputs


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.2f}", xpad: float = 0.02) -> None:
    xmax = ax.get_xlim()[1]
    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(width + xmax * xpad, y, fmt.format(width), va="center", ha="left", fontsize=6.5)


def wrap_text(text: str, width: int = 20) -> str:
    return "\n".join(wrap(text, width=width))


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    facecolor: str = "#f8fafc",
    edgecolor: str = ACCENT,
    text_size: float = 7.2,
    radius: float = 0.025,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=text_size)
    return box


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(PROCESSED_DIR / "detection_counting_manifest.csv")
    manifest["object_count"] = pd.to_numeric(manifest["object_count"], errors="coerce")
    manifest["split"] = pd.Categorical(manifest["split"], categories=["train", "val", "test"], ordered=True)
    return manifest


def build_methodology_flowchart() -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.04, 0.74), (0.18, 0.13), "Raw BSF larvae\nimages + YOLO labels", "#eef8fa")
    draw_box(ax, (0.29, 0.74), (0.18, 0.13), "Dataset curation\nand label checks", "#f8fafc")
    draw_box(ax, (0.54, 0.74), (0.18, 0.13), "Train / val /\ntest split", "#f8fafc")
    draw_box(ax, (0.78, 0.74), (0.18, 0.13), "Benchmark-ready\nmanifest", "#eef8fa")

    draw_arrow(ax, (0.22, 0.805), (0.29, 0.805))
    draw_arrow(ax, (0.47, 0.805), (0.54, 0.805))
    draw_arrow(ax, (0.72, 0.805), (0.78, 0.805))

    draw_box(ax, (0.07, 0.44), (0.25, 0.16), "Classical CV\nthresholding, blobs,\nwatershed", "#fff7ed", WARN)
    draw_box(ax, (0.375, 0.44), (0.25, 0.16), "Feature regressors\nExtra Trees, RF,\nHGB, KNN", "#f3f4f6", ACCENT_3)
    draw_box(ax, (0.68, 0.44), (0.25, 0.16), "YOLO detectors\nYOLOv8n, YOLO11n,\nYOLO11s", "#eef8fa", ACCENT)

    draw_arrow(ax, (0.87, 0.74), (0.80, 0.60))
    draw_arrow(ax, (0.87, 0.74), (0.50, 0.60))
    draw_arrow(ax, (0.87, 0.74), (0.19, 0.60))

    draw_box(ax, (0.16, 0.18), (0.20, 0.12), "Predicted\nlarvae count", "#f8fafc")
    draw_box(ax, (0.41, 0.18), (0.20, 0.12), "Ground-truth\nannotation count", "#f8fafc")
    draw_box(ax, (0.66, 0.18), (0.20, 0.12), "Metrics\nMAE, RMSE, R2,\nmAP", "#eef8fa")

    draw_arrow(ax, (0.20, 0.44), (0.25, 0.30))
    draw_arrow(ax, (0.50, 0.44), (0.27, 0.30))
    draw_arrow(ax, (0.80, 0.44), (0.28, 0.30))
    draw_arrow(ax, (0.36, 0.24), (0.41, 0.24))
    draw_arrow(ax, (0.61, 0.24), (0.66, 0.24))

    ax.text(0.04, 0.94, "Overall experimental workflow", fontsize=10, fontweight="bold")
    ax.text(
        0.04,
        0.905,
        "The study evaluates direct image-processing baselines, feature-based regressors, and detector-based counting under the same test split.",
        fontsize=7.5,
        color=MUTED,
    )
    save_figure(fig, "fig01_methodology_flowchart")


def build_detector_counting_pipeline() -> None:
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ((0.04, 0.58), (0.18, 0.17), "Input image"),
        ((0.29, 0.58), (0.18, 0.17), "YOLO\ninference"),
        ((0.54, 0.58), (0.18, 0.17), "Confidence\nfiltering + NMS"),
        ((0.78, 0.58), (0.18, 0.17), "Detected boxes\n= predicted count"),
    ]
    for xy, wh, text in boxes:
        draw_box(ax, xy, wh, text, "#eef8fa", ACCENT)
    for start_x, end_x in [(0.22, 0.29), (0.47, 0.54), (0.72, 0.78)]:
        draw_arrow(ax, (start_x, 0.665), (end_x, 0.665))

    draw_box(ax, (0.04, 0.19), (0.22, 0.15), "Annotation file\nnumber of labels", "#f8fafc", ACCENT_3)
    draw_box(ax, (0.39, 0.19), (0.22, 0.15), "Counting error\nabsolute and squared", "#fff7ed", WARN)
    draw_box(ax, (0.74, 0.19), (0.22, 0.15), "Reporting\nMAE, RMSE, R2,\nexact match", "#f8fafc", ACCENT_3)

    draw_arrow(ax, (0.87, 0.58), (0.50, 0.34))
    draw_arrow(ax, (0.26, 0.265), (0.39, 0.265))
    draw_arrow(ax, (0.61, 0.265), (0.74, 0.265))

    ax.text(0.04, 0.90, "Detector-based counting protocol", fontsize=10, fontweight="bold")
    ax.text(
        0.04,
        0.855,
        "Each detected bounding box is treated as one larva; predicted counts are compared against the number of annotated larvae per image.",
        fontsize=7.5,
        color=MUTED,
    )
    save_figure(fig, "fig02_detector_counting_pipeline")


def build_dataset_overview(manifest: pd.DataFrame) -> None:
    split_stats = (
        manifest.groupby("split", observed=True)
        .agg(images=("image_name", "count"), larvae=("object_count", "sum"), mean_count=("object_count", "mean"))
        .reindex(["train", "val", "test"])
    )

    fig, axes = plt.subplots(1, 3, figsize=WIDE, gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]})

    colors = [ACCENT, ACCENT_2, ACCENT_3]
    bars = axes[0].bar(split_stats.index.astype(str), split_stats["images"], color=colors, edgecolor=INK, linewidth=0.4)
    axes[0].set_title("(a) Image split")
    axes[0].set_ylabel("Images")
    clean_axis(axes[0])
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, f"{int(bar.get_height())}", ha="center", fontsize=7)

    bars = axes[1].bar(split_stats.index.astype(str), split_stats["larvae"], color=colors, edgecolor=INK, linewidth=0.4)
    axes[1].set_title("(b) Annotated larvae")
    axes[1].set_ylabel("Instances")
    clean_axis(axes[1])
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 35, f"{int(bar.get_height())}", ha="center", fontsize=7)

    bins = np.arange(0, manifest["object_count"].max() + 10, 5)
    for split, color in zip(["train", "val", "test"], colors):
        values = manifest.loc[manifest["split"].astype(str) == split, "object_count"]
        axes[2].hist(values, bins=bins, alpha=0.48, label=split, color=color, edgecolor="white", linewidth=0.25)
    axes[2].set_title("(c) Object-count distribution")
    axes[2].set_xlabel("Larvae per image")
    axes[2].set_ylabel("Images")
    clean_axis(axes[2])
    axes[2].legend(frameon=False)

    fig.suptitle("Dataset overview", x=0.02, ha="left", y=1.02, fontsize=10, fontweight="bold")
    save_figure(fig, "fig03_dataset_overview")


def build_method_family_comparison() -> None:
    table = pd.read_csv(PAPER_DIR / "results_table_detection_counting.csv")
    aggregate = pd.read_csv(YOLO_MULTI_DIR / "aggregate_results.csv")
    run_level = pd.read_csv(YOLO_MULTI_DIR / "run_level_results.csv")

    extra = table.loc[table["method"] == "extra_trees_count_regression"].iloc[0]
    hgb = table.loc[table["method"] == "hist_gradient_boosting_count_regression"].iloc[0]
    watershed = table.loc[table["method"] == "watershed_counting"].iloc[0]
    yolo11n_mean = aggregate.loc[aggregate["variant"] == "yolo11n"].iloc[0]
    yolo11n_best = run_level.loc[run_level["test_count_mae"].idxmin()]

    rows = pd.DataFrame(
        [
            {"label": "Direct CV\nWatershed", "mae": watershed["test_mae"], "r2": watershed["test_r2"], "color": ACCENT_2},
            {"label": "Classical reg.\nExtra Trees", "mae": extra["test_mae"], "r2": extra["test_r2"], "color": ACCENT_3},
            {"label": "Classical reg.\nHGB", "mae": hgb["test_mae"], "r2": hgb["test_r2"], "color": "#4b5563"},
            {"label": "YOLO11n\nmean", "mae": yolo11n_mean["test_count_mae_mean"], "r2": yolo11n_mean["test_count_r2_mean"], "color": ACCENT},
            {"label": "YOLO11n\nbest run", "mae": yolo11n_best["test_count_mae"], "r2": yolo11n_best["test_count_r2"], "color": GOOD},
        ]
    )

    plot_rows = rows.iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    y = np.arange(len(plot_rows))
    bars = axes[0].barh(y, plot_rows["mae"], color=plot_rows["color"], edgecolor=INK, linewidth=0.4)
    axes[0].set_title("(a) Counting error")
    axes[0].set_xlabel("MAE (lower is better)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_rows["label"])
    clean_axis(axes[0], grid_axis="x")
    for bar in bars:
        axes[0].text(bar.get_width() + 0.25, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}", va="center", fontsize=7)

    bars = axes[1].barh(y, plot_rows["r2"], color=plot_rows["color"], edgecolor=INK, linewidth=0.4)
    axes[1].set_title("(b) Count-fit quality")
    axes[1].set_xlabel("R2 (higher is better)")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0, 1)
    clean_axis(axes[1], grid_axis="x")
    for bar in bars:
        axes[1].text(bar.get_width() + 0.018, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}", va="center", fontsize=7)

    fig.suptitle("Best representative methods by family", x=0.02, ha="left", y=1.03, fontsize=10, fontweight="bold")
    save_figure(fig, "fig04_method_family_comparison")


def build_classical_counting_benchmark() -> None:
    ranking = pd.read_csv(REPORTS_DIR / "detection_counting_benchmark" / "ranking.csv")
    ranking = ranking.loc[ranking["method"] != "yolo_v8n_smoke"].copy()
    ranking["label"] = ranking["method"].map(METHOD_LABELS).fillna(ranking["method"])
    ranking = ranking.sort_values("test_mae", ascending=True)

    colors = []
    for method in ranking["method"]:
        if method in {"extra_trees_count_regression", "hist_gradient_boosting_count_regression"}:
            colors.append(GOOD)
        elif method.endswith("_count_regression"):
            colors.append(ACCENT_3)
        else:
            colors.append(ACCENT_2)

    fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"width_ratios": [1.1, 0.9]})

    y = np.arange(len(ranking))
    bars = axes[0].barh(y, ranking["test_mae"], color=colors, edgecolor=INK, linewidth=0.35)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ranking["label"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE (lower is better)")
    axes[0].set_title("(a) All non-YOLO counting baselines")
    clean_axis(axes[0], grid_axis="x")
    label_bars(axes[0], bars, "{:.1f}", xpad=0.01)

    top = ranking.head(4).copy()
    x = np.arange(len(top))
    axes[1].bar(x - 0.18, top["test_mae"], width=0.36, color=ACCENT, edgecolor=INK, linewidth=0.35, label="MAE")
    ax2 = axes[1].twinx()
    ax2.bar(x + 0.18, top["test_r2"], width=0.36, color=GOOD, edgecolor=INK, linewidth=0.35, label="R2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(top["label"], rotation=25, ha="right")
    axes[1].set_ylabel("MAE")
    ax2.set_ylabel("R2")
    axes[1].set_title("(b) Top feature regressors")
    clean_axis(axes[1])
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, 1)
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")

    fig.suptitle("Classical counting benchmark", x=0.02, ha="left", y=1.02, fontsize=10, fontweight="bold")
    save_figure(fig, "fig05_classical_counting_benchmark")


def build_yolo_multiseed_metrics() -> None:
    aggregate = pd.read_csv(YOLO_MULTI_DIR / "aggregate_results.csv")
    aggregate["label"] = aggregate["variant"].map({"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n"}).fillna(aggregate["variant"])
    aggregate = aggregate.sort_values("label")

    fig, axes = plt.subplots(1, 4, figsize=WIDE)
    metrics = [
        ("test_detection_mAP50_mean", "test_detection_mAP50_std", "mAP50", "(a) Detection"),
        ("test_detection_mAP50_95_mean", "test_detection_mAP50_95_std", "mAP50-95", "(b) Localization"),
        ("test_count_mae_mean", "test_count_mae_std", "MAE", "(c) Counting error"),
        ("test_count_r2_mean", "test_count_r2_std", "R2", "(d) Count fit"),
    ]
    colors = [ACCENT_3, ACCENT]
    for ax, (mean_col, std_col, ylabel, title) in zip(axes, metrics):
        x = np.arange(len(aggregate))
        ax.bar(
            x,
            aggregate[mean_col],
            yerr=aggregate[std_col],
            color=colors,
            edgecolor=INK,
            linewidth=0.35,
            capsize=3,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(aggregate["label"], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        clean_axis(ax)
        if "mAP" in ylabel or ylabel == "R2":
            ax.set_ylim(0, 1)
        for idx, value in enumerate(aggregate[mean_col]):
            ax.text(idx, value + (0.03 if ylabel != "MAE" else 0.7), f"{value:.3f}" if ylabel != "MAE" else f"{value:.2f}", ha="center", fontsize=6.5)

    fig.suptitle("Multi-seed YOLO evaluation (mean +/- SD, n = 3)", x=0.02, ha="left", y=1.04, fontsize=10, fontweight="bold")
    save_figure(fig, "fig06_yolo_multiseed_metrics")


def build_density_error_analysis() -> None:
    density = pd.read_csv(YOLO_MULTI_DIR / "density_summary.csv")
    density["label"] = density["variant"].map({"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n"}).fillna(density["variant"])
    density["density_bin"] = pd.Categorical(density["density_bin"], categories=["low", "medium", "high"], ordered=True)
    density = density.sort_values(["density_bin", "label"])

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    bins = ["low", "medium", "high"]
    labels = ["YOLOv8n", "YOLO11n"]
    x = np.arange(len(bins))
    width = 0.34
    colors = {"YOLOv8n": ACCENT_3, "YOLO11n": ACCENT}

    for offset, label in [(-width / 2, "YOLOv8n"), (width / 2, "YOLO11n")]:
        sub = density.loc[density["label"] == label].set_index("density_bin").reindex(bins)
        axes[0].bar(
            x + offset,
            sub["mae_mean"],
            width=width,
            yerr=sub["mae_std"],
            label=label,
            color=colors[label],
            edgecolor=INK,
            linewidth=0.35,
            capsize=3,
        )
        axes[1].bar(
            x + offset,
            sub["exact_match_rate_mean"],
            width=width,
            yerr=sub["exact_match_rate_std"],
            label=label,
            color=colors[label],
            edgecolor=INK,
            linewidth=0.35,
            capsize=3,
        )

    for ax, ylabel, title in [
        (axes[0], "MAE", "(a) Error by density"),
        (axes[1], "Exact match rate", "(b) Exact-count agreement"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Density bin")
        ax.set_title(title)
        clean_axis(ax)
    axes[1].set_ylim(0, 0.75)
    axes[0].legend(frameon=False)

    fig.suptitle("Density-aware counting analysis", x=0.02, ha="left", y=1.04, fontsize=10, fontweight="bold")
    save_figure(fig, "fig07_density_error_analysis")


def build_prediction_scatter() -> None:
    classical = pd.read_csv(REPORTS_DIR / "detection_counting_benchmark" / "predictions.csv")
    yolo = pd.read_csv(YOLO_MULTI_DIR / "predictions.csv")

    panels = [
        (
            classical.loc[(classical["method"] == "extra_trees_count_regression") & (classical["split"] == "test")].copy(),
            "Extra Trees",
            "MAE = 7.03",
            ACCENT_3,
        ),
        (
            yolo.loc[(yolo["method"] == "yolo11n_seed2") & (yolo["split"] == "test")].copy(),
            "YOLO11n seed 2",
            "MAE = 6.07",
            ACCENT,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    for ax, (frame, title, subtitle, color) in zip(axes, panels):
        frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
        frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
        max_value = max(frame["y_true"].max(), frame["y_pred"].max()) + 5
        ax.scatter(frame["y_true"], frame["y_pred"], s=18, color=color, alpha=0.82, edgecolor="white", linewidth=0.25)
        ax.plot([0, max_value], [0, max_value], color=BAD, linewidth=1.0, linestyle="--", label="ideal")
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)
        ax.set_xlabel("Ground-truth count")
        ax.set_ylabel("Predicted count")
        ax.set_title(f"{title}\n{subtitle}")
        clean_axis(ax)
        ax.legend(frameon=False, loc="upper left")

    fig.suptitle("Predicted versus ground-truth larvae counts on the test set", x=0.02, ha="left", y=1.04, fontsize=10, fontweight="bold")
    save_figure(fig, "fig08_prediction_scatter")


def build_training_curves() -> None:
    run_level = pd.read_csv(YOLO_MULTI_DIR / "run_level_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    colors = {"yolov8n": ACCENT_3, "yolo11n": ACCENT}

    for variant, sub in run_level.groupby("variant"):
        curves = []
        for _, row in sub.iterrows():
            csv_path = Path(str(row["run_dir"]).replace("/workspace", str(PROJECT_ROOT).replace("\\", "/"))) / "results.csv"
            csv_path = Path(str(csv_path))
            if not csv_path.exists():
                continue
            curve = pd.read_csv(csv_path)
            curves.append(curve[["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]])
        if not curves:
            continue
        merged = pd.concat(curves, keys=range(len(curves)), names=["run", "idx"]).reset_index(level="run")
        for ax, col, ylabel in [
            (axes[0], "metrics/mAP50(B)", "mAP50"),
            (axes[1], "metrics/mAP50-95(B)", "mAP50-95"),
        ]:
            stats = merged.groupby("epoch")[col].agg(["mean", "std"]).reset_index()
            ax.plot(stats["epoch"], stats["mean"], color=colors.get(variant, MUTED), linewidth=1.4, label=METHOD_LABELS.get(variant, variant))
            ax.fill_between(
                stats["epoch"].to_numpy(float),
                (stats["mean"] - stats["std"]).fillna(0).to_numpy(float),
                (stats["mean"] + stats["std"]).fillna(0).to_numpy(float),
                color=colors.get(variant, MUTED),
                alpha=0.16,
                linewidth=0,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)
            clean_axis(ax)

    axes[0].set_title("(a) mAP50")
    axes[1].set_title("(b) mAP50-95")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("YOLO validation convergence across seeds", x=0.02, ha="left", y=1.04, fontsize=10, fontweight="bold")
    save_figure(fig, "fig09_training_curves")


def build_qualitative_montage() -> None:
    source = YOLO_MULTI_DIR / "plots" / "qualitative_success_failure_montage.png"
    if not source.exists():
        return

    ensure_dir()
    target_png = FIG_DIR / "fig10_qualitative_success_failure_montage.png"
    target_pdf = FIG_DIR / "fig10_qualitative_success_failure_montage.pdf"

    image = Image.open(source).convert("RGB")
    image = ImageOps.contain(image, (4200, 2800), method=Image.Resampling.LANCZOS)
    image.save(target_png, dpi=(DPI, DPI), optimize=True)

    fig, ax = plt.subplots(figsize=(7.16, 4.3))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Qualitative success and failure examples", loc="left", fontsize=10, fontweight="bold")
    fig.savefig(target_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_caption_file(manifest: pd.DataFrame) -> None:
    split_stats = (
        manifest.groupby("split", observed=True)
        .agg(images=("image_name", "count"), larvae=("object_count", "sum"))
        .reindex(["train", "val", "test"])
    )
    captions = f"""# IEEE Figure Captions

Use the `.pdf` files for IEEE whenever possible because they preserve vector text and lines. Use the `.png` files as 600 dpi raster fallbacks.

1. **Fig. 1. Overall experimental workflow.** Raw BSF larvae images and YOLO-format annotations were curated into a benchmark-ready manifest, split into train, validation, and test partitions, and evaluated using classical image-processing baselines, feature-based count regressors, and YOLO detector-based counting.

2. **Fig. 2. Detector-based counting protocol.** Each predicted bounding box is interpreted as one larva. The resulting predicted count is compared with the number of annotated larvae per image using MAE, RMSE, R2, and exact-count agreement.

3. **Fig. 3. Dataset overview.** The benchmark contains {int(split_stats.loc['train', 'images'])} training, {int(split_stats.loc['val', 'images'])} validation, and {int(split_stats.loc['test', 'images'])} test images, corresponding to {int(split_stats.loc['train', 'larvae'])}, {int(split_stats.loc['val', 'larvae'])}, and {int(split_stats.loc['test', 'larvae'])} annotated larvae, respectively.

4. **Fig. 4. Best representative methods by family.** Direct classical computer vision was substantially weaker than feature-based count regressors and detector-based counting. The best single detector-based run (YOLO11n seed 2) achieved the lowest test MAE.

5. **Fig. 5. Classical counting benchmark.** Feature-based regressors were the strongest non-deep-learning methods. Extra Trees produced the lowest classical MAE, while HistGradientBoosting produced the strongest classical RMSE/R2 behavior.

6. **Fig. 6. Multi-seed YOLO evaluation.** Bars show mean +/- standard deviation across three random seeds. YOLOv8n achieved the strongest average detection scores, while YOLO11n achieved the strongest average detector-based counting error.

7. **Fig. 7. Density-aware counting analysis.** Counting error increased substantially in high-density scenes, indicating that crowded larvae regions are the main remaining failure mode.

8. **Fig. 8. Predicted versus ground-truth counts.** Scatter plots compare the strongest classical MAE baseline (Extra Trees) against the best single detector-based counting run (YOLO11n seed 2) on the test set.

9. **Fig. 9. YOLO validation convergence across seeds.** Validation mAP curves show that both lightweight detectors converged within the 20-epoch training budget, with shaded regions representing one standard deviation across seeds.

10. **Fig. 10. Qualitative success and failure examples.** Representative test images illustrate exact or near-exact counting cases and failure cases, especially over-counting in crowded scenes.
"""
    (FIG_DIR / "figure_captions_ieee.md").write_text(captions, encoding="utf-8")


def build_readme() -> None:
    files = sorted(FIG_DIR.glob("fig*"))
    rows = []
    for path in files:
        rows.append(f"- `{path.name}`")
    for path in [FIG_DIR / "figure_captions_ieee.md", FIG_DIR / "ieee_figure_snippets.tex"]:
        if path.exists():
            rows.append(f"- `{path.name}`")
    readme = "# IEEE-ready figures\n\n" + "\n".join(rows) + "\n\nRegenerate with:\n\n```powershell\npython scripts\\build_ieee_figures.py\n```\n"
    (FIG_DIR / "README.md").write_text(readme, encoding="utf-8")


def build_latex_snippets() -> None:
    snippets = r"""% IEEE figure snippets.
% Put the PDF figures in the same relative path or update the paths below.

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{reports/paper_q4_prep/ieee_figures/fig01_methodology_flowchart.pdf}
  \caption{Overall experimental workflow. Raw BSF larvae images and YOLO-format annotations were curated into a benchmark-ready manifest, split into train, validation, and test partitions, and evaluated using classical image-processing baselines, feature-based count regressors, and YOLO detector-based counting.}
  \label{fig:methodology_workflow}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{reports/paper_q4_prep/ieee_figures/fig02_detector_counting_pipeline.pdf}
  \caption{Detector-based counting protocol. Each predicted bounding box is interpreted as one larva, and the resulting predicted count is compared with the number of annotated larvae per image.}
  \label{fig:detector_counting_protocol}
\end{figure}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{reports/paper_q4_prep/ieee_figures/fig03_dataset_overview.pdf}
  \caption{Dataset overview showing train, validation, and test image counts, annotated larvae counts, and object-count distribution.}
  \label{fig:dataset_overview}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{reports/paper_q4_prep/ieee_figures/fig04_method_family_comparison.pdf}
  \caption{Best representative methods by family. Direct classical computer vision was weaker than feature-based count regressors and detector-based counting, while the best YOLO11n run achieved the lowest test MAE.}
  \label{fig:method_family_comparison}
\end{figure}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{reports/paper_q4_prep/ieee_figures/fig05_classical_counting_benchmark.pdf}
  \caption{Classical counting benchmark. Feature-based regressors were the strongest non-deep-learning methods; Extra Trees produced the lowest classical MAE, while HistGradientBoosting produced the strongest classical RMSE/R2 behavior.}
  \label{fig:classical_counting_benchmark}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{reports/paper_q4_prep/ieee_figures/fig06_yolo_multiseed_metrics.pdf}
  \caption{Multi-seed YOLO evaluation. Bars show mean and standard deviation across three random seeds for YOLOv8n and YOLO11n.}
  \label{fig:yolo_multiseed_metrics}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{reports/paper_q4_prep/ieee_figures/fig07_density_error_analysis.pdf}
  \caption{Density-aware counting analysis. Counting error increased substantially in high-density scenes, indicating that crowded larvae regions are the main remaining failure mode.}
  \label{fig:density_error_analysis}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{reports/paper_q4_prep/ieee_figures/fig08_prediction_scatter.pdf}
  \caption{Predicted versus ground-truth counts for the strongest classical MAE baseline and the best detector-based single run on the test set.}
  \label{fig:prediction_scatter}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{reports/paper_q4_prep/ieee_figures/fig09_training_curves.pdf}
  \caption{YOLO validation convergence across seeds. Shaded regions represent one standard deviation across seeds.}
  \label{fig:training_curves}
\end{figure}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{reports/paper_q4_prep/ieee_figures/fig10_qualitative_success_failure_montage.pdf}
  \caption{Qualitative success and failure examples from the best detector-based run, including exact-count successes and crowded-scene over-counting failures.}
  \label{fig:qualitative_examples}
\end{figure*}
"""
    (FIG_DIR / "ieee_figure_snippets.tex").write_text(snippets, encoding="utf-8")


def main() -> None:
    ensure_dir()
    ieee_style()
    manifest = load_manifest()

    build_methodology_flowchart()
    build_detector_counting_pipeline()
    build_dataset_overview(manifest)
    build_method_family_comparison()
    build_classical_counting_benchmark()
    build_yolo_multiseed_metrics()
    build_density_error_analysis()
    build_prediction_scatter()
    build_training_curves()
    build_qualitative_montage()
    build_caption_file(manifest)
    build_latex_snippets()
    build_readme()

    print(f"Saved IEEE-ready figures to {FIG_DIR}")
    for path in sorted(FIG_DIR.glob("fig*")):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
