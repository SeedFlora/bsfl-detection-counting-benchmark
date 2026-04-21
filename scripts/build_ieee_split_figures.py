from __future__ import annotations

import json
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
FIG_DIR = PAPER_DIR / "ieee_figures_split"
YOLO_MULTI_DIR = REPORTS_DIR / "yolo_multiseed" / "yolo_multiseed_strengthening"

DPI = 600
INK = "#1f2933"
MUTED = "#64748b"
GRID = "#d9e2ec"
ACCENT = "#007c89"
TEAL_LIGHT = "#dff3f5"
AMBER = "#b26a00"
GRAY = "#6b7280"
DARK_GRAY = "#4b5563"
GREEN = "#198754"
RED = "#b3261e"

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
}


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    ensure_dir()
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.16)
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str = "#f8fafc",
    edge: str = ACCENT,
    size: float = 8,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=MUTED,
            shrinkA=3,
            shrinkB=3,
        )
    )


def manifest() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_DIR / "detection_counting_manifest.csv")
    frame["object_count"] = pd.to_numeric(frame["object_count"], errors="coerce")
    frame["split"] = pd.Categorical(frame["split"], categories=["train", "val", "test"], ordered=True)
    return frame


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(PAPER_DIR / "results_table_detection_counting.csv")
    ranking = pd.read_csv(REPORTS_DIR / "detection_counting_benchmark" / "ranking.csv")
    aggregate = pd.read_csv(YOLO_MULTI_DIR / "aggregate_results.csv")
    run_level = pd.read_csv(YOLO_MULTI_DIR / "run_level_results.csv")
    return table, ranking, aggregate, run_level


def methodology_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("Raw images\nand YOLO labels", 0.04, TEAL_LIGHT, ACCENT),
        ("Dataset curation\nand label checks", 0.27, "#f8fafc", ACCENT),
        ("Train, validation,\nand test split", 0.50, "#f8fafc", ACCENT),
        ("Benchmark-ready\nmanifest", 0.73, TEAL_LIGHT, ACCENT),
    ]
    for text, x, face, edge in steps:
        draw_box(ax, x, 0.68, 0.18, 0.18, text, face, edge)
    for start, end in [((0.22, 0.77), (0.27, 0.77)), ((0.45, 0.77), (0.50, 0.77)), ((0.68, 0.77), (0.73, 0.77))]:
        draw_arrow(ax, start, end)

    branches = [
        ("Direct image\nprocessing", 0.07, "#fff7ed", AMBER),
        ("Feature-based\ncount regressors", 0.39, "#f3f4f6", GRAY),
        ("YOLO detector-\nbased counting", 0.71, TEAL_LIGHT, ACCENT),
    ]
    for text, x, face, edge in branches:
        draw_box(ax, x, 0.36, 0.22, 0.18, text, face, edge)
        draw_arrow(ax, (0.82, 0.68), (x + 0.11, 0.54))

    draw_box(ax, 0.24, 0.08, 0.20, 0.15, "Predicted\nlarvae count", "#f8fafc", ACCENT)
    draw_box(ax, 0.56, 0.08, 0.20, 0.15, "Evaluation\nMAE, RMSE, R2, mAP", TEAL_LIGHT, ACCENT)
    for x in [0.18, 0.50, 0.82]:
        draw_arrow(ax, (x, 0.36), (0.34, 0.23))
    draw_arrow(ax, (0.44, 0.155), (0.56, 0.155))

    ax.text(0.02, 0.95, "Experimental workflow", fontsize=11, fontweight="bold", ha="left")
    save(fig, "fig01_methodology_flowchart")


def detector_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.6), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("Input\nimage", 0.04),
        ("YOLO\ninference", 0.28),
        ("Confidence filtering\nand NMS", 0.52),
        ("Detected boxes\nas predicted count", 0.76),
    ]
    for text, x in steps:
        draw_box(ax, x, 0.56, 0.18, 0.20, text, TEAL_LIGHT, ACCENT)
    for start, end in [((0.22, 0.66), (0.28, 0.66)), ((0.46, 0.66), (0.52, 0.66)), ((0.70, 0.66), (0.76, 0.66))]:
        draw_arrow(ax, start, end)

    draw_box(ax, 0.08, 0.17, 0.22, 0.17, "Annotation file\ntrue count", "#f8fafc", GRAY)
    draw_box(ax, 0.39, 0.17, 0.22, 0.17, "Count comparison\nper image", "#fff7ed", AMBER)
    draw_box(ax, 0.70, 0.17, 0.22, 0.17, "Counting metrics\nMAE, RMSE, R2", "#f8fafc", GRAY)
    draw_arrow(ax, (0.85, 0.56), (0.50, 0.34))
    draw_arrow(ax, (0.30, 0.255), (0.39, 0.255))
    draw_arrow(ax, (0.61, 0.255), (0.70, 0.255))

    ax.text(0.02, 0.92, "Detector-based counting protocol", fontsize=11, fontweight="bold", ha="left")
    save(fig, "fig02_detector_counting_pipeline")


def dataset_figures(frame: pd.DataFrame) -> None:
    stats = (
        frame.groupby("split", observed=True)
        .agg(images=("image_name", "count"), larvae=("object_count", "sum"))
        .reindex(["train", "val", "test"])
    )
    colors = [ACCENT, AMBER, GRAY]

    fig, ax = plt.subplots(figsize=(3.7, 2.8), constrained_layout=True)
    bars = ax.bar(stats.index.astype(str), stats["images"], color=colors, edgecolor=INK, linewidth=0.4)
    ax.set_title("Image split")
    ax.set_ylabel("Images")
    clean_axis(ax)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, f"{int(bar.get_height())}", ha="center")
    save(fig, "fig03_dataset_image_split")

    fig, ax = plt.subplots(figsize=(3.7, 2.8), constrained_layout=True)
    bars = ax.bar(stats.index.astype(str), stats["larvae"], color=colors, edgecolor=INK, linewidth=0.4)
    ax.set_title("Annotated larvae per split")
    ax.set_ylabel("Larvae instances")
    clean_axis(ax)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 70, f"{int(bar.get_height())}", ha="center")
    save(fig, "fig04_dataset_instance_split")

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    bins = np.arange(0, frame["object_count"].max() + 10, 5)
    for split, color in zip(["train", "val", "test"], colors):
        ax.hist(
            frame.loc[frame["split"].astype(str) == split, "object_count"],
            bins=bins,
            alpha=0.52,
            label=split,
            color=color,
            edgecolor="white",
            linewidth=0.25,
        )
    ax.set_title("Object-count distribution")
    ax.set_xlabel("Larvae per image")
    ax.set_ylabel("Images")
    clean_axis(ax)
    ax.legend(frameon=False)
    save(fig, "fig05_dataset_count_distribution")


def family_metric_figures(table: pd.DataFrame, aggregate: pd.DataFrame, run_level: pd.DataFrame) -> None:
    rows = pd.DataFrame(
        [
            {
                "label": "Direct CV\nWatershed",
                "mae": table.loc[table["method"] == "watershed_counting", "test_mae"].iloc[0],
                "r2": table.loc[table["method"] == "watershed_counting", "test_r2"].iloc[0],
                "color": AMBER,
            },
            {
                "label": "Classical reg.\nExtra Trees",
                "mae": table.loc[table["method"] == "extra_trees_count_regression", "test_mae"].iloc[0],
                "r2": table.loc[table["method"] == "extra_trees_count_regression", "test_r2"].iloc[0],
                "color": GRAY,
            },
            {
                "label": "Classical reg.\nHistGradientBoosting",
                "mae": table.loc[table["method"] == "hist_gradient_boosting_count_regression", "test_mae"].iloc[0],
                "r2": table.loc[table["method"] == "hist_gradient_boosting_count_regression", "test_r2"].iloc[0],
                "color": DARK_GRAY,
            },
            {
                "label": "YOLO11n\nmean",
                "mae": aggregate.loc[aggregate["variant"] == "yolo11n", "test_count_mae_mean"].iloc[0],
                "r2": aggregate.loc[aggregate["variant"] == "yolo11n", "test_count_r2_mean"].iloc[0],
                "color": ACCENT,
            },
            {
                "label": "YOLO11n\nbest run",
                "mae": run_level["test_count_mae"].min(),
                "r2": run_level.loc[run_level["test_count_mae"].idxmin(), "test_count_r2"],
                "color": GREEN,
            },
        ]
    )
    rows = rows.iloc[::-1].reset_index(drop=True)

    for metric, title, xlabel, stem, limit in [
        ("mae", "Counting error by method family", "MAE (lower is better)", "fig06_method_family_mae", None),
        ("r2", "Count-fit quality by method family", "R2 (higher is better)", "fig07_method_family_r2", (0, 1.0)),
    ]:
        fig, ax = plt.subplots(figsize=(4.3, 3.1), constrained_layout=True)
        y = np.arange(len(rows))
        bars = ax.barh(y, rows[metric], color=rows["color"], edgecolor=INK, linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(rows["label"])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        if limit:
            ax.set_xlim(*limit)
        clean_axis(ax, grid_axis="x")
        pad = 0.25 if metric == "mae" else 0.02
        fmt = "{:.2f}" if metric == "mae" else "{:.3f}"
        for bar in bars:
            ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2, fmt.format(bar.get_width()), va="center")
        save(fig, stem)


def classical_figures(ranking: pd.DataFrame) -> None:
    ranking = ranking.loc[ranking["method"] != "yolo_v8n_smoke"].copy()
    ranking["label"] = ranking["method"].map(METHOD_LABELS).fillna(ranking["method"])
    ranking = ranking.sort_values("test_mae", ascending=True)
    colors = [GREEN if method in {"extra_trees_count_regression", "hist_gradient_boosting_count_regression"} else GRAY if method.endswith("_count_regression") else AMBER for method in ranking["method"]]

    fig, ax = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)
    y = np.arange(len(ranking))
    bars = ax.barh(y, ranking["test_mae"], color=colors, edgecolor=INK, linewidth=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(ranking["label"])
    ax.invert_yaxis()
    ax.set_xlabel("MAE (lower is better)")
    ax.set_title("Classical counting baseline MAE")
    clean_axis(ax, grid_axis="x")
    for bar in bars:
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}", va="center")
    save(fig, "fig08_classical_baseline_mae")

    top = ranking.head(4).copy()
    for metric, title, ylabel, stem, ylim in [
        ("test_mae", "Top classical regressor MAE", "MAE", "fig09_top_classical_mae", None),
        ("test_r2", "Top classical regressor R2", "R2", "fig10_top_classical_r2", (0, 1.0)),
    ]:
        plot_top = top.iloc[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
        y = np.arange(len(plot_top))
        bars = ax.barh(y, plot_top[metric], color=[ACCENT, GRAY, DARK_GRAY, "#94a3b8"][::-1], edgecolor=INK, linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel(ylabel)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_top["label"])
        if ylim:
            ax.set_xlim(*ylim)
        clean_axis(ax, grid_axis="x")
        pad = 0.25 if metric == "test_mae" else 0.02
        for bar in bars:
            value = bar.get_width()
            text = f"{value:.2f}" if metric == "test_mae" else f"{value:.3f}"
            ax.text(value + pad, bar.get_y() + bar.get_height() / 2, text, va="center")
        save(fig, stem)


def yolo_metric_figures(aggregate: pd.DataFrame) -> None:
    data = aggregate.copy()
    data["label"] = data["variant"].map({"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n"})
    data = data.sort_values("label")
    specs = [
        ("test_detection_mAP50_mean", "test_detection_mAP50_std", "mAP50", "YOLO average detection mAP50", "fig11_yolo_map50", (0, 1.0)),
        ("test_detection_mAP50_95_mean", "test_detection_mAP50_95_std", "mAP50-95", "YOLO average detection mAP50-95", "fig12_yolo_map50_95", (0, 1.0)),
        ("test_count_mae_mean", "test_count_mae_std", "MAE", "YOLO average counting MAE", "fig13_yolo_count_mae", None),
        ("test_count_r2_mean", "test_count_r2_std", "R2", "YOLO average counting R2", "fig14_yolo_count_r2", (0, 1.0)),
    ]
    for mean_col, std_col, ylabel, title, stem, ylim in specs:
        fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
        bars = ax.bar(
            data["label"],
            data[mean_col],
            yerr=data[std_col],
            color=[ACCENT, GRAY],
            edgecolor=INK,
            linewidth=0.4,
            capsize=4,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        clean_axis(ax)
        for bar in bars:
            value = bar.get_height()
            text = f"{value:.2f}" if ylabel == "MAE" else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + (0.55 if ylabel == "MAE" else 0.025), text, ha="center")
        save(fig, stem)


def density_figures() -> None:
    density = pd.read_csv(YOLO_MULTI_DIR / "density_summary.csv")
    density["label"] = density["variant"].map({"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n"})
    density["density_bin"] = pd.Categorical(density["density_bin"], categories=["low", "medium", "high"], ordered=True)
    colors = {"YOLOv8n": GRAY, "YOLO11n": ACCENT}
    bins = ["low", "medium", "high"]
    x = np.arange(len(bins))
    width = 0.34

    for metric, err, ylabel, title, stem, ylim in [
        ("mae_mean", "mae_std", "MAE", "Counting error by density", "fig15_density_mae", None),
        ("exact_match_rate_mean", "exact_match_rate_std", "Exact match rate", "Exact-count agreement by density", "fig16_density_exact_match", (0, 0.75)),
    ]:
        fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
        for offset, label in [(-width / 2, "YOLOv8n"), (width / 2, "YOLO11n")]:
            sub = density.loc[density["label"] == label].set_index("density_bin").reindex(bins)
            ax.bar(
                x + offset,
                sub[metric],
                width=width,
                yerr=sub[err],
                color=colors[label],
                edgecolor=INK,
                linewidth=0.4,
                capsize=4,
                label=label,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Density bin")
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Medium", "High"])
        if ylim:
            ax.set_ylim(*ylim)
        clean_axis(ax)
        ax.legend(frameon=False)
        save(fig, stem)


def scatter_figures() -> None:
    classical = pd.read_csv(REPORTS_DIR / "detection_counting_benchmark" / "predictions.csv")
    yolo = pd.read_csv(YOLO_MULTI_DIR / "predictions.csv")
    panels = [
        (
            classical.loc[(classical["method"] == "extra_trees_count_regression") & (classical["split"] == "test")].copy(),
            "Extra Trees count prediction",
            "MAE = 7.03",
            GRAY,
            "fig17_scatter_extra_trees",
        ),
        (
            yolo.loc[(yolo["method"] == "yolo11n_seed2") & (yolo["split"] == "test")].copy(),
            "YOLO11n seed 2 count prediction",
            "MAE = 6.07",
            ACCENT,
            "fig18_scatter_yolo11n_seed2",
        ),
    ]
    for frame, title, subtitle, color, stem in panels:
        frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
        frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
        max_value = max(frame["y_true"].max(), frame["y_pred"].max()) + 7
        fig, ax = plt.subplots(figsize=(3.8, 3.3), constrained_layout=True)
        ax.scatter(frame["y_true"], frame["y_pred"], s=22, color=color, alpha=0.84, edgecolor="white", linewidth=0.3)
        ax.plot([0, max_value], [0, max_value], color=RED, linewidth=1.1, linestyle="--", label="Ideal")
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel("Ground-truth count")
        ax.set_ylabel("Predicted count")
        clean_axis(ax)
        ax.legend(frameon=False, loc="upper left")
        save(fig, stem)


def training_figures(run_level: pd.DataFrame) -> None:
    colors = {"yolov8n": GRAY, "yolo11n": ACCENT}
    for metric, ylabel, title, stem in [
        ("metrics/mAP50(B)", "mAP50", "YOLO validation mAP50 convergence", "fig19_training_map50"),
        ("metrics/mAP50-95(B)", "mAP50-95", "YOLO validation mAP50-95 convergence", "fig20_training_map50_95"),
    ]:
        fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
        for variant, sub in run_level.groupby("variant"):
            curves = []
            for _, row in sub.iterrows():
                run_dir = str(row["run_dir"]).replace("/workspace", str(PROJECT_ROOT).replace("\\", "/"))
                csv_path = Path(run_dir) / "results.csv"
                if csv_path.exists():
                    curves.append(pd.read_csv(csv_path)[["epoch", metric]])
            if not curves:
                continue
            merged = pd.concat(curves, keys=range(len(curves)), names=["run", "idx"]).reset_index(level="run")
            stats = merged.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(stats["epoch"], stats["mean"], color=colors.get(variant, MUTED), linewidth=1.5, label=variant.replace("yolo", "YOLO"))
            ax.fill_between(
                stats["epoch"].to_numpy(float),
                (stats["mean"] - stats["std"]).fillna(0).to_numpy(float),
                (stats["mean"] + stats["std"]).fillna(0).to_numpy(float),
                color=colors.get(variant, MUTED),
                alpha=0.16,
                linewidth=0,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        clean_axis(ax)
        ax.legend(frameon=False, loc="lower right")
        save(fig, stem)


def qualitative() -> None:
    source = YOLO_MULTI_DIR / "plots" / "qualitative_success_failure_montage.png"
    if not source.exists():
        return
    ensure_dir()
    image = Image.open(source).convert("RGB")
    image = ImageOps.contain(image, (4200, 2800), method=Image.Resampling.LANCZOS)
    image.save(FIG_DIR / "fig21_qualitative_success_failure_examples.png", dpi=(DPI, DPI), optimize=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    ax.imshow(image)
    ax.axis("off")
    save(fig, "fig21_qualitative_success_failure_examples")


def captions(frame: pd.DataFrame) -> None:
    stats = (
        frame.groupby("split", observed=True)
        .agg(images=("image_name", "count"), larvae=("object_count", "sum"))
        .reindex(["train", "val", "test"])
    )
    text = f"""# IEEE Split Figure Captions

These captions match the single-panel figures in `ieee_figures_split`. The files avoid subfigure labels and provide extra whitespace to reduce clipping in IEEE layouts.

1. **Fig. 1. Experimental workflow.** Raw images and YOLO-format annotations were curated into a benchmark-ready manifest and evaluated using direct image-processing methods, feature-based count regressors, and detector-based counting.

2. **Fig. 2. Detector-based counting protocol.** Each detected bounding box is interpreted as one larva, and the predicted image-level count is compared with the annotated count.

3. **Fig. 3. Image split.** The benchmark contains {int(stats.loc['train', 'images'])} training, {int(stats.loc['val', 'images'])} validation, and {int(stats.loc['test', 'images'])} test images.

4. **Fig. 4. Annotated larvae per split.** The training, validation, and test splits contain {int(stats.loc['train', 'larvae'])}, {int(stats.loc['val', 'larvae'])}, and {int(stats.loc['test', 'larvae'])} annotated larvae, respectively.

5. **Fig. 5. Object-count distribution.** Larvae counts vary substantially across images, with a smaller number of high-density images.

6. **Fig. 6. Counting error by method family.** The best single YOLO11n run achieved the lowest MAE, while feature-based regressors remained strong non-deep-learning baselines.

7. **Fig. 7. Count-fit quality by method family.** HistGradientBoosting and the best YOLO11n run achieved the highest R2 values.

8. **Fig. 8. Classical counting baseline MAE.** Feature-based regressors outperformed direct thresholding, connected-component, blob, and watershed baselines.

9. **Fig. 9. Top classical regressor MAE.** Extra Trees achieved the lowest MAE among the classical count regressors.

10. **Fig. 10. Top classical regressor R2.** HistGradientBoosting achieved the highest R2 among the classical count regressors.

11. **Fig. 11. YOLO average detection mAP50.** YOLOv8n achieved the highest average mAP50 across three seeds.

12. **Fig. 12. YOLO average detection mAP50-95.** YOLOv8n achieved the highest average mAP50-95 across three seeds.

13. **Fig. 13. YOLO average counting MAE.** YOLO11n achieved the lower average detector-based counting MAE across three seeds.

14. **Fig. 14. YOLO average counting R2.** YOLO11n achieved the higher average detector-based count R2 across three seeds.

15. **Fig. 15. Counting error by density.** Counting error increased strongly in high-density scenes.

16. **Fig. 16. Exact-count agreement by density.** Exact-count agreement decreased in high-density scenes.

17. **Fig. 17. Extra Trees count prediction.** Predicted and ground-truth counts are compared on the test set for the strongest classical MAE baseline.

18. **Fig. 18. YOLO11n seed 2 count prediction.** Predicted and ground-truth counts are compared on the test set for the best detector-based single run.

19. **Fig. 19. YOLO validation mAP50 convergence.** Validation mAP50 curves show convergence behavior across seeds.

20. **Fig. 20. YOLO validation mAP50-95 convergence.** Validation mAP50-95 curves show convergence behavior across seeds.

21. **Fig. 21. Qualitative success and failure examples.** Representative test examples illustrate exact-count successes and crowded-scene failures.
"""
    (FIG_DIR / "figure_captions_ieee_split.md").write_text(text, encoding="utf-8")


def snippets() -> None:
    figure_files = sorted(FIG_DIR.glob("fig*.pdf"))
    lines = [
        "% IEEE single-panel figure snippets.",
        "% Use figure* for wide diagrams and qualitative examples; otherwise figure is usually enough.",
        "",
    ]
    wide = {"fig01", "fig21"}
    for path in figure_files:
        stem = path.stem
        env = "figure*" if any(stem.startswith(prefix) for prefix in wide) else "figure"
        width = r"\textwidth" if env == "figure*" else r"\columnwidth"
        label = stem.replace("_", ":")
        title = " ".join(stem.split("_")[1:]).capitalize()
        lines.extend(
            [
                rf"\begin{{{env}}}[t]",
                r"  \centering",
                rf"  \includegraphics[width={width}]{{reports/paper_q4_prep/ieee_figures_split/{path.name}}}",
                rf"  \caption{{{title}.}}",
                rf"  \label{{fig:{label}}}",
                rf"\end{{{env}}}",
                "",
            ]
        )
    (FIG_DIR / "ieee_split_figure_snippets.tex").write_text("\n".join(lines), encoding="utf-8")


def readme() -> None:
    files = sorted([path.name for path in FIG_DIR.glob("*")])
    text = "# IEEE split figures\n\n" + "\n".join(f"- `{name}`" for name in files)
    text += "\n\nRegenerate with:\n\n```powershell\npython scripts\\build_ieee_split_figures.py\n```\n"
    (FIG_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dir()
    style()
    frame = manifest()
    table, ranking, aggregate, run_level = load_results()
    methodology_flowchart()
    detector_pipeline()
    dataset_figures(frame)
    family_metric_figures(table, aggregate, run_level)
    classical_figures(ranking)
    yolo_metric_figures(aggregate)
    density_figures()
    scatter_figures()
    training_figures(run_level)
    qualitative()
    captions(frame)
    snippets()
    readme()

    print(f"Saved split IEEE-ready figures to {FIG_DIR}")
    for path in sorted(FIG_DIR.glob("fig*.pdf")):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
