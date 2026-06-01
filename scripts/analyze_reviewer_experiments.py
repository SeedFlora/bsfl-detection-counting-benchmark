from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.paths import REPORTS_DIR

try:
    from scipy import stats
except Exception:  # pragma: no cover - fallback is for lean environments.
    stats = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise reviewer experiments and run paired statistics.")
    parser.add_argument(
        "--postprocessing-dir",
        default=str(REPORTS_DIR / "reviewer_experiments" / "postprocessing" / "reviewer_full_20260601_postprocessing"),
        help="Directory containing selected_results.csv from benchmark_yolo_postprocessing_sweep.py.",
    )
    parser.add_argument(
        "--robustness-dir",
        default=str(REPORTS_DIR / "reviewer_experiments" / "robustness" / "reviewer_full_20260601_robustness"),
        help="Directory containing condition_results.csv from benchmark_yolo_robustness.py.",
    )
    parser.add_argument(
        "--training-dir",
        default=str(REPORTS_DIR / "reviewer_experiments" / "training" / "reviewer_full_20260601_training_c"),
        help="Directory containing run_level_results.csv from benchmark_yolo_reviewer_training.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPORTS_DIR / "paper_q4_prep"),
        help="Directory for reviewer_experiment_results_summary.md and statistical_tests.csv.",
    )
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[skip] Missing {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def seed_from_label(label: str) -> int | None:
    match = re.search(r"seed(\d+)", str(label))
    return int(match.group(1)) if match else None


def finite_pairs(a: Iterable[float], b: Iterable[float]) -> tuple[list[float], list[float]]:
    aa: list[float] = []
    bb: list[float] = []
    for left, right in zip(a, b):
        if pd.notna(left) and pd.notna(right):
            aa.append(float(left))
            bb.append(float(right))
    return aa, bb


def fallback_two_sided_sign_p(diffs: list[float]) -> float:
    non_zero = [diff for diff in diffs if diff != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    positives = sum(diff > 0 for diff in non_zero)
    k = min(positives, n - positives)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * prob)


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2
        for original_idx, _ in indexed[i:j]:
            ranks[original_idx] = rank
        i = j
    return ranks


def paired_statistics(
    left: Iterable[float],
    right: Iterable[float],
    *,
    comparison: str,
    metric: str,
    left_name: str,
    right_name: str,
    context: str,
) -> dict[str, float | int | str]:
    aa, bb = finite_pairs(left, right)
    diffs = [a - b for a, b in zip(aa, bb)]
    n = len(diffs)
    mean_left = sum(aa) / n if n else float("nan")
    mean_right = sum(bb) / n if n else float("nan")
    mean_diff = sum(diffs) / n if n else float("nan")

    if n > 1:
        diff_series = pd.Series(diffs)
        std_diff = float(diff_series.std(ddof=1))
        dz = mean_diff / std_diff if std_diff else float("nan")
    else:
        std_diff = float("nan")
        dz = float("nan")

    if stats is not None and n > 1:
        try:
            t_result = stats.ttest_rel(aa, bb, nan_policy="omit")
            t_stat = float(t_result.statistic)
            t_p = float(t_result.pvalue)
        except Exception:
            t_stat = float("nan")
            t_p = float("nan")
        try:
            w_p = float(stats.wilcoxon(aa, bb, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            w_p = fallback_two_sided_sign_p(diffs)
    else:
        t_stat = float("nan")
        t_p = float("nan")
        w_p = fallback_two_sided_sign_p(diffs)

    non_zero = [diff for diff in diffs if diff != 0]
    if non_zero:
        abs_ranks = average_ranks([abs(diff) for diff in non_zero])
        w_pos = sum(rank for rank, diff in zip(abs_ranks, non_zero) if diff > 0)
        w_neg = sum(rank for rank, diff in zip(abs_ranks, non_zero) if diff < 0)
        rank_biserial = (w_pos - w_neg) / sum(abs_ranks)
    else:
        rank_biserial = 0.0

    return {
        "context": context,
        "comparison": comparison,
        "metric": metric,
        "left": left_name,
        "right": right_name,
        "n_pairs": n,
        "left_mean": mean_left,
        "right_mean": mean_right,
        "mean_difference_left_minus_right": mean_diff,
        "std_difference": std_diff,
        "paired_t_statistic": t_stat,
        "paired_t_p": t_p,
        "wilcoxon_or_sign_p": w_p,
        "cohens_dz": dz,
        "rank_biserial": rank_biserial,
    }


def compare_by_key(
    frame: pd.DataFrame,
    *,
    key_cols: list[str],
    group_col: str,
    left_value: str,
    right_value: str,
    metrics: list[str],
    context: str,
) -> list[dict[str, float | int | str]]:
    if frame.empty:
        return []
    rows: list[dict[str, float | int | str]] = []
    left = frame[frame[group_col] == left_value]
    right = frame[frame[group_col] == right_value]
    merged = left.merge(right, on=key_cols, suffixes=("_left", "_right"))
    if merged.empty:
        return rows
    for metric in metrics:
        rows.append(
            paired_statistics(
                merged[f"{metric}_left"],
                merged[f"{metric}_right"],
                comparison=f"{left_value} vs {right_value}",
                metric=metric,
                left_name=left_value,
                right_name=right_value,
                context=context,
            )
        )
    return rows


def aggregate_mean_std(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join([col, stat]) for col, stat in grouped.columns]
    return grouped.reset_index()


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No data available._"
    visible = frame.head(max_rows).copy()
    for column in visible.columns:
        if pd.api.types.is_float_dtype(visible[column]):
            visible[column] = visible[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            visible[column] = visible[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(visible.columns) + " |"
    divider = "| " + " | ".join("---" for _ in visible.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in visible.to_numpy()]
    return "\n".join([header, divider, *rows])


def main() -> None:
    args = parse_args()
    post_dir = Path(args.postprocessing_dir)
    robustness_dir = Path(args.robustness_dir)
    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    post = read_csv_if_exists(post_dir / "selected_results.csv")
    robust = read_csv_if_exists(robustness_dir / "condition_results.csv")
    training = read_csv_if_exists(training_dir / "run_level_results.csv")

    stats_rows: list[dict[str, float | int | str]] = []

    if not post.empty:
        post = post.copy()
        post["seed"] = post["run_label"].map(seed_from_label)
        post_test = post[post["split"] == "test"].copy()
        stats_rows.extend(
            compare_by_key(
                post_test,
                key_cols=["seed"],
                group_col="variant",
                left_value="yolov8n",
                right_value="yolo11n",
                metrics=["mae", "rmse", "r2"],
                context="postprocessing_selected_test_by_seed",
            )
        )
    else:
        post_test = pd.DataFrame()

    if not robust.empty:
        robust = robust.copy()
        robust["seed"] = robust["run_label"].map(seed_from_label)
        for condition in sorted(robust["condition"].dropna().unique()):
            condition_frame = robust[robust["condition"] == condition]
            stats_rows.extend(
                compare_by_key(
                    condition_frame,
                    key_cols=["seed"],
                    group_col="variant",
                    left_value="yolov8n",
                    right_value="yolo11n",
                    metrics=["mae", "rmse", "r2"],
                    context=f"robustness_{condition}_by_seed",
                )
            )
        for variant in sorted(robust["variant"].dropna().unique()):
            variant_frame = robust[robust["variant"] == variant]
            for condition in sorted(set(variant_frame["condition"].dropna()) - {"original"}):
                pair = variant_frame[variant_frame["condition"].isin(["original", condition])]
                stats_rows.extend(
                    compare_by_key(
                        pair,
                        key_cols=["seed"],
                        group_col="condition",
                        left_value=condition,
                        right_value="original",
                        metrics=["mae"],
                        context=f"robustness_degradation_{variant}_by_seed",
                    )
                )
    if not training.empty:
        for case in sorted(training["case"].dropna().unique()):
            case_frame = training[training["case"] == case]
            stats_rows.extend(
                compare_by_key(
                    case_frame,
                    key_cols=["seed"],
                    group_col="variant",
                    left_value="yolov8n",
                    right_value="yolo11n",
                    metrics=["test_count_mae", "test_count_rmse", "test_count_r2", "test_detection_mAP50"],
                    context=f"training_{case}_by_seed",
                )
            )
        for variant in sorted(training["variant"].dropna().unique()):
            variant_frame = training[training["variant"] == variant]
            for case in ["default50", "minimal_aug20", "robust_aug20"]:
                pair = variant_frame[variant_frame["case"].isin([case, "default20"])]
                stats_rows.extend(
                    compare_by_key(
                        pair,
                        key_cols=["seed"],
                        group_col="case",
                        left_value=case,
                        right_value="default20",
                        metrics=["test_count_mae", "test_detection_mAP50"],
                        context=f"training_case_sensitivity_{variant}_by_seed",
                    )
                )

    stats_frame = pd.DataFrame(stats_rows)
    stats_path = output_dir / "statistical_tests.csv"
    stats_frame.to_csv(stats_path, index=False)

    post_agg = aggregate_mean_std(post_test, ["variant"], ["mae", "rmse", "r2", "exact_match_rate"])
    robust_agg = aggregate_mean_std(robust, ["condition", "variant"], ["mae", "rmse", "r2", "exact_match_rate"])
    train_agg = aggregate_mean_std(
        training,
        ["case", "variant"],
        ["test_count_mae", "test_count_rmse", "test_count_r2", "test_detection_mAP50", "test_detection_mAP50_95"],
    )
    if not train_agg.empty:
        train_agg = train_agg.sort_values(["test_count_mae_mean", "test_detection_mAP50_mean"], ascending=[True, False])

    summary_path = output_dir / "reviewer_experiment_results_summary.md"
    summary_text = "\n".join(
        [
            "# Reviewer Experiment Results Summary",
            "",
            "## Post-processing Sweep",
            markdown_table(post_agg),
            "",
            "## Robustness Sensitivity",
            markdown_table(robust_agg.sort_values(["condition", "mae_mean"]) if not robust_agg.empty else robust_agg, 40),
            "",
            "## Training Ablations",
            markdown_table(train_agg, 40),
            "",
            "## Paired Statistical Tests",
            markdown_table(stats_frame, 80),
            "",
            f"Statistical test CSV: `{stats_path}`",
        ]
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved {summary_path}")
    print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
