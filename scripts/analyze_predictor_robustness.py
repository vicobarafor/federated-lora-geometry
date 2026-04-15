import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/content/drive/MyDrive/fedgeox"
SUMMARY_DIR = os.path.join(ROOT, "results", "summaries")
PLOT_DIR = os.path.join(ROOT, "results", "plots")

os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RUNS_PATH = os.path.join(SUMMARY_DIR, "early_predictor_runs.csv")
runs = pd.read_csv(RUNS_PATH)

# Keep only rows with the core variables we need
core = runs.copy()
for c in ["task", "method", "early_cos_mean", "early_neg_cos_frac", "final_acc"]:
    if c not in core.columns:
        raise ValueError(f"Missing required column: {c}")

# ----------------------------
# 1) Per-task correlations
# ----------------------------
task_rows = []
for task in sorted(core["task"].dropna().unique()):
    sub = core[core["task"] == task].copy()

    for feat in ["early_cos_mean", "early_neg_cos_frac", "early_align_mean", "early_cos_min"]:
        if feat not in sub.columns:
            continue
        pair = sub[[feat, "final_acc"]].dropna()
        if len(pair) < 3:
            continue
        r = pair[feat].corr(pair["final_acc"])
        task_rows.append({
            "group_type": "task",
            "group": task,
            "feature": feat,
            "target": "final_acc",
            "pearson_r": float(r),
            "n": int(len(pair))
        })

        if "final_f1" in sub.columns:
            pair_f1 = sub[[feat, "final_f1"]].dropna()
            if len(pair_f1) >= 3:
                r_f1 = pair_f1[feat].corr(pair_f1["final_f1"])
                task_rows.append({
                    "group_type": "task",
                    "group": task,
                    "feature": feat,
                    "target": "final_f1",
                    "pearson_r": float(r_f1),
                    "n": int(len(pair_f1))
                })

task_corr = pd.DataFrame(task_rows)
task_corr_path = os.path.join(SUMMARY_DIR, "predictor_per_task_correlations.csv")
task_corr.to_csv(task_corr_path, index=False)

# ----------------------------
# 2) Per-method correlations
# ----------------------------
method_rows = []
for method in sorted(core["method"].dropna().unique()):
    sub = core[core["method"] == method].copy()

    for feat in ["early_cos_mean", "early_neg_cos_frac", "early_align_mean", "early_cos_min"]:
        if feat not in sub.columns:
            continue
        pair = sub[[feat, "final_acc"]].dropna()
        if len(pair) < 3:
            continue
        r = pair[feat].corr(pair["final_acc"])
        method_rows.append({
            "group_type": "method",
            "group": method,
            "feature": feat,
            "target": "final_acc",
            "pearson_r": float(r),
            "n": int(len(pair))
        })

        if "final_f1" in sub.columns:
            pair_f1 = sub[[feat, "final_f1"]].dropna()
            if len(pair_f1) >= 3:
                r_f1 = pair_f1[feat].corr(pair_f1["final_f1"])
                method_rows.append({
                    "group_type": "method",
                    "group": method,
                    "feature": feat,
                    "target": "final_f1",
                    "pearson_r": float(r_f1),
                    "n": int(len(pair_f1))
                })

method_corr = pd.DataFrame(method_rows)
method_corr_path = os.path.join(SUMMARY_DIR, "predictor_per_method_correlations.csv")
method_corr.to_csv(method_corr_path, index=False)

# ----------------------------
# 3) Good vs bad separation
# ----------------------------
# Task-specific thresholds: chosen to reflect meaningful degradation for your tasks.
def classify_bad_run(row):
    task = str(row["task"]).lower()
    acc = row["final_acc"]
    if pd.isna(acc):
        return np.nan
    if task == "mrpc":
        return int(acc < 0.65)
    if task == "sst2":
        return int(acc < 0.82)
    if task == "qqp":
        return int(acc < 0.75)
    return np.nan

sep = core.copy()
sep["is_bad_run"] = sep.apply(classify_bad_run, axis=1)

sep_rows = []
for task in sorted(sep["task"].dropna().unique()):
    sub = sep[(sep["task"] == task) & (sep["is_bad_run"].notna())].copy()
    if len(sub) < 6:
        continue

    for feat in ["early_cos_mean", "early_neg_cos_frac", "early_align_mean"]:
        if feat not in sub.columns:
            continue
        pair = sub[[feat, "is_bad_run"]].dropna().copy()
        if len(pair) < 6:
            continue

        # Median split prediction
        median = pair[feat].median()
        if feat == "early_neg_cos_frac":
            pred_bad = pair[feat] > median
        else:
            pred_bad = pair[feat] < median

        pred_acc = float((pred_bad.astype(int) == pair["is_bad_run"].astype(int)).mean())

        good_mean = float(pair.loc[pair["is_bad_run"] == 0, feat].mean()) if (pair["is_bad_run"] == 0).any() else np.nan
        bad_mean = float(pair.loc[pair["is_bad_run"] == 1, feat].mean()) if (pair["is_bad_run"] == 1).any() else np.nan

        sep_rows.append({
            "task": task,
            "feature": feat,
            "n": int(len(pair)),
            "median_threshold": float(median),
            "threshold_prediction_accuracy": pred_acc,
            "good_mean": good_mean,
            "bad_mean": bad_mean,
            "direction": "higher_is_worse" if feat == "early_neg_cos_frac" else "lower_is_worse"
        })

sep_df = pd.DataFrame(sep_rows)
sep_path = os.path.join(SUMMARY_DIR, "predictor_good_bad_separation.csv")
sep_df.to_csv(sep_path, index=False)

# ----------------------------
# 4) Threshold plots by heterogeneity
# ----------------------------
# Uses rows with explicit dr values and focuses on methods central to the paper.
thr = core.copy()
thr = thr[thr["dr"].notna()].copy()
thr = thr[thr["method"].isin(["fedavg", "fedgeo"])].copy()
thr = thr[thr["task"].isin(["mrpc", "sst2"])].copy()

if len(thr) > 0:
    grouped = (
        thr.groupby(["task", "method", "dr"], as_index=False)
           .agg(
               early_cos_mean_mean=("early_cos_mean", "mean"),
               early_neg_cos_frac_mean=("early_neg_cos_frac", "mean"),
               final_acc_mean=("final_acc", "mean"),
               count=("run_name", "count") if "run_name" in thr.columns else ("method", "count")
           )
    )
    grouped_path = os.path.join(SUMMARY_DIR, "predictor_threshold_by_dr.csv")
    grouped.to_csv(grouped_path, index=False)

    for task in sorted(grouped["task"].unique()):
        sub = grouped[grouped["task"] == task].copy()
        if len(sub) == 0:
            continue

        # early_cos_mean vs dr
        plt.figure(figsize=(6, 4.5))
        for method in sorted(sub["method"].unique()):
            g = sub[sub["method"] == method].sort_values("dr")
            plt.plot(g["dr"], g["early_cos_mean_mean"], marker="o", label=method)
        plt.xlabel("dominant label ratio (dr)")
        plt.ylabel("early_cos_mean")
        plt.title(f"{task.upper()}: early cosine vs heterogeneity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{task}_threshold_early_cos_vs_dr.png"), dpi=220, bbox_inches="tight")
        plt.close()

        # early_neg_cos_frac vs dr
        plt.figure(figsize=(6, 4.5))
        for method in sorted(sub["method"].unique()):
            g = sub[sub["method"] == method].sort_values("dr")
            plt.plot(g["dr"], g["early_neg_cos_frac_mean"], marker="o", label=method)
        plt.xlabel("dominant label ratio (dr)")
        plt.ylabel("early_neg_cos_frac")
        plt.title(f"{task.upper()}: early negative-cosine fraction vs heterogeneity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{task}_threshold_early_negfrac_vs_dr.png"), dpi=220, bbox_inches="tight")
        plt.close()

        # final_acc vs dr
        plt.figure(figsize=(6, 4.5))
        for method in sorted(sub["method"].unique()):
            g = sub[sub["method"] == method].sort_values("dr")
            plt.plot(g["dr"], g["final_acc_mean"], marker="o", label=method)
        plt.xlabel("dominant label ratio (dr)")
        plt.ylabel("final_acc")
        plt.title(f"{task.upper()}: final accuracy vs heterogeneity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{task}_threshold_finalacc_vs_dr.png"), dpi=220, bbox_inches="tight")
        plt.close()

# ----------------------------
# 5) Histograms for separation
# ----------------------------
for task in sorted(sep["task"].dropna().unique()):
    sub = sep[(sep["task"] == task) & (sep["is_bad_run"].notna())].copy()
    if len(sub) < 6:
        continue

    for feat in ["early_cos_mean", "early_neg_cos_frac"]:
        if feat not in sub.columns:
            continue

        pair = sub[[feat, "is_bad_run"]].dropna()
        if len(pair) < 6:
            continue

        good = pair[pair["is_bad_run"] == 0][feat]
        bad = pair[pair["is_bad_run"] == 1][feat]
        if len(good) == 0 or len(bad) == 0:
            continue

        plt.figure(figsize=(6, 4.5))
        plt.hist(good, bins=12, alpha=0.5, label="good")
        plt.hist(bad, bins=12, alpha=0.5, label="bad")
        plt.xlabel(feat)
        plt.ylabel("count")
        plt.title(f"{task.upper()}: good vs bad run separation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{task}_{feat}_good_bad_hist.png"), dpi=220, bbox_inches="tight")
        plt.close()

print("Saved:")
print(" -", task_corr_path)
print(" -", method_corr_path)
print(" -", sep_path)
if 'grouped_path' in locals():
    print(" -", grouped_path)
print("Plots saved to:", PLOT_DIR)
print("\nTop per-task correlations:")
print(task_corr.sort_values(["target", "pearson_r"], ascending=[True, False]).head(12))
print("\nTop per-method correlations:")
print(method_corr.sort_values(["target", "pearson_r"], ascending=[True, False]).head(12))
print("\nGood/bad separation:")
print(sep_df.sort_values(["task", "threshold_prediction_accuracy"], ascending=[True, False]))
