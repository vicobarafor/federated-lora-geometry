import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/content/drive/MyDrive/fedgeox"
SUMMARY_DIR = os.path.join(ROOT, "results", "summaries")
PLOT_DIR = os.path.join(ROOT, "results", "plots")

runs = pd.read_csv(os.path.join(SUMMARY_DIR, "early_predictor_runs.csv"))

# Keep only core paper tasks/methods
runs = runs[runs["task"].isin(["mrpc", "sst2"])].copy()
runs = runs[runs["method"].isin(["fedavg", "fedgeo"])].copy()

name = runs["run_name"].astype(str).str.lower()

# Exclude exploratory / old variant runs
exclude_terms = [
    "smoke", "pilot", "soft", "diverse", "bounded", "eps_", "v3", "v4",
    "fedprox", "seed42", "seed123", "seed999"
]
mask_excl = pd.Series(False, index=runs.index)
for t in exclude_terms:
    mask_excl = mask_excl | name.str.contains(t, regex=False, na=False)

runs = runs.loc[~mask_excl].copy()

# Recompute after filtering
name = runs["run_name"].astype(str).str.lower()

# Keep only paper-relevant runs
keep_terms = [
    "hetero_fedavg", "hetero_fedgeo", "hetero_fedgeo_v5b",
    "dr60", "dr70", "dr80", "dr90",
    "seed_",
    "clients20", "clients50", "scale"
]
mask_keep = pd.Series(False, index=runs.index)
for t in keep_terms:
    mask_keep = mask_keep | name.str.contains(t, regex=False, na=False)

# Also keep simple baseline names if present
baseline_patterns = [
    r"^mrpc_(fedavg|fedgeo|fedgeo_v5b)$",
    r"^sst2_(fedavg|fedgeo|fedgeo_v5b)$",
    r"^mrpc_hetero_fedavg$",
    r"^mrpc_hetero_fedgeo.*$",
    r"^sst2_hetero_fedavg$",
    r"^sst2_hetero_fedgeo.*$",
]
for pat in baseline_patterns:
    mask_keep = mask_keep | name.str.match(pat, na=False)

runs = runs.loc[mask_keep].copy()

runs.to_csv(os.path.join(SUMMARY_DIR, "core_subset_runs.csv"), index=False)

print("Core subset size:", len(runs))
if len(runs) == 0:
    raise ValueError("Core subset is empty. Check filtering rules.")
print(runs[["run_name", "task", "method", "dr"]].sort_values(["task", "method", "run_name"]).to_string(index=False))

features = ["early_cos_mean", "early_neg_cos_frac", "early_align_mean", "early_cos_min"]

rows = []
for task in sorted(runs["task"].unique()):
    for method in sorted(runs["method"].unique()):
        sub = runs[(runs["task"] == task) & (runs["method"] == method)].copy()
        for feat in features:
            pair = sub[[feat, "final_acc"]].dropna()
            if len(pair) < 3:
                continue
            rows.append({
                "task": task,
                "method": method,
                "feature": feat,
                "target": "final_acc",
                "pearson_r": float(pair[feat].corr(pair["final_acc"])),
                "n": int(len(pair))
            })
            if "final_f1" in sub.columns:
                pairf = sub[[feat, "final_f1"]].dropna()
                if len(pairf) >= 3:
                    rows.append({
                        "task": task,
                        "method": method,
                        "feature": feat,
                        "target": "final_f1",
                        "pearson_r": float(pairf[feat].corr(pairf["final_f1"])),
                        "n": int(len(pairf))
                    })

corr = pd.DataFrame(rows)
corr_path = os.path.join(SUMMARY_DIR, "core_subset_correlations.csv")
corr.to_csv(corr_path, index=False)

print("\nCore subset correlations:")
if len(corr):
    print(corr.sort_values(["task", "method", "target", "pearson_r"], ascending=[True, True, True, False]).to_string(index=False))
else:
    print("No valid correlations found.")

for task in sorted(runs["task"].unique()):
    for method in sorted(runs["method"].unique()):
        sub = runs[(runs["task"] == task) & (runs["method"] == method)].copy()
        if len(sub) < 3:
            continue

        for x, y in [("early_cos_mean", "final_acc"), ("early_neg_cos_frac", "final_acc")]:
            pair = sub[[x, y, "run_name"]].dropna()
            if len(pair) < 3:
                continue
            r = pair[x].corr(pair[y])

            plt.figure(figsize=(5.8, 4.5))
            plt.scatter(pair[x], pair[y], alpha=0.85)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{task.upper()} / {method}: {x} vs {y} (r={r:.3f}, n={len(pair)})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out = os.path.join(PLOT_DIR, f"{task}_{method}_{x}_vs_{y}_core.png")
            plt.savefig(out, dpi=220, bbox_inches="tight")
            plt.close()

thr = runs[runs["dr"].notna()].copy()
if len(thr):
    grouped = (
        thr.groupby(["task", "method", "dr"], as_index=False)
           .agg(
               early_cos_mean_mean=("early_cos_mean", "mean"),
               early_neg_cos_frac_mean=("early_neg_cos_frac", "mean"),
               final_acc_mean=("final_acc", "mean"),
               count=("run_name", "count")
           )
    )
    grouped_path = os.path.join(SUMMARY_DIR, "core_subset_threshold_curves.csv")
    grouped.to_csv(grouped_path, index=False)

    for task in sorted(grouped["task"].unique()):
        sub = grouped[grouped["task"] == task].copy()
        if len(sub) == 0:
            continue

        plt.figure(figsize=(6, 4.5))
        for method in sorted(sub["method"].unique()):
            g = sub[sub["method"] == method].sort_values("dr")
            plt.plot(g["dr"], g["early_cos_mean_mean"], marker="o", label=method)
        plt.xlabel("dr")
        plt.ylabel("early_cos_mean")
        plt.title(f"{task.upper()}: early cosine threshold curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{task}_core_threshold_early_cos.png"), dpi=220, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4.5))
        for method in sorted(sub["method"].unique()):
            g = sub[sub["method"] == method].sort_values("dr")
            plt.plot(g["dr"], g["final_acc_mean"], marker="o", label=method)
        plt.xlabel("dr")
        plt.ylabel("final_acc")
        plt.title(f"{task.upper()}: performance threshold curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{task}_core_threshold_final_acc.png"), dpi=220, bbox_inches="tight")
        plt.close()

print("\nSaved:")
print(" -", corr_path)
if 'grouped_path' in locals():
    print(" -", grouped_path)
print("Plots saved to:", PLOT_DIR)
