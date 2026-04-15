import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/content/drive/MyDrive/fedgeox"
RAW_DIR = os.path.join(ROOT, "results/raw")
PLOT_DIR = os.path.join(ROOT, "results/plots")
SUMMARY_DIR = os.path.join(ROOT, "results/summaries")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def infer_method(run_name):
    s = run_name.lower()
    if "fedprox" in s:
        return "fedprox"
    if "geo" in s:
        return "fedgeo"
    return "fedavg"

def infer_task(run_name):
    s = run_name.lower()
    if "mrpc" in s:
        return "mrpc"
    if "sst2" in s or "sst-2" in s:
        return "sst2"
    if "qqp" in s:
        return "qqp"
    return "unknown"

def infer_dr(run_name):
    s = run_name.lower()
    if "dr90" in s:
        return 0.9
    if "dr80" in s:
        return 0.8
    if "dr70" in s:
        return 0.7
    if "dr60" in s:
        return 0.6
    return np.nan

records = []

for run_name in sorted(os.listdir(RAW_DIR)):
    run_dir = os.path.join(RAW_DIR, run_name)
    if not os.path.isdir(run_dir):
        continue

    for sub in sorted(os.listdir(run_dir)):
        metrics_path = os.path.join(run_dir, sub, "metrics.jsonl")
        if not os.path.exists(metrics_path):
            continue

        rows = load_jsonl(metrics_path)
        if not rows:
            continue

        df = pd.DataFrame(rows)
        if "round" not in df.columns:
            continue
        if "mean_pairwise_cosine" not in df.columns:
            continue
        if "mean_alignment_to_mean" not in df.columns:
            continue
        if "eval_accuracy" not in df.columns:
            continue

        df = df.sort_values("round").reset_index(drop=True)

        early = df[df["round"] < 3].copy()
        if len(early) < 2:
            continue

        final_row = df.iloc[-1]

        early_cos = pd.to_numeric(early["mean_pairwise_cosine"], errors="coerce").dropna()
        early_align = pd.to_numeric(early["mean_alignment_to_mean"], errors="coerce").dropna()

        if len(early_cos) < 2 or len(early_align) < 2:
            continue

        record = {
            "run_name": run_name,
            "timestamp": sub,
            "method": infer_method(run_name),
            "task": infer_task(run_name),
            "dr": infer_dr(run_name),

            "early_cos_mean": float(early_cos.mean()),
            "early_cos_min": float(early_cos.min()),
            "early_cos_delta": float(early_cos.iloc[-1] - early_cos.iloc[0]),
            "early_neg_cos_frac": float((early_cos < 0).mean()),

            "early_align_mean": float(early_align.mean()),
            "early_align_min": float(early_align.min()),
            "early_align_delta": float(early_align.iloc[-1] - early_align.iloc[0]),

            "final_acc": float(pd.to_numeric(final_row["eval_accuracy"], errors="coerce")),
            "final_loss": float(pd.to_numeric(final_row["eval_loss"], errors="coerce")) if "eval_loss" in df.columns else np.nan,
        }

        if "eval_f1" in df.columns:
            record["final_f1"] = float(pd.to_numeric(final_row["eval_f1"], errors="coerce"))
        else:
            record["final_f1"] = np.nan

        records.append(record)

runs_df = pd.DataFrame(records)
runs_path = os.path.join(SUMMARY_DIR, "early_predictor_runs.csv")
runs_df.to_csv(runs_path, index=False)

print(f"Collected {len(runs_df)} runs")
print(f"Saved run-level summary to: {runs_path}")

features = [
    "early_cos_mean",
    "early_cos_min",
    "early_cos_delta",
    "early_neg_cos_frac",
    "early_align_mean",
    "early_align_min",
    "early_align_delta",
]

targets = ["final_acc", "final_f1"]

corr_rows = []
for feat in features:
    for targ in targets:
        sub = runs_df[[feat, targ]].dropna()
        if len(sub) < 3:
            continue
        r = sub[feat].corr(sub[targ])
        corr_rows.append({
            "feature": feat,
            "target": targ,
            "pearson_r": float(r),
            "n": int(len(sub))
        })

corr_df = pd.DataFrame(corr_rows)

if len(corr_df) == 0:
    print("No valid correlations found.")
else:
    corr_df = corr_df.sort_values(["target", "pearson_r"], ascending=[True, False])
    corr_path = os.path.join(SUMMARY_DIR, "early_predictor_correlations.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved correlations to: {corr_path}")
    print(corr_df)

# Scatter plots
plot_specs = [
    ("early_cos_mean", "final_acc"),
    ("early_neg_cos_frac", "final_acc"),
    ("early_align_mean", "final_acc"),
    ("early_align_delta", "final_acc"),
]

if "final_f1" in runs_df.columns and runs_df["final_f1"].notna().sum() >= 3:
    plot_specs.extend([
        ("early_cos_mean", "final_f1"),
        ("early_neg_cos_frac", "final_f1"),
        ("early_align_mean", "final_f1"),
    ])

markers = {
    "fedavg": "o",
    "fedgeo": "s",
    "fedprox": "^"
}

for x, y in plot_specs:
    sub = runs_df[[x, y, "method", "task"]].dropna()
    if len(sub) < 3:
        continue

    plt.figure(figsize=(6, 5))
    for method in sorted(sub["method"].unique()):
        g = sub[sub["method"] == method]
        plt.scatter(g[x], g[y], label=method, marker=markers.get(method, "o"), alpha=0.8)

    r = sub[x].corr(sub[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y} (r={r:.3f}, n={len(sub)})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{x}_vs_{y}.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")

print("Done.")
