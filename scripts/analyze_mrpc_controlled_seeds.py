import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/content/drive/MyDrive/fedgeox")
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_DIR = ROOT / "results" / "summaries"
PLOT_DIR = ROOT / "results" / "plots"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

records = []

# Scan the controlled MRPC experiment folders directly
for exp_dir in sorted(RAW_DIR.glob("mrpc_control_*")):
    if not exp_dir.is_dir():
        continue

    run_name = exp_dir.name

    # each experiment folder contains a timestamped subfolder
    subdirs = sorted([p for p in exp_dir.iterdir() if p.is_dir()])
    if not subdirs:
        continue

    # use the latest timestamped folder
    latest = subdirs[-1]
    metrics_path = latest / "metrics.jsonl"
    if not metrics_path.exists():
        continue

    rows = load_jsonl(metrics_path)
    if not rows:
        continue

    df = pd.DataFrame(rows)
    if "round" not in df.columns:
        continue

    df = df.sort_values("round").reset_index(drop=True)

    # need at least first 3 rounds for early metrics
    early = df[df["round"] < 3].copy()
    if len(early) < 2:
        continue

    final = df.iloc[-1]

    method = "fedgeo" if "fedgeo" in run_name else "fedavg"
    dr = 0.9 if "dr90" in run_name else (0.7 if "dr70" in run_name else np.nan)

    # parse seed from "...seedX"
    try:
        seed = int(run_name.split("seed")[-1])
    except Exception:
        seed = np.nan

    early_cos = pd.to_numeric(early["mean_pairwise_cosine"], errors="coerce")
    early_align = pd.to_numeric(early["mean_alignment_to_mean"], errors="coerce")

    rec = {
        "run_name": run_name,
        "metrics_path": str(metrics_path),
        "method": method,
        "dr": dr,
        "seed": seed,
        "early_cos_mean": float(early_cos.mean()),
        "early_cos_min": float(early_cos.min()),
        "early_neg_cos_frac": float((early_cos < 0).mean()),
        "early_align_mean": float(early_align.mean()),
        "final_acc": float(pd.to_numeric(final["eval_accuracy"], errors="coerce")),
        "final_loss": float(pd.to_numeric(final["eval_loss"], errors="coerce")) if "eval_loss" in df.columns else np.nan,
    }

    if "eval_f1" in df.columns:
        rec["final_f1"] = float(pd.to_numeric(final["eval_f1"], errors="coerce"))
    else:
        rec["final_f1"] = np.nan

    rec["is_bad_run"] = int(rec["final_acc"] < 0.65)
    records.append(rec)

runs = pd.DataFrame(records).sort_values(["dr", "method", "seed"]).reset_index(drop=True)
runs_path = SUMMARY_DIR / "mrpc_controlled_seed_runs.csv"
runs.to_csv(runs_path, index=False)

print("Collected runs:", len(runs))
print(runs[["run_name", "method", "dr", "seed", "final_acc"]].to_string(index=False))

summary = (
    runs.groupby(["dr", "method"], as_index=False)
        .agg(
            n=("run_name", "count"),
            mean_acc=("final_acc", "mean"),
            std_acc=("final_acc", "std"),
            mean_f1=("final_f1", "mean"),
            std_f1=("final_f1", "std"),
            bad_run_rate=("is_bad_run", "mean"),
            mean_early_cos=("early_cos_mean", "mean"),
            mean_early_neg_frac=("early_neg_cos_frac", "mean"),
            mean_early_align=("early_align_mean", "mean"),
        )
        .sort_values(["dr", "method"])
)

summary_path = SUMMARY_DIR / "mrpc_controlled_seed_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n=== Controlled MRPC Results ===")
print(summary.to_string(index=False))

corr_rows = []
for (method, dr), sub in runs.groupby(["method", "dr"]):
    if len(sub) < 3:
        continue
    for feat in ["early_cos_mean", "early_neg_cos_frac", "early_align_mean", "early_cos_min"]:
        for target in ["final_acc", "final_f1"]:
            pair = sub[[feat, target]].dropna()
            if len(pair) < 3:
                continue
            corr_rows.append({
                "method": method,
                "dr": dr,
                "feature": feat,
                "target": target,
                "pearson_r": float(pair[feat].corr(pair[target])),
                "n": int(len(pair))
            })

corr = pd.DataFrame(corr_rows).sort_values(["dr", "method", "target", "pearson_r"], ascending=[True, True, True, False])
corr_path = SUMMARY_DIR / "mrpc_controlled_seed_correlations.csv"
corr.to_csv(corr_path, index=False)

print("\n=== Controlled Correlations ===")
if len(corr):
    print(corr.to_string(index=False))
else:
    print("No valid correlations.")

# Plot 1: accuracy vs heterogeneity
if len(summary):
    plt.figure(figsize=(6, 4.5))
    for method, marker in [("fedavg", "o"), ("fedgeo", "s")]:
        sub = summary[summary["method"] == method].sort_values("dr")
        if len(sub) == 0:
            continue
        plt.plot(sub["dr"], sub["mean_acc"], marker=marker, label=method)
    plt.xlabel("dr")
    plt.ylabel("mean final accuracy")
    plt.title("MRPC controlled seeds: accuracy vs heterogeneity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mrpc_controlled_accuracy_vs_dr.png", dpi=220, bbox_inches="tight")
    plt.close()

    # Plot 2: bad run rate vs heterogeneity
    plt.figure(figsize=(6, 4.5))
    for method, marker in [("fedavg", "o"), ("fedgeo", "s")]:
        sub = summary[summary["method"] == method].sort_values("dr")
        if len(sub) == 0:
            continue
        plt.plot(sub["dr"], sub["bad_run_rate"], marker=marker, label=method)
    plt.xlabel("dr")
    plt.ylabel("bad run rate")
    plt.title("MRPC controlled seeds: bad-run rate vs heterogeneity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mrpc_controlled_bad_rate_vs_dr.png", dpi=220, bbox_inches="tight")
    plt.close()

# Plot 3: early cosine vs final accuracy
if len(runs):
    plt.figure(figsize=(6, 5))
    for method, marker in [("fedavg", "o"), ("fedgeo", "s")]:
        for dr in sorted(runs["dr"].dropna().unique()):
            sub = runs[(runs["method"] == method) & (runs["dr"] == dr)]
            if len(sub) == 0:
                continue
            plt.scatter(
                sub["early_cos_mean"],
                sub["final_acc"],
                marker=marker,
                alpha=0.8,
                label=f"{method}, dr={dr}"
            )
    plt.xlabel("early_cos_mean")
    plt.ylabel("final_acc")
    plt.title("MRPC controlled seeds: early cosine vs final accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mrpc_controlled_earlycos_vs_acc.png", dpi=220, bbox_inches="tight")
    plt.close()

print("\nSaved:")
print(" -", runs_path)
print(" -", summary_path)
print(" -", corr_path)
print("Plots saved to:", PLOT_DIR)
