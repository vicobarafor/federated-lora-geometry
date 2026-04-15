import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/content/drive/MyDrive/fedgeox")
RAW_DIR = ROOT / "results" / "raw"
SUMMARY_DIR = ROOT / "results" / "summaries"
PLOT_DIR = ROOT / "results" / "plots"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

records = []

for exp_dir in sorted(RAW_DIR.glob("mrpc_final_*_c50_*")):
    if not exp_dir.is_dir():
        continue

    run_name = exp_dir.name
    subdirs = sorted([p for p in exp_dir.iterdir() if p.is_dir()])
    if not subdirs:
        continue

    latest = subdirs[-1]
    metrics_path = latest / "metrics.jsonl"
    if not metrics_path.exists():
        continue

    rows = load_jsonl(metrics_path)
    if not rows:
        continue

    df = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    if len(df) < 3:
        continue

    early = df[df["round"] < 3].copy()
    final = df.iloc[-1]

    method = "fedgeo" if "fedgeo" in run_name else "fedavg"
    dr = 0.9 if "dr9" in run_name or "dr90" in run_name else 0.7
    seed = int(run_name.split("seed")[-1])

    early_cos = pd.to_numeric(early["mean_pairwise_cosine"], errors="coerce")
    early_align = pd.to_numeric(early["mean_alignment_to_mean"], errors="coerce")

    records.append({
        "run_name": run_name,
        "method": method,
        "dr": dr,
        "seed": seed,
        "early_cos_mean": float(early_cos.mean()),
        "early_neg_cos_frac": float((early_cos < 0).mean()),
        "early_align_mean": float(early_align.mean()),
        "final_acc": float(pd.to_numeric(final["eval_accuracy"], errors="coerce")),
        "final_f1": float(pd.to_numeric(final["eval_f1"], errors="coerce")) if "eval_f1" in df.columns else np.nan,
        "is_bad_run": int(float(pd.to_numeric(final["eval_accuracy"], errors="coerce")) < 0.65),
    })

runs = pd.DataFrame(records).sort_values(["dr", "method", "seed"])
runs.to_csv(SUMMARY_DIR / "mrpc_c50_final_runs.csv", index=False)

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
        )
        .sort_values(["dr", "method"])
)

summary.to_csv(SUMMARY_DIR / "mrpc_c50_final_summary.csv", index=False)

print("=== MRPC c50 final summary ===")
print(summary.to_string(index=False))

plt.figure(figsize=(6, 4.5))
for method, marker in [("fedavg", "o"), ("fedgeo", "s")]:
    sub = summary[summary["method"] == method].sort_values("dr")
    if len(sub):
        plt.plot(sub["dr"], sub["mean_acc"], marker=marker, label=method)
plt.xlabel("dr")
plt.ylabel("mean final accuracy")
plt.title("MRPC 50 clients: accuracy vs heterogeneity")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "mrpc_c50_final_accuracy_vs_dr.png", dpi=220, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4.5))
for method, marker in [("fedavg", "o"), ("fedgeo", "s")]:
    sub = summary[summary["method"] == method].sort_values("dr")
    if len(sub):
        plt.plot(sub["dr"], sub["bad_run_rate"], marker=marker, label=method)
plt.xlabel("dr")
plt.ylabel("bad run rate")
plt.title("MRPC 50 clients: bad-run rate vs heterogeneity")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "mrpc_c50_final_bad_rate_vs_dr.png", dpi=220, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(" -", SUMMARY_DIR / "mrpc_c50_final_runs.csv")
print(" -", SUMMARY_DIR / "mrpc_c50_final_summary.csv")
print("Plots saved to:", PLOT_DIR)
