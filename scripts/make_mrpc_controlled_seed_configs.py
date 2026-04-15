import os
import yaml
from copy import deepcopy

BASES = [
    ("configs/glue/mrpc_hetero_fedavg_dr70.yaml", "mrpc_control_fedavg_dr70"),
    ("configs/glue/mrpc_hetero_fedavg_dr90.yaml", "mrpc_control_fedavg_dr90"),
    ("configs/glue/mrpc_hetero_fedgeo_v5b_dr70.yaml", "mrpc_control_fedgeo_v5b_dr70"),
    ("configs/glue/mrpc_hetero_fedgeo_v5b_dr90.yaml", "mrpc_control_fedgeo_v5b_dr90"),
]

OUT_DIR = "configs/glue/generated_mrpc_controlled"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(8))  # 0..7 first; extend to 10 later if needed

written = []

for base_path, prefix in BASES:
    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)

    for seed in SEEDS:
        new_cfg = deepcopy(cfg)
        new_cfg["experiment_name"] = f"{prefix}_seed{seed}"

        if "runtime" not in new_cfg:
            new_cfg["runtime"] = {}
        new_cfg["runtime"]["seed"] = int(seed)

        out_path = os.path.join(OUT_DIR, f"{prefix}_seed{seed}.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False)

        written.append(out_path)

print(f"Wrote {len(written)} configs to {OUT_DIR}")
for p in written:
    print(p)
