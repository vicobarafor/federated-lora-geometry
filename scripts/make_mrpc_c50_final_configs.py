import os
import yaml
from copy import deepcopy

BASES = [
    ("configs/glue/mrpc_scale_fedavg_c50.yaml", "mrpc_final_fedavg_c50"),
    ("configs/glue/mrpc_scale_fedgeo_v5b_c50.yaml", "mrpc_final_fedgeo_v5b_c50"),
]

OUT_DIR = "configs/glue/generated_mrpc_c50_final"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [0, 1, 2]
DRS = [0.7, 0.9]

written = []

for base_path, prefix in BASES:
    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)

    for dr in DRS:
        for seed in SEEDS:
            new_cfg = deepcopy(cfg)
            new_cfg["experiment_name"] = f"{prefix}_dr{int(dr*10)}_seed{seed}"

            new_cfg.setdefault("data", {})
            new_cfg["data"]["dominant_label_ratio"] = float(dr)

            new_cfg.setdefault("runtime", {})
            new_cfg["runtime"]["seed"] = int(seed)

            out_path = os.path.join(OUT_DIR, f"{prefix}_dr{int(dr*10)}_seed{seed}.yaml")
            with open(out_path, "w") as f:
                yaml.safe_dump(new_cfg, f, sort_keys=False)

            written.append(out_path)

print(f"Wrote {len(written)} configs:")
for p in written:
    print(p)
