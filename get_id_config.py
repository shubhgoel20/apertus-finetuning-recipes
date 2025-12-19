import os
import re
import yaml
from collections import defaultdict

# Directory containing the yaml files
YAML_DIR = "/path/to/your/yaml_directory"

# Regex to extract id from filename: sft_lora_{id}.yaml
FILENAME_REGEX = re.compile(r"sft_lora_(.+)\.ya?ml$")

id_to_params = {}
rank_to_ids = defaultdict(list)

for filename in os.listdir(YAML_DIR):
    match = FILENAME_REGEX.match(filename)
    if not match:
        continue

    exp_id = match.group(1)
    filepath = os.path.join(YAML_DIR, filename)

    with open(filepath, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract required parameters
    params = {
        "learning_rate": cfg.get("learning_rate"),
        "num_train_epochs": cfg.get("num_train_epochs"),
        "lora_r": cfg.get("lora_r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "lora_dropout": cfg.get("lora_dropout"),
        "warmup_ratio": cfg.get("warmup_ratio"),
    }

    id_to_params[exp_id] = params

    # Build rank -> ids mapping
    lora_r = params["lora_r"]
    rank_to_ids[lora_r].append(exp_id)

# Optional: pretty-print results
print("ID -> Params mapping:")
for k, v in id_to_params.items():
    print(f"{k}: {v}")

print("\nRank -> IDs mapping:")
for rank, ids in rank_to_ids.items():
    print(f"{rank}: {ids}")