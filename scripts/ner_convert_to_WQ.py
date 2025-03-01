import subprocess

target_label_levels = ["1", "3", "5"]
data_files = [
    "data/pile-ner.jsonl",
    "data/ZSE-test.jsonl",
    "data/ZSE-validation.jsonl",
]

# Iterate over each dataset and split
for label_levels in target_label_levels:
    for data_file in data_files:
        subprocess.run((f"python -m chrisdata.cli ner convert_to_WQ {data_file}"
                        f" --label_level_main {label_levels}").split())
