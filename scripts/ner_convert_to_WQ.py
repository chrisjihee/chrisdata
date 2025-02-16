import subprocess

# List of datasets
datasets = [
    "crossner_ai",
    "crossner_literature",
    "crossner_music",
    "crossner_politics",
    "crossner_science",
    "mit-movie",
    "mit-restaurant",
]
target_label_levels = ["1", "3", "5"]

# Iterate over each dataset and split
for dataset in datasets:
    for label_levels in target_label_levels:
        for split in ["train", "test", "dev"]:
            input_path_1 = f"data/gner/each-BL/{dataset}-{split}.jsonl"
            input_path_2 = f"data/gner/each-sampled-BL/{dataset}-{split}=100.jsonl"
            subprocess.run((f"python -m chrisdata.cli ner convert_to_WQ {input_path_1}"
                            f" --label_level_main {label_levels}").split())
            subprocess.run((f"python -m chrisdata.cli ner convert_to_WQ {input_path_2}"
                            f" --label_level_main {label_levels}").split())

for label_levels in target_label_levels:
    for input_path in [
        "data/gner/united/pile-ner.jsonl",
        "data/gner/united/zero-shot-dev.jsonl",
        "data/gner/united/zero-shot-dev-100.jsonl",
        "data/gner/united/zero-shot-dev-200.jsonl",
        "data/gner/united/zero-shot-test.jsonl",
        "data/gner/united/zero-shot-test-100.jsonl",
        "data/gner/united/zero-shot-test-200.jsonl",
    ]:
        subprocess.run((f"python -m chrisdata.cli ner convert_to_WQ {input_path}"
                        f" --label_level_main {label_levels}").split())
