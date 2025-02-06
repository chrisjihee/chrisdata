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

# Iterate over each dataset and split
for dataset in datasets:
    for split in ["train", "test", "dev"]:
        input_path_1 = f"data/gner/each/{dataset}-{split}.jsonl"
        input_path_2 = f"data/gner/each-sampled/{dataset}-{split}=100.jsonl"

        # Run the conversion commands
        for label_level_main in [1, 2, 3, 4, 5]:
            if label_level_main != 4:
                subprocess.run((f"python -m chrisdata.cli ner convert_to_WQ {input_path_1}"
                                f" --label_level_main {label_level_main}").split())
                subprocess.run((f"python -m chrisdata.cli ner convert_to_WQ {input_path_2}"
                                f" --label_level_main {label_level_main}").split())
            else:
                for label_level_sub in [1, 2, 3]:
                    subprocess.run((f"python -m chrisdata.cli ner convert_to_WQ {input_path_1}"
                                    f" --label_level_main {label_level_main} --label_level_sub {label_level_sub}").split())
                    subprocess.run((f"python -m chrisdata.cli ner convert_to_WQ {input_path_2}"
                                    f" --label_level_main {label_level_main} --label_level_sub {label_level_sub}").split())
