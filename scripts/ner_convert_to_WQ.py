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

# List of splits
splits = ["train", "test", "dev"]

# Iterate over each dataset and split
for dataset in datasets:
    for split in splits:
        input_path_1 = f"data/gner/each/{dataset}-{split}.jsonl"
        output_path_1 = "data/gner/each-WQ"

        input_path_2 = f"data/gner/each-sampled/{dataset}-{split}=100.jsonl"
        output_path_2 = "data/gner/each-sampled-WQ"

        # Run the conversion commands
        command_1 = f"python -m chrisdata.cli ner convert_to_WQ {input_path_1} {output_path_1}".split()
        command_2 = f"python -m chrisdata.cli ner convert_to_WQ {input_path_2} {output_path_2}".split()

        subprocess.run(command_1)
        subprocess.run(command_2)
