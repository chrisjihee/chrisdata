"""
This script merges the .conll files located under:
    AnatEM-1.0.2/conll/train
    AnatEM-1.0.2/conll/devel
    AnatEM-1.0.2/conll/test

into three files:
    train.txt, dev.txt, test.txt

It also creates a label.txt file containing all unique labels from
those three splits.

Additionally, it prints out the number of "samples" in each split.
Here, we define a "sample" as one sentence (a block of non-blank lines
separated by blank lines in CoNLL format).

Usage:
    python anatem.py
"""

import os
import re
from pathlib import Path


def merge_conll_files(input_dir, output_file, label_set):
    """
    Merge all *.conll files from 'input_dir' into a single file 'output_file'.
    Collect labels into 'label_set'.

    :param input_dir: Directory containing one or more .conll files.
    :param output_file: Output file path to write the merged CoNLL data.
    :param label_set: A set() object that will be updated with any labels seen.
    :return: Number of samples (sentences) in the merged data.
    """
    conll_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".conll") and not f.startswith("._")
    ]
    conll_files.sort()

    num_samples = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in conll_files:
            file_path = os.path.join(input_dir, filename)

            with open(file_path, "r", encoding="utf-8") as in_f:
                current_sentence_lines = []

                for line in in_f:
                    line = line.rstrip("\n")

                    if line.strip() == "":
                        # Blank line => end of a sentence
                        if current_sentence_lines:
                            # We have a completed sentence
                            num_samples += 1
                            # Write that sentence (plus a blank line)
                            for sent_line in current_sentence_lines:
                                out_f.write(f"{sent_line}\n")
                            out_f.write("\n")
                            current_sentence_lines = []
                    else:
                        # Not a blank line => token/label
                        # Typically in CoNLL format: token \t label (or space separated)
                        parts = line.split()

                        # If the file is a standard 2-column CoNLL: token label
                        # (some CoNLL variants have more columns; adjust as needed)
                        if len(parts) >= 2:
                            token = parts[0]
                            label = parts[-1]  # usually the last column is the label
                            label_set.add(label)

                            current_sentence_lines.append(line)
                        else:
                            # If there's only one column or no columns, just keep it as is
                            # (But typically you won't see this in normal CoNLL data)
                            current_sentence_lines.append(line)

                # If file doesn't end with a blank line, we might still have a final sentence
                if current_sentence_lines:
                    num_samples += 1
                    for sent_line in current_sentence_lines:
                        out_f.write(f"{sent_line}\n")
                    out_f.write("\n")

    return num_samples


def main():
    input_dir = Path("data/AnatEM/conll")
    output_dir = Path("data/AnatEM")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_set = set()
    train_samples = merge_conll_files(input_dir / "train", output_dir / "train.txt", label_set)
    eval_samples = merge_conll_files(input_dir / "dev", output_dir / "dev.txt", label_set)
    test_samples = merge_conll_files(input_dir / "test", output_dir / "test.txt", label_set)

    class_names = []
    for label_name in sorted(label_set):
        class_name = re.sub(r"^[BIES]-|^O$", "", label_name)
        if class_name and class_name not in class_names:
            class_names.append(class_name)

    (output_dir / "label.txt").write_text("\n".join(class_names) + "\n")
    print(f"  # train : {train_samples}")
    print(f"  # dev   : {eval_samples}")
    print(f"  # test  : {test_samples}")
    print(f"  # label : {len(label_set)}")


if __name__ == "__main__":
    main()
