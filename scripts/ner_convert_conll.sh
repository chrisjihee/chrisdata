#!/bin/bash

# List of datasets
datasets=(
  "crossner_ai"
  "crossner_literature"
  "crossner_music"
  "crossner_politics"
  "crossner_science"
  "mit-movie"
  "mit-restaurant"
)

# List of splits
splits=("train" "test" "dev")

# Iterate over each dataset and split
for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    python -m chrisdata.cli ner convert_conll --split_name "$split" --verbose 1 "data/gner/each/$dataset" "data/gner/each/$dataset.jsonl"
  done
done
