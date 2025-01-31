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
    python -m chrisdata.cli ner sample_jsonl "data/gner/each/$dataset-$split.jsonl" --num_samples 100
  done
done
