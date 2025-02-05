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
    python -m chrisdata.cli ner convert_to_WQ "data/gner/each/$dataset-$split.jsonl" "data/gner/each-WQ"
    python -m chrisdata.cli ner convert_to_WQ "data/gner/each-sampled/$dataset-$split=100.jsonl" "data/gner/each-sampled-WQ"
  done
done
