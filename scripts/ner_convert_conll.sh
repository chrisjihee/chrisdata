#!/bin/bash

# 데이터셋 목록
datasets=(
  "crossner_ai"
  "crossner_literature"
  "crossner_music"
  "crossner_politics"
  "crossner_science"
  "mit-movie"
  "mit-restaurant"
)

# split 목록
splits=("train" "test" "dev")

# 반복 실행
for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    python -m chrisdata.cli ner convert_conll --split_name "$split" --verbose 1 "data/gner/each/$dataset" "data/gner/each/$dataset.jsonl"
  done
done
