python -m chrisdata.cli ner convert_conll GNER/data/* GNER/data/no-pile-ner-train.jsonl --split-name train
python -m chrisdata.cli ner convert_conll GNER/data/* GNER/data/no-pile-ner-test.jsonl --split-name test
python -m chrisdata.cli ner convert_conll GNER/data/* GNER/data/no-pile-ner-dev.jsonl --split-name dev
