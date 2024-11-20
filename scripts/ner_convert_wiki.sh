#python -m chrisdata.cli ner convert_wiki output/GNER/wiki_passage_from_pile.jsonl \
#                                         GNER/data/linked-entity-pile.jsonl --split-name train

python -m chrisdata.cli ner convert_wiki output/GNER/wiki_passage_from_zero-train.jsonl \
                                         GNER/data/linked-entity-zero-train.jsonl --split-name train

python -m chrisdata.cli ner convert_wiki output/GNER/wiki_passage_from_zero-test.jsonl \
                                         GNER/data/linked-entity-zero-test.jsonl --split-name test

python -m chrisdata.cli ner convert_wiki output/GNER/wiki_passage_from_zero-dev.jsonl \
                                         GNER/data/linked-entity-zero-dev.jsonl --split-name validation
