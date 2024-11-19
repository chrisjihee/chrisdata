python -m chrisdata.cli ner crawl_wiki GNER/data/pile-ner.json output/GNER/wiki_passage_from_pile.jsonl
python -m chrisdata.cli ner crawl_wiki GNER/data/zero-shot-train.jsonl output/GNER/wiki_passage_from_zero-train.jsonl
python -m chrisdata.cli ner crawl_wiki GNER/data/zero-shot-test.jsonl output/GNER/wiki_passage_from_zero-test.jsonl
python -m chrisdata.cli ner crawl_wiki GNER/data/zero-shot-dev.jsonl output/GNER/wiki_passage_from_zero-dev.jsonl
