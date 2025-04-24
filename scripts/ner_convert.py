from chrisdata.ner.gner import normalize_conll_dirs, convert_to_hybrid_round_cot_version, normalize_jsonl_file


def make_dev_set():
    # convert_to_hybrid_round_cot_version("data/GoLLIE/baseline/ace05.ner.dev.jsonl")
    # convert_to_hybrid_round_cot_version("data/GoLLIE/baseline/conll03.ner.dev.jsonl")
    # convert_to_hybrid_round_cot_version("data/GoLLIE/processed/crossner.crossner_ai.dev.jsonl")
    convert_to_hybrid_round_cot_version("data/GoLLIE/processed_w_examples/crossner.crossner_ai.dev.jsonl")


def make_test_set():
    pass


def make_train_set():
    pass


if __name__ == "__main__":
    normalize_jsonl_file("data/GNER/pile-ner.jsonl")
    # normalize_conll_dirs("data/GNER/*")

    # make_dev_set()
    # make_test_set()
    # make_train_set()
