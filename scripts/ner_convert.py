from chrisdata.ner.gner import *


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
    normalize_jsonl_file2("data/GNER/pile-ner.jsonl", instruction_file="conf/instruct/GNER-paper.txt")
    normalize_jsonl_file1("data/GNER/pile-ner.jsonl", instruction_file="conf/instruct/GNER-paper.txt")
    # normalize_conll_dirs("data/GNER/*")

    # make_dev_set()
    # make_test_set()
    # make_train_set()
