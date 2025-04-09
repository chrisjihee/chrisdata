from chrisdata.ner.gner import convert_to_hybrid_round_cot_version


def make_dev_set():
    convert_to_hybrid_round_cot_version("data/GoLLIE/baseline/ace05.ner.dev.jsonl")


def make_test_set():
    pass


def make_train_set():
    pass


if __name__ == "__main__":
    make_dev_set()
    # make_test_set()
    # make_train_set()
