from chrisdata.ner.gner import stratified_sample_jsonl, convert_to_hybrid_round_version


# DEV SET
def make_dev_set():
    dev_sampled_70 = stratified_sample_jsonl(
        input_file="data/GNER/ZSE-dev.jsonl",
        output_file="data/GNER/ZSE-dev-sampled.jsonl",
        max_num_samples=10,
    )  # 70
    dev_sampled_210 = stratified_sample_jsonl(
        input_file="data/GNER/ZSE-dev.jsonl",
        output_file="data/GNER/ZSE-dev-sampled.jsonl",
        max_num_samples=30,
    )  # 210
    dev_sampled_700 = stratified_sample_jsonl(
        input_file="data/GNER/ZSE-dev.jsonl",
        output_file="data/GNER/ZSE-dev-sampled.jsonl",
        max_num_samples=100,
    )  # 700
    dev_sampled_1400 = stratified_sample_jsonl(
        input_file="data/GNER/ZSE-dev.jsonl",
        output_file="data/GNER/ZSE-dev-sampled.jsonl",
        max_num_samples=200,
    )  # 1,400
    convert_to_hybrid_round_version(
        mr_input_file=dev_sampled_70,
        sr_input_file=dev_sampled_70,
    )  # 870
    convert_to_hybrid_round_version(
        mr_input_file=dev_sampled_70,
        sr_input_file=dev_sampled_700,
    )  # 1,500
    convert_to_hybrid_round_version(
        mr_input_file=dev_sampled_210,
        sr_input_file=dev_sampled_700,
    )  # 3,100
    convert_to_hybrid_round_version(
        mr_input_file=dev_sampled_700,
        sr_input_file=dev_sampled_1400,
    )  # 9,400


# TEST SET
def make_test_set():
    test_sampled_70 = stratified_sample_jsonl(
        input_file="data/GNER/ZSE-test.jsonl",
        output_file="data/GNER/ZSE-test-sampled.jsonl",
        max_num_samples=10,
    )  # 70
    test_sampled_210 = stratified_sample_jsonl(
        input_file="data/GNER/ZSE-test.jsonl",
        output_file="data/GNER/ZSE-test-sampled.jsonl",
        max_num_samples=30,
    )  # 210
    test_sampled_700 = stratified_sample_jsonl(
        input_file="data/GNER/ZSE-test.jsonl",
        output_file="data/GNER/ZSE-test-sampled.jsonl",
        max_num_samples=100,
    )  # 700
    test_sampled_2100 = stratified_sample_jsonl(
        input_file="data/GNER/ZSE-test.jsonl",
        output_file="data/GNER/ZSE-test-sampled.jsonl",
        max_num_samples=300,
    )  # 2,100
    convert_to_hybrid_round_version(
        mr_input_file=test_sampled_70,
        sr_input_file=test_sampled_70,
    )  # 870
    convert_to_hybrid_round_version(
        mr_input_file=test_sampled_70,
        sr_input_file=test_sampled_700,
    )  # 1,500
    convert_to_hybrid_round_version(
        mr_input_file=test_sampled_210,
        sr_input_file=test_sampled_700,
    )  # 3,100
    convert_to_hybrid_round_version(
        mr_input_file=test_sampled_700,
        sr_input_file=test_sampled_2100,
    )  # 10,100


# TRAINING SET
def make_train_set():
    train_sampled_1000 = stratified_sample_jsonl(
        input_file="data/GNER/pile-ner.jsonl",
        output_file="data/GNER/pile-ner-sampled.jsonl",
        min_num_word=10, max_num_word=80,
        min_num_label=3, max_num_label=7,
        min_num_samples=3, max_num_samples=10,
    )  # 902
    train_sampled_20000 = stratified_sample_jsonl(
        input_file="data/GNER/pile-ner.jsonl",
        output_file="data/GNER/pile-ner-sampled.jsonl",
        min_num_word=10, max_num_word=100,
        min_num_label=3, max_num_label=7,
        min_num_samples=3, max_num_samples=10,
    )  # 19,988
    train_sampled_30000 = stratified_sample_jsonl(
        input_file="data/GNER/pile-ner.jsonl",
        output_file="data/GNER/pile-ner-sampled.jsonl",
        min_num_word=10, max_num_word=100,
        min_num_label=3, max_num_label=10,
        min_num_samples=3, max_num_samples=10,
    )  # 30,129
    train_sampled_40000 = stratified_sample_jsonl(
        input_file="data/GNER/pile-ner.jsonl",
        output_file="data/GNER/pile-ner-sampled.jsonl",
        min_num_word=10, max_num_word=100,
        min_num_label=3, max_num_label=20,
        min_num_samples=3, max_num_samples=10,
    )  # 39,852
    convert_to_hybrid_round_version(
        sr_input_file="data/GNER/pile-ner.jsonl",
        mr_inst_file=None,
    )  # 103,814
    convert_to_hybrid_round_version(
        mr_input_file=train_sampled_1000,
        sr_input_file=train_sampled_1000,
    )  # 4,244
    convert_to_hybrid_round_version(
        mr_input_file=train_sampled_20000,
        sr_input_file="data/GNER/pile-ner.jsonl",
    )  # 207,842
    convert_to_hybrid_round_version(
        mr_input_file=train_sampled_30000,
        sr_input_file="data/GNER/pile-ner.jsonl",
    )  # 298,291
    convert_to_hybrid_round_version(
        mr_input_file=train_sampled_40000,
        sr_input_file="data/GNER/pile-ner.jsonl",
    )  # 433,594


if __name__ == "__main__":
    make_dev_set()
    # make_test_set()
    # make_train_set()
