from chrisdata.ner.gner import stratified_sample_jsonl, convert_to_hybrid_round_version

# VALIDATION SET
validation_sampled_70 = stratified_sample_jsonl(
    input_file="data/ZSE-validation.jsonl",
    output_file="data/ZSE-validation-sampled.jsonl",
    max_num_samples=10,
)  # 70
validation_sampled_210 = stratified_sample_jsonl(
    input_file="data/ZSE-validation.jsonl",
    output_file="data/ZSE-validation-sampled.jsonl",
    max_num_samples=30,
)  # 210
validation_sampled_700 = stratified_sample_jsonl(
    input_file="data/ZSE-validation.jsonl",
    output_file="data/ZSE-validation-sampled.jsonl",
    max_num_samples=100,
)  # 700
validation_sampled_2100 = stratified_sample_jsonl(
    input_file="data/ZSE-validation.jsonl",
    output_file="data/ZSE-validation-sampled.jsonl",
    max_num_samples=300,
)  # 2,100
convert_to_hybrid_round_version(
    mr_input_file=validation_sampled_70,
    sr_input_file=validation_sampled_700,
)
convert_to_hybrid_round_version(
    mr_input_file=validation_sampled_210,
    sr_input_file=validation_sampled_700,
)
convert_to_hybrid_round_version(
    mr_input_file=validation_sampled_700,
    sr_input_file=validation_sampled_2100,
)

# TEST SET
test_sampled_70 = stratified_sample_jsonl(
    input_file="data/ZSE-test.jsonl",
    output_file="data/ZSE-test-sampled.jsonl",
    max_num_samples=10,
)  # 70
test_sampled_210 = stratified_sample_jsonl(
    input_file="data/ZSE-test.jsonl",
    output_file="data/ZSE-test-sampled.jsonl",
    max_num_samples=30,
)  # 210
test_sampled_700 = stratified_sample_jsonl(
    input_file="data/ZSE-test.jsonl",
    output_file="data/ZSE-test-sampled.jsonl",
    max_num_samples=100,
)  # 700
test_sampled_2100 = stratified_sample_jsonl(
    input_file="data/ZSE-test.jsonl",
    output_file="data/ZSE-test-sampled.jsonl",
    max_num_samples=300,
)  # 2,100
convert_to_hybrid_round_version(
    mr_input_file=test_sampled_70,
    sr_input_file=test_sampled_700,
)
convert_to_hybrid_round_version(
    mr_input_file=test_sampled_210,
    sr_input_file=test_sampled_700,
)
convert_to_hybrid_round_version(
    mr_input_file=test_sampled_700,
    sr_input_file=test_sampled_2100,
)

# TRAINING SET
train_sampled_20000 = stratified_sample_jsonl(
    input_file="data/pile-ner.jsonl",
    output_file="data/pile-ner-sampled.jsonl",
    min_num_word=10, max_num_word=100,
    min_num_label=3, max_num_label=7,
    min_num_samples=3, max_num_samples=10,
)  # 19,988
train_sampled_30000 = stratified_sample_jsonl(
    input_file="data/pile-ner.jsonl",
    output_file="data/pile-ner-sampled.jsonl",
    min_num_word=10, max_num_word=100,
    min_num_label=3, max_num_label=10,
    min_num_samples=3, max_num_samples=10,
)  # 30,129
train_sampled_40000 = stratified_sample_jsonl(
    input_file="data/pile-ner.jsonl",
    output_file="data/pile-ner-sampled.jsonl",
    min_num_word=10, max_num_word=100,
    min_num_label=3, max_num_label=20,
    min_num_samples=3, max_num_samples=10,
)  # 39,852
train_sampled_50000 = stratified_sample_jsonl(
    input_file="data/pile-ner.jsonl",
    output_file="data/pile-ner-sampled.jsonl",
    min_num_word=7, max_num_word=120,
    min_num_label=3, max_num_label=20,
    min_num_samples=3, max_num_samples=10,
)  # 49,800
convert_to_hybrid_round_version(
    sr_input_file="data/pile-ner.jsonl",
    mr_inst_file=None,
)
for train_sampled in [
    train_sampled_20000,
    train_sampled_30000,
    train_sampled_40000,
    train_sampled_50000,
]:
    convert_to_hybrid_round_version(
        mr_input_file=train_sampled,
        sr_input_file="data/pile-ner.jsonl",
    )
