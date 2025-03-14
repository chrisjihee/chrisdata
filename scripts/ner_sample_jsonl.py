from chrisdata.ner.gner import stratified_sample_jsonl_lines

# stratified_sample_jsonl_lines(input_file="data/pile-ner.jsonl", min_num_word=7, max_num_word=120, min_num_label=3, max_num_label=20, min_num_samples=3, max_num_samples=10)
# stratified_sample_jsonl_lines(input_file="data/pile-ner.jsonl", min_num_word=10, max_num_word=100, min_num_label=3, max_num_label=20, min_num_samples=3, max_num_samples=10)
# stratified_sample_jsonl_lines(input_file="data/pile-ner.jsonl", min_num_word=10, max_num_word=100, min_num_label=3, max_num_label=10, min_num_samples=3, max_num_samples=10)
# stratified_sample_jsonl_lines(input_file="data/pile-ner.jsonl", min_num_word=10, max_num_word=100, min_num_label=3, max_num_label=7, min_num_samples=3, max_num_samples=10)
stratified_sample_jsonl_lines(input_file="data/ZSE-validation.jsonl", max_num_samples=100,
                              output_file="data/ZSE-validation-sampled.jsonl")
stratified_sample_jsonl_lines(input_file="data/ZSE-test.jsonl", max_num_samples=100,
                              output_file="data/ZSE-test-sampled.jsonl")
