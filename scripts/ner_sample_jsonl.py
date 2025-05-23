from chrisdata.ner.gner import stratified_sample_jsonl, convert_to_hybrid_round_version


def make_dev_set_for_ZSE(input_file, output_file):
    dev_sampled_per10 = stratified_sample_jsonl(input_file=input_file, max_num_samples=10, show_population=True)  # N70
    dev_sampled_per30 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30)  # N210
    dev_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100)  # N700
    dev_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200)  # N1400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=dev_sampled_per200)  # SR1400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per10, sr_input_file=dev_sampled_per10)  # HR870
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per30, sr_input_file=dev_sampled_per100)  # HR3100
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per100)  # HR8700
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per200)  # HR9400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per200, sr_input_file=dev_sampled_per200)  # HR17400


def make_dev_set_for_SFT(input_file, output_file):
    dev_sampled_per10 = stratified_sample_jsonl(input_file=input_file, max_num_samples=10, ignore_data=["conllpp"], show_population=True)  # N180
    dev_sampled_per30 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30, ignore_data=["conllpp"])  # N540
    dev_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100, ignore_data=["conllpp"])  # N1800
    dev_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200, ignore_data=["conllpp"])  # N3600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=dev_sampled_per100)  # SR1800
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=dev_sampled_per200)  # SR3600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per10, sr_input_file=dev_sampled_per10)  # HR1380
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per30, sr_input_file=dev_sampled_per100)  # HR5400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per100)  # HR13800
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per200)  # HR15600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per200, sr_input_file=dev_sampled_per200)  # HR27600


def make_test_set_for_ZSE(input_file, output_file):
    test_sampled_per10 = stratified_sample_jsonl(input_file=input_file, max_num_samples=10, show_population=True)  # N70
    test_sampled_per30 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30)  # N210
    test_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100)  # N700
    test_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200)  # N1400
    test_sampled_per500 = stratified_sample_jsonl(input_file=input_file, max_num_samples=500)  # N3312
    test_sampled_per3000 = stratified_sample_jsonl(input_file=input_file, max_num_samples=3000)  # N6470
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per500)  # SR3312
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per3000)  # SR6470
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per10, sr_input_file=test_sampled_per10)  # HR870
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per30, sr_input_file=test_sampled_per100)  # HR3100
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per100)  # HR8700
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per200)  # HR9400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per200)  # HR17400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per500)  # HR19312


def make_test_set_for_SFT(input_file, output_file):
    test_sampled_per10 = stratified_sample_jsonl(input_file=input_file, max_num_samples=10, ignore_data=["conllpp"], show_population=True)  # N180
    test_sampled_per30 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30, ignore_data=["conllpp"])  # N540
    test_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100, ignore_data=["conllpp"])  # N1800
    test_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200, ignore_data=["conllpp"])  # N3600
    test_sampled_per500 = stratified_sample_jsonl(input_file=input_file, max_num_samples=500, ignore_data=["conllpp"])  # N9000
    test_sampled_per30000 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30000, ignore_data=["conllpp"])  # N131378
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per200)  # SR3600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per500)  # SR9000
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per30000)  # SR131378
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per10, sr_input_file=test_sampled_per10)  # HR1380
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per30, sr_input_file=test_sampled_per100)  # HR5400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per100)  # HR13800
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per200)  # HR15600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per200)  # HR27600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per500)  # HR33000


def make_train_set_for_ZSE(input_file, output_file):
    train_sampled_tiny_ = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=81,
                                                  min_num_label=3, max_num_label=7,
                                                  min_num_samples=3, max_num_samples=10)  # N1207
    train_sampled_small = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=7,
                                                  min_num_samples=3, max_num_samples=10)  # N20026
    train_sampled_base_ = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=10,
                                                  min_num_samples=3, max_num_samples=10)  # N30197
    train_sampled_large = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=15,
                                                  min_num_samples=3, max_num_samples=10)  # N37652
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_tiny_, sr_input_file=train_sampled_tiny_)  # HR6040 (6k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_small, sr_input_file=input_file)  # HR207372 (200k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base_, sr_input_file=input_file)  # HR297809 (300k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_large, sr_input_file=input_file)  # HR392516 (400k)
    convert_to_hybrid_round_version(output_file=output_file, mr_inst_file=None, sr_input_file=input_file)  # SR103814 (100k)


def make_train_set_for_SFT(input_file, output_file):
    train_sampled_tiny_ = stratified_sample_jsonl(input_file=input_file, show_population=True,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=50)  # N900
    train_sampled_small = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=600)  # N10800
    train_sampled_base_ = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=1200)  # N21600
    train_sampled_large = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=1800)  # N32400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_tiny_, sr_input_file=train_sampled_tiny_)  # HR6900 (7k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_small, sr_input_file=input_file)  # HR225180 (230k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base_, sr_input_file=input_file)  # HR297180 (300k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_large, sr_input_file=input_file)  # HR369180 (370k)
    convert_to_hybrid_round_version(output_file=output_file, mr_inst_file=None, sr_input_file=input_file)  # SR153180 (150k)


if __name__ == "__main__":
    make_dev_set_for_ZSE(output_file="data/HybridGNER/ZSE-validation.jsonl", input_file="data/GNER-N2/ZSE-validation.jsonl")
    make_dev_set_for_SFT(output_file="data/HybridGNER/SFT-validation.jsonl", input_file="data/GNER-N2/SFT-validation.jsonl")
    make_test_set_for_ZSE(output_file="data/HybridGNER/ZSE-test.jsonl", input_file="data/GNER-N2/ZSE-test.jsonl")
    make_test_set_for_SFT(output_file="data/HybridGNER/SFT-test.jsonl", input_file="data/GNER-N2/SFT-test.jsonl")
    # make_train_set_for_ZSE(output_file="data/HybridGNER/ZSE-train.jsonl", input_file="data/GNER-N2/ZSE-train.jsonl")
    # make_train_set_for_SFT(output_file="data/HybridGNER/SFT-train.jsonl", input_file="data/GNER-N2/SFT-train.jsonl")
