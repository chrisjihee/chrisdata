from chrisdata.ner.gner import stratified_sample_jsonl, convert_to_hybrid_round_version


def make_dev_set_for_ZSE(input_file, output_file):
    dev_sampled_per30 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30, show_population=True)  # N210
    dev_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100)  # N700
    dev_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200)  # N1400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=dev_sampled_per200)  # SR1400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per30, sr_input_file=dev_sampled_per100)  # HR3100
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per100)  # HR8700
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per200)  # HR9400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per200, sr_input_file=dev_sampled_per200)  # HR17400


def make_dev_set_for_SFT(input_file, output_file):
    dev_sampled_per30 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30, ignore_data=["conllpp"], show_population=True)  # N540
    dev_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100, ignore_data=["conllpp"])  # N1800
    dev_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200, ignore_data=["conllpp"])  # N3600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=dev_sampled_per100)  # SR1800
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=dev_sampled_per200)  # SR3600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per30, sr_input_file=dev_sampled_per100)  # HR5400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per100)  # HR13800
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per200)  # HR15600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per200, sr_input_file=dev_sampled_per200)  # HR27600


def make_test_set_for_ZSE(input_file, output_file):
    test_sampled_per30 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30, show_population=True)  # N210
    test_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100)  # N700
    test_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200)  # N1400
    test_sampled_per500 = stratified_sample_jsonl(input_file=input_file, max_num_samples=500)  # N3312
    test_sampled_per3000 = stratified_sample_jsonl(input_file=input_file, max_num_samples=3000)  # N6470
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per500)  # SR3312
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per3000)  # SR6470
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per30, sr_input_file=test_sampled_per100)  # HR3100
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per100)  # HR8700
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per200)  # HR9400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per200)  # HR17400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per500)  # HR19312


def make_test_set_for_SFT(input_file, output_file):
    test_sampled_per30 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30, ignore_data=["conllpp"], show_population=True)  # N540
    test_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100, ignore_data=["conllpp"])  # N1800
    test_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200, ignore_data=["conllpp"])  # N3600
    test_sampled_per500 = stratified_sample_jsonl(input_file=input_file, max_num_samples=500, ignore_data=["conllpp"])  # N9000
    test_sampled_per30000 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30000, ignore_data=["conllpp"])  # N131378
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per200)  # SR3600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per500)  # SR9000
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per30000)  # SR131378
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per30, sr_input_file=test_sampled_per100)  # HR5400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per100)  # HR13800
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per200)  # HR15600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per200)  # HR27600
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per500)  # HR33000


def make_train_set_for_ZSE(input_file, output_file):
    train_sampled_tiny1 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=90,
                                                  min_num_label=3, max_num_label=6,
                                                  min_num_samples=3, max_num_samples=1)  # N2964 (3k) [Top10]
    train_sampled_tiny2 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=6,
                                                  min_num_samples=3, max_num_samples=1)  # N4973 (5k) [Top10]
    train_sampled_tiny3 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=6,
                                                  min_num_samples=3, max_num_samples=2)  # N9946 (10k) [Top10]
    train_sampled_base1 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=7,
                                                  min_num_samples=3, max_num_samples=10)  # N20026 (20k) [Top10]
    train_sampled_base2 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=10,
                                                  min_num_samples=3, max_num_samples=10)  # N30197 (30k)
    train_sampled_base3 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=15,
                                                  min_num_samples=3, max_num_samples=10)  # N37652 (38k)
    convert_to_hybrid_round_version(output_file=output_file, mr_inst_file=None, sr_input_file=input_file)  # SR103814 (100k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_tiny1, sr_input_file=train_sampled_tiny1)  # HR16938 (17k) [Top10]
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_tiny2, sr_input_file=train_sampled_tiny2)  # HR28790 (29k) [Top10]
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_tiny3, sr_input_file=train_sampled_tiny3)  # HR57578 (58k) [Top10]
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base1, sr_input_file=train_sampled_base1)  # HR123584 (124k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base1, sr_input_file=input_file)  # HR207372 (200k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base2, sr_input_file=input_file)  # HR297809 (300k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base3, sr_input_file=input_file)  # HR392516 (400k)


def make_train_set_for_SFT(input_file, output_file):
    train_sampled_tiny1 = stratified_sample_jsonl(input_file=input_file, show_population=True,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=150)  # N2700 (3k) [Top10]
    train_sampled_tiny2 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=300)  # N5400 (5k) [Top10]
    train_sampled_tiny3 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=600)  # N10800 (11k) [Top10]
    train_sampled_base1 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=1200)  # N21600 (22k) [Top10]
    train_sampled_base2 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=1800)  # N32400 (32k)
    train_sampled_base3 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=2400)  # N43200 (43k)
    convert_to_hybrid_round_version(output_file=output_file, mr_inst_file=None, sr_input_file=input_file)  # SR153180 (153k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_tiny1, sr_input_file=train_sampled_tiny1)  # HR20700 (21k) [Top10]
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_tiny2, sr_input_file=train_sampled_tiny2)  # HR41400 (41k) [Top10]
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_tiny3, sr_input_file=train_sampled_tiny3)  # HR82800 (83k) [Top10]
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base1, sr_input_file=train_sampled_base1)  # HR165600 (170k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_tiny3, sr_input_file=input_file)  # HR225180 (230k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base1, sr_input_file=input_file)  # HR297180 (300k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base2, sr_input_file=input_file)  # HR369180 (370k)
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base3, sr_input_file=input_file)  # HR441180 (440k)


if __name__ == "__main__":
    make_dev_set_for_ZSE(output_file="data/HybridGNER/ZSE-validation.jsonl", input_file="data/GNER-N2/ZSE-validation.jsonl")
    make_dev_set_for_SFT(output_file="data/HybridGNER/SFT-validation.jsonl", input_file="data/GNER-N2/SFT-validation.jsonl")
    make_test_set_for_ZSE(output_file="data/HybridGNER/ZSE-test.jsonl", input_file="data/GNER-N2/ZSE-test.jsonl")
    make_test_set_for_SFT(output_file="data/HybridGNER/SFT-test.jsonl", input_file="data/GNER-N2/SFT-test.jsonl")
    make_train_set_for_ZSE(output_file="data/HybridGNER/ZSE-train.jsonl", input_file="data/GNER-N2/ZSE-train.jsonl")
    make_train_set_for_SFT(output_file="data/HybridGNER/SFT-train.jsonl", input_file="data/GNER-N2/SFT-train.jsonl")
