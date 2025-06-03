from chrisdata.ner.gner import stratified_sample_jsonl, convert_to_hybrid_round_version, split_data_into_two_files


def make_dev_set_for_ZSE(input_file, output_file):
    dev_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100, show_population=True)  # N700
    dev_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200)  # N1400
    dev_sampled_per200_SR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=dev_sampled_per200)  # SR1400
    dev_sampled_per100_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per100)  # HR8700
    dev_sampled_per200_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per200, sr_input_file=dev_sampled_per200)  # HR17400
    split_data_into_two_files(input_file=dev_sampled_per200_SR, split_ratio="7:3")
    split_data_into_two_files(input_file=dev_sampled_per200_SR, split_ratio="5:5")
    split_data_into_two_files(input_file=dev_sampled_per100_HR, split_ratio="7:3")
    split_data_into_two_files(input_file=dev_sampled_per100_HR, split_ratio="5:5")


def make_dev_set_for_SFT(input_file, output_file):
    dev_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100, ignore_data=["conllpp"], show_population=True)  # N1800
    dev_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200, ignore_data=["conllpp"])  # N3600
    dev_sampled_per200_SR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=dev_sampled_per200)  # SR3600
    dev_sampled_per100_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per100, sr_input_file=dev_sampled_per100)  # HR13800
    dev_sampled_per200_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_per200, sr_input_file=dev_sampled_per200)  # HR27600
    split_data_into_two_files(input_file=dev_sampled_per200_SR, split_ratio="7:3")
    split_data_into_two_files(input_file=dev_sampled_per200_SR, split_ratio="5:5")
    split_data_into_two_files(input_file=dev_sampled_per100_HR, split_ratio="7:3")
    split_data_into_two_files(input_file=dev_sampled_per100_HR, split_ratio="5:5")


def make_test_set_for_ZSE(input_file, output_file):
    test_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100, show_population=True)  # N700
    test_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200)  # N1400
    test_sampled_per400 = stratified_sample_jsonl(input_file=input_file, max_num_samples=400)  # N2800
    test_sampled_per3000 = stratified_sample_jsonl(input_file=input_file, max_num_samples=3000)  # N6470
    test_sampled_per200_SR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per200)  # SR1400
    test_sampled_per400_SR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per400)  # SR2800
    test_sampled_per3000_SR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per3000)  # SR6470
    test_sampled_per100_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per100)  # HR8700
    test_sampled_per200_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per200)  # HR17400
    test_sampled_per400_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per400, sr_input_file=test_sampled_per400)  # HR34800
    split_data_into_two_files(input_file=test_sampled_per400_SR, split_ratio="7:3")
    split_data_into_two_files(input_file=test_sampled_per400_SR, split_ratio="5:5")
    split_data_into_two_files(input_file=test_sampled_per200_HR, split_ratio="7:3")
    split_data_into_two_files(input_file=test_sampled_per200_HR, split_ratio="5:5")


def make_test_set_for_SFT(input_file, output_file):
    test_sampled_per100 = stratified_sample_jsonl(input_file=input_file, max_num_samples=100, ignore_data=["conllpp"], show_population=True)  # N1800
    test_sampled_per200 = stratified_sample_jsonl(input_file=input_file, max_num_samples=200, ignore_data=["conllpp"])  # N3600
    test_sampled_per400 = stratified_sample_jsonl(input_file=input_file, max_num_samples=400, ignore_data=["conllpp"])  # N7200
    test_sampled_per30000 = stratified_sample_jsonl(input_file=input_file, max_num_samples=30000, ignore_data=["conllpp"])  # N131378
    test_sampled_per200_SR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per200)  # SR3600
    test_sampled_per400_SR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per400)  # SR7200
    test_sampled_per30000_SR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_per30000)  # SR131378
    test_sampled_per100_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per100, sr_input_file=test_sampled_per100)  # HR13800
    test_sampled_per200_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per200, sr_input_file=test_sampled_per200)  # HR27600
    test_sampled_per400_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_per400, sr_input_file=test_sampled_per400)  # HR55200
    split_data_into_two_files(input_file=test_sampled_per400_SR, split_ratio="7:3")
    split_data_into_two_files(input_file=test_sampled_per400_SR, split_ratio="5:5")
    split_data_into_two_files(input_file=test_sampled_per200_HR, split_ratio="7:3")
    split_data_into_two_files(input_file=test_sampled_per200_HR, split_ratio="5:5")


def make_train_set_for_ZSE(input_file, output_file):
    train_sampled_base0 = stratified_sample_jsonl(input_file=input_file, show_population=False,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=6,
                                                  min_num_samples=3, max_num_samples=2)  # N9946 (10k)
    train_sampled_base1 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=7,
                                                  min_num_samples=3, max_num_samples=10)  # N20026 (20k)
    train_sampled_base2 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=10, max_num_word=100,
                                                  min_num_label=3, max_num_label=15,
                                                  min_num_samples=3, max_num_samples=10)  # N37652 (38k)
    train_sampled_base0_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base0, sr_input_file=train_sampled_base0)  # HR57578 (58k)
    train_sampled_base1_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base1, sr_input_file=train_sampled_base1)  # HR123584 (124k)
    train_sampled_full1_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base1, sr_input_file=input_file)  # HR207372 (200k)
    train_sampled_full2_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base2, sr_input_file=input_file)  # HR392516 (400k)
    train_sampled_full0_SR = convert_to_hybrid_round_version(output_file=output_file, mr_inst_file=None, sr_input_file=input_file)  # SR103814 (100k)
    split_data_into_two_files(input_file=train_sampled_full0_SR, split_ratio="7:3")
    split_data_into_two_files(input_file=train_sampled_full0_SR, split_ratio="5:5")
    split_data_into_two_files(input_file=train_sampled_full1_HR, split_ratio="7:3")
    split_data_into_two_files(input_file=train_sampled_full1_HR, split_ratio="5:5")


def make_train_set_for_SFT(input_file, output_file):
    train_sampled_base0 = stratified_sample_jsonl(input_file=input_file, show_population=True,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=600)  # N10800 (11k)
    train_sampled_base1 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=1200)  # N21600 (22k)
    train_sampled_base2 = stratified_sample_jsonl(input_file=input_file,
                                                  min_num_word=3, max_num_word=100,
                                                  min_num_label=1, max_num_label=30,
                                                  min_num_samples=3, max_num_samples=2400)  # N43200 (43k)
    train_sampled_base0_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base0, sr_input_file=train_sampled_base0)  # HR82800 (83k)
    train_sampled_base1_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base1, sr_input_file=train_sampled_base1)  # HR165600 (170k)
    train_sampled_full1_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base1, sr_input_file=input_file)  # HR297180 (300k)
    train_sampled_full2_HR = convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_base2, sr_input_file=input_file)  # HR441180 (440k)
    train_sampled_full0_SR = convert_to_hybrid_round_version(output_file=output_file, mr_inst_file=None, sr_input_file=input_file)  # SR153180 (153k)
    split_data_into_two_files(input_file=train_sampled_full0_SR, split_ratio="7:3")
    split_data_into_two_files(input_file=train_sampled_full0_SR, split_ratio="5:5")
    split_data_into_two_files(input_file=train_sampled_full1_HR, split_ratio="7:3")
    split_data_into_two_files(input_file=train_sampled_full1_HR, split_ratio="5:5")



if __name__ == "__main__":
    make_dev_set_for_ZSE(output_file="data/HybridGNER/ZSE-validation.jsonl", input_file="data/GNER-N2/ZSE-validation.jsonl")
    make_dev_set_for_SFT(output_file="data/HybridGNER/SFT-validation.jsonl", input_file="data/GNER-N2/SFT-validation.jsonl")
    make_test_set_for_ZSE(output_file="data/HybridGNER/ZSE-test.jsonl", input_file="data/GNER-N2/ZSE-test.jsonl")
    make_test_set_for_SFT(output_file="data/HybridGNER/SFT-test.jsonl", input_file="data/GNER-N2/SFT-test.jsonl")
    make_train_set_for_ZSE(output_file="data/HybridGNER/ZSE-train.jsonl", input_file="data/GNER-N2/ZSE-train.jsonl")
    make_train_set_for_SFT(output_file="data/HybridGNER/SFT-train.jsonl", input_file="data/GNER-N2/SFT-train.jsonl")
