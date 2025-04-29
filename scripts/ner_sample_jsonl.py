from chrisdata.ner.gner import stratified_sample_jsonl, convert_to_hybrid_round_version


def make_dev_set_for_ZSE(input_file, output_file):
    dev_sampled_70 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=10)  # 70
    dev_sampled_210 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=30)  # 210
    dev_sampled_700 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=100)  # 700
    dev_sampled_1400 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=200)  # 1,400
    dev_combined_870 = convert_to_hybrid_round_version(mr_input_file=dev_sampled_70, sr_input_file=dev_sampled_70)  # 870
    dev_combined_1500 = convert_to_hybrid_round_version(mr_input_file=dev_sampled_70, sr_input_file=dev_sampled_700)  # 1,500
    dev_combined_3100 = convert_to_hybrid_round_version(mr_input_file=dev_sampled_210, sr_input_file=dev_sampled_700)  # 3,100
    dev_combined_9400 = convert_to_hybrid_round_version(mr_input_file=dev_sampled_700, sr_input_file=dev_sampled_1400)  # 9,400
    return ([dev_sampled_70, dev_sampled_210, dev_sampled_700, dev_sampled_1400] +
            [dev_combined_870, dev_combined_1500, dev_combined_3100, dev_combined_9400])


def make_dev_set_for_SFT(input_file, output_file):
    dev_sampled_160 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=10)  # 160
    dev_sampled_480 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=30)  # 480
    dev_sampled_1600 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=100)  # 1,600
    dev_sampled_3200 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=200)  # 3,200
    dev_combined_1k = convert_to_hybrid_round_version(mr_input_file=dev_sampled_160, sr_input_file=dev_sampled_160)  # 1,290
    dev_combined_3k = convert_to_hybrid_round_version(mr_input_file=dev_sampled_160, sr_input_file=dev_sampled_1600)  # 2,730
    dev_combined_5k = convert_to_hybrid_round_version(mr_input_file=dev_sampled_480, sr_input_file=dev_sampled_1600)  # 4,990
    dev_combined_15k = convert_to_hybrid_round_version(mr_input_file=dev_sampled_1600, sr_input_file=dev_sampled_3200)  # 14,500
    return ([dev_sampled_160, dev_sampled_480, dev_sampled_1600, dev_sampled_3200] +
            [dev_combined_1k, dev_combined_3k, dev_combined_5k, dev_combined_15k])


def make_test_set_for_ZSE(input_file, output_file):
    test_sampled_70 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=10)  # 70
    test_sampled_210 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=30)  # 210
    test_sampled_700 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=100)  # 700
    test_sampled_1400 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=200)  # 1,400
    test_combined_870 = convert_to_hybrid_round_version(mr_input_file=test_sampled_70, sr_input_file=test_sampled_70)  # 870
    test_combined_1500 = convert_to_hybrid_round_version(mr_input_file=test_sampled_70, sr_input_file=test_sampled_700)  # 1,500
    test_combined_3100 = convert_to_hybrid_round_version(mr_input_file=test_sampled_210, sr_input_file=test_sampled_700)  # 3,100
    test_combined_9400 = convert_to_hybrid_round_version(mr_input_file=test_sampled_700, sr_input_file=test_sampled_1400)  # 9,400
    return ([test_sampled_70, test_sampled_210, test_sampled_700, test_sampled_1400] +
            [test_combined_870, test_combined_1500, test_combined_3100, test_combined_9400])


def make_test_set_for_SFT(input_file, output_file):
    test_sampled_160 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=10)  # 160
    test_sampled_480 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=30)  # 480
    test_sampled_1600 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=100)  # 1,600
    test_sampled_3200 = stratified_sample_jsonl(input_file=input_file, output_file=output_file, max_num_samples=200)  # 3,200
    test_combined_1k = convert_to_hybrid_round_version(mr_input_file=test_sampled_160, sr_input_file=test_sampled_160)  # 1,290
    test_combined_3k = convert_to_hybrid_round_version(mr_input_file=test_sampled_160, sr_input_file=test_sampled_1600)  # 2,730
    test_combined_5k = convert_to_hybrid_round_version(mr_input_file=test_sampled_480, sr_input_file=test_sampled_1600)  # 4,990
    test_combined_15k = convert_to_hybrid_round_version(mr_input_file=test_sampled_1600, sr_input_file=test_sampled_3200)  # 14,500
    return ([test_sampled_160, test_sampled_480, test_sampled_1600, test_sampled_3200] +
            [test_combined_1k, test_combined_3k, test_combined_5k, test_combined_15k])


def make_train_set_for_ZSE(input_file, output_file):
    train_sampled_1k = stratified_sample_jsonl(input_file=input_file, output_file=output_file,
                                               min_num_word=10, max_num_word=81,
                                               min_num_label=3, max_num_label=7,
                                               min_num_samples=3, max_num_samples=10)  # 1207
    train_sampled_20k = stratified_sample_jsonl(input_file=input_file, output_file=output_file,
                                                min_num_word=10, max_num_word=100,
                                                min_num_label=3, max_num_label=7,
                                                min_num_samples=3, max_num_samples=10)  # 19,988
    train_sampled_30k = stratified_sample_jsonl(input_file=input_file, output_file=output_file,
                                                min_num_word=10, max_num_word=100,
                                                min_num_label=3, max_num_label=10,
                                                min_num_samples=3, max_num_samples=10)  # 30,129
    train_sampled_40k = stratified_sample_jsonl(input_file=input_file, output_file=output_file,
                                                min_num_word=10, max_num_word=100,
                                                min_num_label=3, max_num_label=20,
                                                min_num_samples=3, max_num_samples=10)  # 39,852
    train_combined_100k = convert_to_hybrid_round_version(sr_input_file=input_file, mr_inst_file=None)  # 103,814
    train_combined_4k = convert_to_hybrid_round_version(mr_input_file=train_sampled_1k, sr_input_file=train_sampled_1k)  # 4,244
    train_combined_200k = convert_to_hybrid_round_version(mr_input_file=train_sampled_20k, sr_input_file=input_file)  # 207,842
    train_combined_300k = convert_to_hybrid_round_version(mr_input_file=train_sampled_30k, sr_input_file=input_file)  # 298,291
    train_combined_430k = convert_to_hybrid_round_version(mr_input_file=train_sampled_40k, sr_input_file=input_file)  # 433,594
    return ([train_sampled_1k, train_sampled_20k, train_sampled_30k, train_sampled_40k] +
            [train_combined_100k, train_combined_4k, train_combined_200k, train_combined_300k, train_combined_430k])


def make_train_set_for_SFT(input_file, output_file):
    train_sampled_1k = stratified_sample_jsonl(input_file=input_file, output_file=output_file,
                                               min_num_word=3, max_num_word=100,
                                               min_num_label=1, max_num_label=30, show_population=True,
                                               min_num_samples=3, max_num_samples=50)  # 900
    train_sampled_18k = stratified_sample_jsonl(input_file=input_file, output_file=output_file,
                                                min_num_word=3, max_num_word=100,
                                                min_num_label=1, max_num_label=30,
                                                min_num_samples=3, max_num_samples=1000)  # 18,000
    train_sampled_27k = stratified_sample_jsonl(input_file=input_file, output_file=output_file,
                                                min_num_word=3, max_num_word=100,
                                                min_num_label=1, max_num_label=30,
                                                min_num_samples=3, max_num_samples=1500)  # 27,000
    train_sampled_36k = stratified_sample_jsonl(input_file=input_file, output_file=output_file,
                                                min_num_word=3, max_num_word=100,
                                                min_num_label=1, max_num_label=30,
                                                min_num_samples=3, max_num_samples=2000)  # 36,000
    train_combined_150k = convert_to_hybrid_round_version(sr_input_file=input_file, mr_inst_file=None)  # 153,180
    train_combined_7k = convert_to_hybrid_round_version(mr_input_file=train_sampled_1k, sr_input_file=train_sampled_1k)  # 6,900
    train_combined_270k = convert_to_hybrid_round_version(mr_input_file=train_sampled_18k, sr_input_file=input_file)  # 273,180
    train_combined_330k = convert_to_hybrid_round_version(mr_input_file=train_sampled_27k, sr_input_file=input_file)  # 333,180
    train_combined_400k = convert_to_hybrid_round_version(mr_input_file=train_sampled_36k, sr_input_file=input_file)  # 393,180
    return ([train_sampled_1k, train_sampled_18k, train_sampled_27k, train_sampled_36k] +
            [train_combined_150k, train_combined_7k, train_combined_270k, train_combined_330k, train_combined_400k])


if __name__ == "__main__":
    ZSE = False
    SFT = True

    if ZSE:
        make_dev_set_for_ZSE(
            input_file="data/GNER-N2/GNER-ZSE-validation.jsonl",
            output_file="data/HybridGNER/GNER-ZSE-validation.jsonl"
        )
        make_test_set_for_ZSE(
            input_file="data/GNER-N2/GNER-ZSE-test.jsonl",
            output_file="data/HybridGNER/GNER-ZSE-test.jsonl"
        )
        make_train_set_for_ZSE(
            input_file="data/GNER-N2/pile-ner.jsonl",
            output_file="data/HybridGNER/pile-ner.jsonl"
        )

    if SFT:
        make_dev_set_for_SFT(
            input_file="data/GNER-N2/GNER-SFT-validation.jsonl",
            output_file="data/HybridGNER/GNER-SFT-validation.jsonl"
        )
        make_test_set_for_SFT(
            input_file="data/GNER-N2/GNER-SFT-test.jsonl",
            output_file="data/HybridGNER/GNER-SFT-test.jsonl"
        )
        make_train_set_for_SFT(
            input_file="data/GNER-N2/GNER-SFT-train.jsonl",
            output_file="data/HybridGNER/GNER-SFT-train.jsonl"
        )
