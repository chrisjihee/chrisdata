from chrisdata.ner.gner import stratified_sample_jsonl, convert_to_hybrid_round_version


def make_dev_set_for_ZSE(input_file, output_file):
    dev_sampled_70 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=10, show_population=True)  # 70
    dev_sampled_210 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=30)  # 210
    dev_sampled_700 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=100)  # 700
    dev_sampled_1400 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=200)  # 1,400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=input_file)  # 1,400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_70, sr_input_file=dev_sampled_70)  # 870
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_70, sr_input_file=dev_sampled_700)  # 1,500
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_210, sr_input_file=dev_sampled_700)  # 3,100
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_700, sr_input_file=dev_sampled_1400)  # 9,400


def make_dev_set_for_SFT(input_file, output_file):
    dev_sampled_160 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=10, show_population=True)  # 160
    dev_sampled_480 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=30)  # 480
    dev_sampled_1600 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=100)  # 1,600
    dev_sampled_3200 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=200)  # 3,200
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=input_file)  # 3,800
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_160, sr_input_file=dev_sampled_160)  # 1,290
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_160, sr_input_file=dev_sampled_1600)  # 2,730
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_480, sr_input_file=dev_sampled_1600)  # 4,990
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=dev_sampled_1600, sr_input_file=dev_sampled_3200)  # 14,500


def make_test_set_for_ZSE(input_file, output_file):
    test_sampled_70 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=10, show_population=True)  # 70
    test_sampled_210 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=30)  # 210
    test_sampled_700 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=100)  # 700
    test_sampled_1400 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=200)  # 1,400
    test_sampled_3300 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=500)  # 3,312
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_3300)  # 3,312
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=input_file)  # 6,470
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_70, sr_input_file=test_sampled_70)  # 870
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_70, sr_input_file=test_sampled_700)  # 1,500
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_210, sr_input_file=test_sampled_700)  # 3,100
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_700, sr_input_file=test_sampled_1400)  # 9,400
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_1400, sr_input_file=test_sampled_3300)  # 19,312


def make_test_set_for_SFT(input_file, output_file):
    test_sampled_190 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=10, show_population=True)  # 190
    test_sampled_570 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=30)  # 570
    test_sampled_1900 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=100)  # 1,900
    test_sampled_3800 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=200)  # 3,800
    test_sampled_9500 = stratified_sample_jsonl(output_file=output_file, input_file=input_file, max_num_samples=500)  # 9,500
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=test_sampled_9500)  # 9,500
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=None, sr_input_file=input_file)  # 134,835
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_190, sr_input_file=test_sampled_190)  # 1,430
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_190, sr_input_file=test_sampled_1900)  # 3,140
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_570, sr_input_file=test_sampled_1900)  # 5,620
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_1900, sr_input_file=test_sampled_3800)  # 16,200
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=test_sampled_3800, sr_input_file=test_sampled_9500)  # 34,300


def make_train_set_for_ZSE(input_file, output_file):
    train_sampled_1k = stratified_sample_jsonl(output_file=output_file, input_file=input_file,
                                               min_num_word=10, max_num_word=81,
                                               min_num_label=3, max_num_label=7,
                                               min_num_samples=3, max_num_samples=10)  # 1207
    train_sampled_20k = stratified_sample_jsonl(output_file=output_file, input_file=input_file,
                                                min_num_word=10, max_num_word=100,
                                                min_num_label=3, max_num_label=7,
                                                min_num_samples=3, max_num_samples=10)  # 19,988
    train_sampled_30k = stratified_sample_jsonl(output_file=output_file, input_file=input_file,
                                                min_num_word=10, max_num_word=100,
                                                min_num_label=3, max_num_label=10,
                                                min_num_samples=3, max_num_samples=10)  # 30,129
    train_sampled_40k = stratified_sample_jsonl(output_file=output_file, input_file=input_file,
                                                min_num_word=10, max_num_word=100,
                                                min_num_label=3, max_num_label=20,
                                                min_num_samples=3, max_num_samples=10)  # 39,852
    convert_to_hybrid_round_version(output_file=output_file, mr_inst_file=None, sr_input_file=input_file)  # 103,814
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_1k, sr_input_file=train_sampled_1k)  # 4,244
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_20k, sr_input_file=input_file)  # 207,842
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_30k, sr_input_file=input_file)  # 298,291
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_40k, sr_input_file=input_file)  # 433,594


def make_train_set_for_SFT(input_file, output_file):
    train_sampled_1k = stratified_sample_jsonl(output_file=output_file, input_file=input_file,
                                               min_num_word=3, max_num_word=100,
                                               min_num_label=1, max_num_label=30, show_population=True,
                                               min_num_samples=3, max_num_samples=50)  # 900
    train_sampled_18k = stratified_sample_jsonl(output_file=output_file, input_file=input_file,
                                                min_num_word=3, max_num_word=100,
                                                min_num_label=1, max_num_label=30,
                                                min_num_samples=3, max_num_samples=1000)  # 18,000
    train_sampled_27k = stratified_sample_jsonl(output_file=output_file, input_file=input_file,
                                                min_num_word=3, max_num_word=100,
                                                min_num_label=1, max_num_label=30,
                                                min_num_samples=3, max_num_samples=1500)  # 27,000
    train_sampled_36k = stratified_sample_jsonl(output_file=output_file, input_file=input_file,
                                                min_num_word=3, max_num_word=100,
                                                min_num_label=1, max_num_label=30,
                                                min_num_samples=3, max_num_samples=2000)  # 36,000
    convert_to_hybrid_round_version(output_file=output_file, mr_inst_file=None, sr_input_file=input_file)  # 153,180
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_1k, sr_input_file=train_sampled_1k)  # 6,900
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_18k, sr_input_file=input_file)  # 273,180
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_27k, sr_input_file=input_file)  # 333,180
    convert_to_hybrid_round_version(output_file=output_file, mr_input_file=train_sampled_36k, sr_input_file=input_file)  # 393,180


if __name__ == "__main__":
    make_dev_set_for_ZSE(output_file="data/HybridGNER/ZSE-validation.jsonl", input_file="data/GNER-N2/ZSE-validation.jsonl")
    make_dev_set_for_SFT(output_file="data/HybridGNER/SFT-validation.jsonl", input_file="data/GNER-N2/SFT-validation.jsonl")
    make_test_set_for_ZSE(output_file="data/HybridGNER/ZSE-test.jsonl", input_file="data/GNER-N2/ZSE-test.jsonl")
    make_test_set_for_SFT(output_file="data/HybridGNER/SFT-test.jsonl", input_file="data/GNER-N2/SFT-test.jsonl")
    make_train_set_for_ZSE(output_file="data/HybridGNER/ZSE-train.jsonl", input_file="data/GNER-N2/ZSE-train.jsonl")
    make_train_set_for_SFT(output_file="data/HybridGNER/SFT-train.jsonl", input_file="data/GNER-N2/SFT-train.jsonl")
