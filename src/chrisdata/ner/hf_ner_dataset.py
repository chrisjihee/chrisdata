from pathlib import Path
from typing import Optional, Dict, List
from unittest.mock import patch

from datasets import load_dataset, ClassLabel
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from chrisbase.io import do_nothing
from progiter import ProgIter


class HfNerDatasetInfo(BaseModel):
    id: str
    hf_name: str
    subset: Optional[str] = Field(default=None)
    lang: Optional[str] = Field(default=None)
    label2id: Optional[Dict[str, int]] = Field(default=None)
    label_names: Optional[List[Optional[str]]] = Field(default=None)
    token_column: str = Field(default="tokens")
    label_column: str = Field(default="ner_tags")
    lang_column: str = Field(default="lang")
    train_splits: list[str] = Field(default=["train"])
    dev_splits: list[str] = Field(default=["validation"])
    test_splits: list[str] = Field(default=["test"])

    @model_validator(mode='after')
    def after(self) -> Self:
        if not self.subset:
            if len(self.hf_name.split("::")) == 2:
                self.hf_name, self.subset = self.hf_name.split("::")
            elif len(self.hf_name.split("::")) == 1:
                self.hf_name, self.subset = self.hf_name, None
            else:
                raise ValueError(f"Invalid hf_name: {self.hf_name}")
        if self.label2id and not self.label_names:
            self.label_names = [None] * len(self.label2id)
            for label, idx in self.label2id.items():
                self.label_names[idx] = label
        return self

    @property
    def source(self) -> str:
        if not self.subset:
            return f"https://huggingface.co/datasets/{self.hf_name}"
        else:
            return f"https://huggingface.co/datasets/{self.hf_name}/viewer/{self.subset}"

    @property
    def split_groups(self) -> Dict[str, List[str]]:
        return {
            "train": self.train_splits,
            "dev": self.dev_splits,
            "test": self.test_splits,
        }


def save_conll_format(split, dataset, output_file, output_mode, label_names, data_info: HfNerDatasetInfo) -> int:
    examples = dataset[split]
    if len(examples) > 10000:
        examples = ProgIter(examples, desc=f"    writing:", time_thresh=1.0)
    num_output = 0
    with Path(output_file).open(output_mode, encoding="utf-8") as f:
        for example in examples:
            if data_info.lang and data_info.lang_column in example:
                if example[data_info.lang_column] != data_info.lang:
                    continue
            tokens = example[data_info.token_column]
            tags = example[data_info.label_column]
            if len(tokens) > 0 and len(tags) > 0:
                for token, tag_id in zip(tokens, tags):
                    label = label_names[tag_id]
                    assert label is not None, f"Missing label for tag_id={tag_id}"
                    f.write(f"{token}\t{label}\n")
            f.write("\n")
            num_output += 1
    return num_output


def download_hf_dataset(data_info: HfNerDatasetInfo, output_dir: str = "data", force_download: bool = False):
    output_dir = Path(output_dir) / data_info.id
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 120)
    print(f"[HF dataset] {data_info.source} => {output_dir}")
    with patch("builtins.print", side_effect=lambda *xs: do_nothing()):
        dataset = load_dataset(
            path=data_info.hf_name,
            name=data_info.subset,
            trust_remote_code=True,
            verification_mode="no_checks",
            download_mode="force_redownload" if force_download else None,
        )
    all_label_names = []
    num_group_samples = {}
    for group in data_info.split_groups:
        num_samples = 0
        for i, split in enumerate(data_info.split_groups[group]):
            assert data_info.token_column in dataset[split].features, f"{data_info.token_column} not in dataset[{split}].features: {dataset[split].features}"
            assert data_info.label_column in dataset[split].features, f"{data_info.label_column} not in dataset[{split}].features: {dataset[split].features}"
            label_names1, label_names2 = None, None
            if isinstance(dataset[split].features[data_info.label_column].feature, ClassLabel):
                label_names1 = dataset[split].features[data_info.label_column].feature.names
            else:
                label_names2 = data_info.label_names
            assert label_names1 or label_names2, \
                f"Missing label names for {data_info.label_column}: 1) {dataset[split].features[data_info.label_column].feature}, 2) {data_info.label_names}"
            label_names = label_names1 or label_names2

            print(f"  [split] {split} : 1) {dataset[split].features[data_info.label_column]}, 2) {data_info.label_names}")  # TODO: remove after checking
            num_samples += save_conll_format(
                split, dataset,
                output_file=output_dir / f"{group}.txt",
                output_mode="w" if i == 0 else "a",
                label_names=label_names,
                data_info=data_info,
            )

            for label_name in label_names:
                if label_name not in all_label_names:
                    all_label_names.append(label_name)
        num_group_samples[group] = num_samples

    (output_dir / "label.txt").write_text("\n".join(all_label_names) + "\n")
    (output_dir / "source.txt").write_text(data_info.source)
    for group in num_group_samples:
        print(f"  # {group:5s} : {num_group_samples[group]:,}")
    print(f"  # label : {len(all_label_names):,}")
    print("-" * 120)


multinerd_label2id = {  # https://huggingface.co/datasets/Babelscape/multinerd
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-LOC": 3, "I-LOC": 4,
    "B-ORG": 5, "I-ORG": 6,
    "B-ANIM": 7, "I-ANIM": 8,
    "B-BIO": 9, "I-BIO": 10,
    "B-CEL": 11, "I-CEL": 12,
    "B-DIS": 13, "I-DIS": 14,
    "B-EVE": 15, "I-EVE": 16,
    "B-FOOD": 17, "I-FOOD": 18,
    "B-INST": 19, "I-INST": 20,
    "B-MEDIA": 21, "I-MEDIA": 22,
    "B-PLANT": 23, "I-PLANT": 24,
    "B-MYTH": 25, "I-MYTH": 26,
    "B-TIME": 27, "I-TIME": 28,
    "B-VEHI": 29, "I-VEHI": 30,
    "B-SUPER": 31, "I-SUPER": 32,
    "B-PHY": 33, "I-PHY": 34,
}

wikineural_label2id = {  # https://huggingface.co/datasets/Babelscape/wikineural
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6,
    'B-MISC': 7, 'I-MISC': 8,
}

ontonotes5_label2id = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12,
    "B-PERCENT": 13,
    "I-PERCENT": 14,
    "B-ORDINAL": 15,
    "B-MONEY": 16,
    "I-MONEY": 17,
    "B-WORK_OF_ART": 18,
    "I-WORK_OF_ART": 19,
    "B-FAC": 20,
    "B-TIME": 21,
    "I-CARDINAL": 22,
    "B-LOC": 23,
    "B-QUANTITY": 24,
    "I-QUANTITY": 25,
    "I-NORP": 26,
    "I-LOC": 27,
    "B-PRODUCT": 28,
    "I-TIME": 29,
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36
}

tweetner7_label2id = {
    "B-corporation": 0,
    "B-creative_work": 1,
    "B-event": 2,
    "B-group": 3,
    "B-location": 4,
    "B-person": 5,
    "B-product": 6,
    "I-corporation": 7,
    "I-creative_work": 8,
    "I-event": 9,
    "I-group": 10,
    "I-location": 11,
    "I-person": 12,
    "I-product": 13,
    "O": 14
}

if __name__ == "__main__":
    dataset_infos = [
        # https://huggingface.co/datasets/spyysalo/bc2gm_corpus
        # HfNerDatasetInfo(id="bc2gm", hf_name="spyysalo/bc2gm_corpus"),

        # https://huggingface.co/datasets/chintagunta85/bc4chemd
        # HfNerDatasetInfo(id="bc4chemd", hf_name="chintagunta85/bc4chemd"),

        # https://huggingface.co/datasets/ghadeermobasher/BC5CDR-Chemical-Disease or https://huggingface.co/datasets/cvlt-mao/bc5cdr
        # HfNerDatasetInfo(id="bc5cdr", hf_name="ghadeermobasher/BC5CDR-Chemical-Disease") or HfNerDatasetInfo(id="bc5cdr", hf_name="cvlt-mao/bc5cdr", label_column="tags"),

        # https://huggingface.co/datasets/strombergnlp/broad_twitter_corpus or https://huggingface.co/datasets/GateNLP/broad_twitter_corpus
        # HfNerDatasetInfo(id="broad_twitter_corpus", hf_name="strombergnlp/broad_twitter_corpus") or HfNerDatasetInfo(id="broad_twitter_corpus", hf_name="GateNLP/broad_twitter_corpus"),

        # https://huggingface.co/datasets/eriktks/conll2003
        # HfNerDatasetInfo(id="conll2003", hf_name="eriktks/conll2003"),

        # https://huggingface.co/datasets/DFKI-SLT/fabner
        # HfNerDatasetInfo(id="FabNER", hf_name="DFKI-SLT/fabner"),

        # https://huggingface.co/datasets/Babelscape/multinerd
        HfNerDatasetInfo(id="MultiNERD-en", hf_name="Babelscape/multinerd", lang="en", label2id=multinerd_label2id),
    ]
    for dataset_info in dataset_infos:
        download_hf_dataset(dataset_info)

    # download_hf_dataset("Babelscape/multinerd", "data/MultiNERD-2", label2id=multinerd_label2id)  # TODO: filter out the non-English samples
    # download_hf_dataset(
    #     "Babelscape/wikineural", "data/WikiNeural-en", label2id=wikineural_label2id,
    #     train_split="train_en", dev_splits=("val_en",), test_splits=("test_en",),
    # )
    # download_hf_dataset(
    #     dataset_path="unimelb-nlp/wikiann::en",
    #     output_dir="data/WikiANN-en",
    # )
    # download_hf_dataset("DFKI-SLT/fabner", "data/FabNER")
    # download_hf_dataset("ncbi/ncbi_disease", "data/ncbi")
    # download_hf_dataset("tner/ontonotes5", "data/Ontonotes", label2id=ontonotes5_label2id)
    # download_hf_dataset("rmyeid/polyglot_ner", "data/PolyglotNER")
    # download_hf_dataset("tner/tweetner7", "data/TweetNER7", label2id=tweetner7_label2id,
    #                     train_split="train_all", dev_splits=["validation_2020", "validation_2021"], test_splits=["test_2020"])

    # for dataset_dir in sorted([x for x in Path("data").glob("*") if x.is_dir()]):
    #     if not Path(dataset_dir / "label.txt").exists():
    #         generate_label_file_from_train(dataset_dir)

    # for dataset_dir in sorted([x for x in Path("data").glob("*") if x.is_dir()]):
    #     print_dataset_stats(dataset_dir)
