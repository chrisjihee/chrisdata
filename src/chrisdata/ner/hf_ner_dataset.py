import re
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
    home: str
    domain: str
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
        if self.label2id and not self.label_names:
            self.label_names = [None] * len(self.label2id)
            for label, idx in self.label2id.items():
                self.label_names[idx] = label
        return self

    @property
    def path(self) -> str:
        return self.home.split("https://huggingface.co/datasets/")[-1]

    @property
    def split_groups(self) -> Dict[str, List[str]]:
        return {
            "train": self.train_splits,
            "dev": self.dev_splits,
            "test": self.test_splits,
        }

    def download_hf_dataset(self, output_dir: str = "data", force_download: bool = False):
        output_dir = Path(output_dir) / self.id
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 120)
        print(f"[HF dataset] {self.home} => {output_dir}")
        with patch("builtins.print", side_effect=lambda *xs: do_nothing()):
            dataset = load_dataset(
                path=self.path,
                name=self.subset,
                trust_remote_code=True,
                verification_mode="no_checks",
                download_mode="force_redownload" if force_download else None,
            )
        all_label_names = []
        num_group_samples = {}
        for group in self.split_groups:
            num_samples = 0
            for i, split in enumerate(self.split_groups[group]):
                assert self.token_column in dataset[split].features, f"{self.token_column} not in dataset[{split}].features: {dataset[split].features}"
                assert self.label_column in dataset[split].features, f"{self.label_column} not in dataset[{split}].features: {dataset[split].features}"
                label_names1, label_names2 = None, None
                if isinstance(dataset[split].features[self.label_column].feature, ClassLabel):
                    label_names1 = dataset[split].features[self.label_column].feature.names
                else:
                    label_names2 = self.label_names
                assert label_names1 or label_names2, \
                    f"Missing label names for {self.label_column}: 1) {dataset[split].features[self.label_column].feature}, 2) {self.label_names}"
                label_names = label_names1 or label_names2

                print(f"  [split] {split} : 1) {dataset[split].features[self.label_column]}, 2) {self.label_names}")  # TODO: remove after checking
                num_samples += self.save_conll_format(
                    split, dataset,
                    output_file=output_dir / f"{group}.txt",
                    output_mode="w" if i == 0 else "a",
                    label_names=label_names,
                )

                for label_name in label_names:
                    if label_name not in all_label_names:
                        all_label_names.append(label_name)

            num_group_samples[group] = num_samples

        all_class_names = []
        for label_name in all_label_names:
            class_name = re.sub(r"^[BIES]-|^O$", "", label_name)
            if class_name and class_name not in all_class_names:
                all_class_names.append(class_name)
        (output_dir / "label.txt").write_text("\n".join(all_class_names) + "\n")
        (output_dir / "source.txt").write_text(self.home)

        for group in num_group_samples:
            print(f"  # {group:5s} : {num_group_samples[group]:,}")
        print(f"  # label : {len(all_label_names):,}")
        print(f"  # class : {len(all_class_names):,} => {' | '.join(all_class_names)}")
        print("-" * 120)

    def save_conll_format(self, split, dataset, output_file, output_mode, label_names) -> int:
        examples = dataset[split]
        if len(examples) > 10000:
            examples = ProgIter(examples, desc=f"    writing:", time_thresh=1.0)
        num_output = 0
        with Path(output_file).open(output_mode, encoding="utf-8") as f:
            for example in examples:
                if self.lang and self.lang_column in example:
                    if example[self.lang_column] != self.lang:
                        continue
                tokens = example[self.token_column]
                tags = example[self.label_column]
                if len(tokens) > 0 and len(tags) > 0:
                    for token, tag_id in zip(tokens, tags):
                        label = label_names[tag_id]
                        assert label is not None, f"Missing label for tag_id={tag_id}"
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")
                    num_output += 1
        return num_output


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

ontonotes5_label2id = {  # https://huggingface.co/datasets/tner/ontonotes5
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

wikineural_label2id = {  # https://huggingface.co/datasets/Babelscape/wikineural
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6,
    'B-MISC': 7, 'I-MISC': 8,
}

if __name__ == "__main__":
    dataset_infos = [
        HfNerDatasetInfo(domain="general", id="conll2003", home="https://huggingface.co/datasets/eriktks/conll2003"),
        HfNerDatasetInfo(domain="general", id="conllpp", home="https://huggingface.co/datasets/ZihanWangKi/conllpp"),
        HfNerDatasetInfo(domain="general", id="MultiNERD", home="https://huggingface.co/datasets/Babelscape/multinerd", lang="en", label2id=multinerd_label2id),
        HfNerDatasetInfo(domain="general", id="Ontonotes", home="https://huggingface.co/datasets/tner/ontonotes5", label_column="tags", label2id=ontonotes5_label2id),
        # HfNerDatasetInfo(domain="general", id="PolyglotNER", home="https://huggingface.co/datasets/rmyeid/polyglot_ner"),  # FileNotFoundError: http://cs.stonybrook.edu/~polyglot/ner2/emnlp_datasets.tgz
        HfNerDatasetInfo(domain="general", id="WikiANN-en", home="https://huggingface.co/datasets/unimelb-nlp/wikiann", subset="en"),
        HfNerDatasetInfo(domain="general", id="WikiNeural", home="https://huggingface.co/datasets/Babelscape/wikineural", label2id=wikineural_label2id,
                         train_splits=["train_en"], dev_splits=["val_en"], test_splits=["test_en"]),

        HfNerDatasetInfo(domain="biomed", id="bc2gm", home="https://huggingface.co/datasets/spyysalo/bc2gm_corpus"),
        HfNerDatasetInfo(domain="biomed", id="bc4chemd", home="https://huggingface.co/datasets/chintagunta85/bc4chemd"),
        HfNerDatasetInfo(domain="biomed", id="bc5cdr", home="https://huggingface.co/datasets/ghadeermobasher/BC5CDR-Chemical-Disease"),
        HfNerDatasetInfo(domain="biomed", id="ncbi", home="https://huggingface.co/datasets/ncbi/ncbi_disease"),

        HfNerDatasetInfo(domain="social media", id="broad_twitter_corpus", home="https://huggingface.co/datasets/strombergnlp/broad_twitter_corpus"),
        HfNerDatasetInfo(domain="social media", id="TweetNER7", home="https://huggingface.co/datasets/tner/tweetner7", label_column="tags",
                         train_splits=["train_2020", "train_2021"], dev_splits=["validation_2020", "validation_2021"], test_splits=["test_2020"]),

        HfNerDatasetInfo(domain="stem", id="FabNER", home="https://huggingface.co/datasets/DFKI-SLT/fabner"),
    ]
    for dataset_info in dataset_infos:
        dataset_info.download_hf_dataset()
