import logging
from pathlib import Path
from typing import Optional, Dict, List
from unittest.mock import patch

from datasets import load_dataset, Dataset
from datasets.utils.tqdm import disable_progress_bars, enable_progress_bars
from pydantic import BaseModel, Field

from chrisbase.data import NewProjectEnv
from chrisbase.io import do_nothing, LoggingFormat

logger = logging.getLogger(__name__)


class HfDatasetsInfo(BaseModel):
    id: str
    home: str
    subset: Optional[str] = Field(default=None)
    train_splits: list[str] = Field(default=["train"])
    test_splits: list[str] = Field(default=["test"])
    val_splits: list[str] = Field(default=["validation"])

    @property
    def path(self) -> str:
        return self.home.split("https://huggingface.co/datasets/")[-1]

    @property
    def split_groups(self) -> Dict[str, List[str]]:
        return {
            "train": self.train_splits,
            "test": self.test_splits,
            "val": self.val_splits,
        }

    def download_dataset(self, output_dir: str = "data", force_download: bool = False):
        output_dir = Path(output_dir) / self.id
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 120)
        logger.info(f"[HF dataset] {self.home} => {output_dir}")
        with patch("builtins.print", side_effect=lambda *xs: do_nothing()):
            dataset = load_dataset(
                path=self.path,
                name=self.subset,
                trust_remote_code=True,
                verification_mode="no_checks",
                download_mode="force_redownload" if force_download else None,
            )
            logger.info(f"  * splits")
            for split, split_columns in dataset.column_names.items():
                logger.info(f"    - {split:10s}: {', '.join(split_columns)} => ({len(dataset[split]):>7,} samples)")
            for split, split_names in self.split_groups.items():
                for split_name in split_names:
                    if split_name in dataset:
                        split_data: Dataset = dataset[split_name]
                        logger.info(f"  * [{split_name}] features")
                        for feature_name, feature_type in split_data.features.items():
                            logger.info(f"    - {feature_name:10s}: {feature_type}")
                        jsonl_path = output_dir / f"{split_name}.jsonl"
                        disable_progress_bars()
                        split_data.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
                        enable_progress_bars()
                        logger.info(f"    => Saved {split_name} to {jsonl_path}")
                    else:
                        logger.info(f"  * Split {split_name} not found in dataset {self.id}")
        logger.info("-" * 120)


if __name__ == "__main__":
    env = NewProjectEnv(logging_level=logging.INFO, logging_format=LoggingFormat.CHECK_20)
    datasets_info_list = [
        # HfDatasetsInfo(id="nsmc", home="https://huggingface.co/datasets/e9t/nsmc"),
        # HfDatasetsInfo(id="KLUE-TC", home="https://huggingface.co/datasets/klue/klue", subset="ynat"),
        # HfDatasetsInfo(id="KLUE-STS", home="https://huggingface.co/datasets/klue/klue", subset="sts"),
        # HfDatasetsInfo(id="KLUE-NLI", home="https://huggingface.co/datasets/klue/klue", subset="nli"),
        # HfDatasetsInfo(id="KMOU-NER", home="https://huggingface.co/datasets/nlp-kmu/kor_ner"),
        HfDatasetsInfo(id="korquad", home="https://huggingface.co/datasets/KorQuAD/squad_kor_v1"),
    ]
    for datasets_info in datasets_info_list:
        datasets_info.download_dataset(output_dir="data")
