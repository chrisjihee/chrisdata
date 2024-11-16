import logging
import re
from itertools import groupby
from typing import List

from pydantic import BaseModel

from chrisbase.data import AppTyper, IOArguments

app = AppTyper()
logger = logging.getLogger(__name__)
entity_text_pattern = re.compile(r'[A-Za-z ]+')


class EntityRelatedPassages(BaseModel):
    id: str
    entity: str
    passages: list[str]
    num_passages: int
    source_url: str


class GNER_TrainSample(BaseModel):
    instruction_inputs: str
    prompt_labels: str


class GNER_TrainSampleComp(BaseModel):
    id: str
    split: str
    instance: GNER_TrainSample


def entity_texts_to_freq_dict(entity_texts: List[str], args: IOArguments):
    # count entity frequency using groupby
    entity_freq = {k: len(list(g)) for k, g in groupby(sorted(entity_texts)) if entity_text_pattern.fullmatch(k)}
    # sort by frequency
    entity_freq = dict(sorted(entity_freq.items(), key=lambda x: x[1], reverse=True))
    # filter out entities with frequency less than min_entity_chars
    entity_freq = {k: v for k, v in entity_freq.items() if v >= args.option.min_entity_freq and len(k) >= args.option.min_entity_chars}
    return entity_freq


from . import convert_test
from . import convert_train
