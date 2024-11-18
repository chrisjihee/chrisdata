import logging
import re

from pydantic import BaseModel

from chrisbase.data import AppTyper

app = AppTyper()
logger = logging.getLogger(__name__)
entity_text_pattern = re.compile(r'[A-Za-z ]+')


class EntityRelatedPassages(BaseModel):
    id: str = None
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


from . import convert_GNER
