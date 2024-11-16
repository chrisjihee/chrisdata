import logging

from chrisbase.data import AppTyper
from pydantic import BaseModel

app = AppTyper()
logger = logging.getLogger(__name__)


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


from . import convert_test
from . import convert_train
