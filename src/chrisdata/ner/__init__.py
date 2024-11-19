import logging
import re
from pathlib import Path

from pydantic import BaseModel

from chrisbase.data import AppTyper

app = AppTyper()
logger = logging.getLogger(__name__)
entity_text_pattern = re.compile(r'[A-Za-z ]+')
bio_tag_pattern = re.compile("([^ ]+)\(([BIO](-[A-Za-z ]+)?)\)")


def bio_to_entities(words, labels):
    # BIO notation
    pairs = zip(words, labels)
    # convert BIO notation to NER format
    entities = []
    entity = None
    for word, label in pairs:
        if label == 'O':
            entity = None
            continue
        if label.startswith('B-'):
            entity = {'type': label[2:], 'words': [word]}
            entities.append(entity)
        elif label.startswith('I-') and entity:
            entity['words'].append(word)
    for entity in entities:
        entity['text'] = ' '.join(entity.pop('words'))
    return entities


class EntityRelatedPassages(BaseModel):
    id: str = None
    entity: str
    passages: list[str]
    num_passages: int
    source_url: str


class GenNERSample(BaseModel):
    id: str = None
    words: list[str] = None
    labels: list[str] = None
    instruction_inputs: str
    prompt_labels: str


class GenNERSampleWrapper(BaseModel):
    id: str = None
    dataset: str = "unknown"
    split: str = "unknown"
    label_list: list[str] = None
    instance: GenNERSample

    def set_missing_values(self, path: Path | str = None):
        instruction_lines = self.instance.instruction_inputs.splitlines()
        label_list, sentence = instruction_lines[-2:]
        label_list = label_list.split("Use the specific entity tags:")[-1].strip()
        label_list = [x.strip() for x in re.split("[,.]|and O", label_list) if x.strip()]
        self.label_list = sorted(label_list)

        words = sentence.split("Sentence:")[-1].strip().split()
        self.instance.words = words

        labels = self.instance.prompt_labels
        labels = [g[1] for g in bio_tag_pattern.findall(labels)]
        self.instance.labels = labels

        self.id = self.instance.id = self.id or self.instance.id
        if self.dataset == "unknown" and path:
            self.dataset = Path(path).stem
        return self


from . import convert_GNER
