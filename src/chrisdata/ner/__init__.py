import logging
import random
import re
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from chrisbase.data import AppTyper
from .gner_dataset import GNERDataset, GNERConfig

app = AppTyper()
logger = logging.getLogger(__name__)

# NER patterns
bio_tag_pattern = re.compile("([^ ]+)\(([BIO](-[A-Za-z ]+)?)\)")
entity_text_pattern = re.compile(r'[A-Za-z ]+')

# Wiki patterns
file_pattern = re.compile("\[\[File:([^]]+)]]")
space_pattern = re.compile("  +")
link2_pattern = re.compile("\[\[([^|\]]+)\|([^]]+)]]")
link1_pattern = re.compile("\[\[([^]]+)]]")
bold3_pattern = re.compile("'''([^']+)'''")
bold2_pattern = re.compile("''([^']+)''")
special_pattern1 = re.compile("{{.+?}}")
special_pattern2 = re.compile("{{[^}]+?}}")
reference_pattern = re.compile("<ref[^>]*>.*?</ref>|<ref[^>]*/>")


# TODO: remove comments! comment_pattern = re.compile("<!--.*?-->")
# <!--Before adding other occupations, please discuss on the talk page. Do not add unless they are notable per [[MOS:LEADSENTENCE]]-->

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
    instruction_inputs: str = None
    prompt_labels: str = None

    @staticmethod
    def from_wiki_passage(wiki_passage: str, label: str, id: str = None) -> "GenNERSample":
        words_labels, prev_end = list(), 0
        for m in link1_pattern.finditer(wiki_passage):
            words_labels.extend([(w, 'O') for w in wiki_passage[prev_end: m.start()].split()])
            words_labels.extend([(w, f'B-{label}' if i == 0 else f'I-{label}') for i, w in enumerate(m.group(1).split())])
            prev_end = m.end()
        words_labels.extend([(w, 'O') for w in wiki_passage[prev_end:].split()])
        words, labels = list(zip(*words_labels))
        return GenNERSample(words=words, labels=labels, id=id)

    def set_instruction_prompt(self, instruction_file: Path | str, label_list: list[str]):
        words = self.words
        labels = self.labels

        dataset = GNERDataset()
        dataset.config = GNERConfig(instruction_file=instruction_file)

        instruction = dataset._get_instruction()
        random.shuffle(label_list)
        instruction += f"\nUse the specific entity tags: {', '.join(label_list)} and O.\n"
        instruction += "Sentence: " + " ".join(words)
        self.instruction_inputs = instruction
        self.prompt_labels = dataset._generate_labeled_string(words, labels)

        return self


class GenNERSampleWrapper(BaseModel):
    id: str = None
    dataset: str = "unknown"
    split: str = "unknown"
    label_list: list[str] = None
    instance: GenNERSample

    def set_missing_values_by_instruction_prompt(self, path: Path | str = None):
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


class GenNERMetrics(BaseModel):
    mit_movie: float = Field(alias="eval_mit-movie_f1")
    mit_restaurant: float = Field(alias="eval_mit-restaurant_f1")
    crossner_ai: float = Field(alias="eval_crossner_ai_f1")
    crossner_literature: float = Field(alias="eval_crossner_literature_f1")
    crossner_music: float = Field(alias="eval_crossner_music_f1")
    crossner_politics: float = Field(alias="eval_crossner_politics_f1")
    crossner_science: float = Field(alias="eval_crossner_science_f1")
    target_average: float = Field(default=None)
    wiki_passage: float = Field(alias="eval_wiki_passage_from_zero-test_f1", default=0.0)
    epoch: float

    def calc(self):
        self.target_average = (self.mit_movie + self.mit_restaurant + self.crossner_ai + self.crossner_literature + self.crossner_music + self.crossner_politics + self.crossner_science) / 7
        return self


from . import gner
