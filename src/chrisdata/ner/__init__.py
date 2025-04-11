import logging
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional

from pydantic import BaseModel, Field

from chrisbase.data import AppTyper
from .gner_dataset import GNERDataset, GNERConfig

app = AppTyper()
logger = logging.getLogger(__name__)

# NER patterns
bio_tag_pattern = re.compile(r"([^ ]+)\(([BIO](-[A-Za-z ]+)?)\)")
entity_text_pattern = re.compile(r'[A-Za-z ]+')

# Wiki patterns
file_pattern = re.compile(r"\[\[File:([^]]+)]]")
space_pattern = re.compile(r"  +")
link2_pattern = re.compile(r"\[\[([^|\]]+)\|([^]]+)]]")
link1_pattern = re.compile(r"\[\[([^]]+)]]")
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


class GoLLIESample(BaseModel):
    ids: list[str | int]
    task_id: str
    scorer_cls: str
    labels: str
    text: str
    unlabelled_sentence: str


class GenSeq2SeqSample(BaseModel):
    id: str = None
    prompt_labels: str = None
    instruction_inputs: str = None
    prediction_output: Optional[str] = None
    prediction_outputs: Optional[List[str]] = None


class GenNERSample(GenSeq2SeqSample):
    id: Optional[str] = None
    group: Optional[str] = None
    words: list[str] = None
    labels: list[str] = None
    target_index: Optional[int] = None
    target_label: Optional[str] = None

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

    # extract words and corresponding labels in the generation texts
    @staticmethod
    def extract(preds_text):
        pattern = r'\(B-.*?\)|\(I-.*?\)|\(O\)'
        words, labels, pre_bound = [], [], 0
        for label_span in re.finditer(pattern, preds_text):
            l, r = label_span.span()
            word, label = preds_text[pre_bound: l], preds_text[l + 1: r - 1]
            if word.strip() != '':
                words.append(word.strip())
                labels.append(label.strip())
            pre_bound = r
        return words, labels

    @staticmethod
    def get_prompt_labels(words: list[str], labels: list[str]) -> str:
        return GNERDataset._generate_labeled_string(words, labels)

    def set_prompt_labels(self):
        words = self.words
        labels = self.labels

        self.prompt_labels = GNERDataset._generate_labeled_string(words, labels)

    def set_instruction_prompt(self, instruction_file: Path | str, label_list: list[str]):
        words = self.words
        labels = self.labels

        dataset = GNERDataset()
        instruction_file = Path(instruction_file)
        if instruction_file.suffix == ".json":
            dataset.config = GNERConfig(instruction_file=instruction_file)
        else:
            dataset.config = GNERConfig()
            dataset.config.instructions = [instruction_file.read_text()]

        instruction = dataset._get_instruction()
        random.shuffle(label_list)
        instruction += f"\nUse the specific entity tags: {', '.join(label_list)} and O.\n"
        instruction += "Sentence: " + " ".join(words)
        self.instruction_inputs = instruction
        self.prompt_labels = GNERDataset._generate_labeled_string(words, labels)

        return self


class GenSeq2SeqSampleWrapper(BaseModel):
    id: Optional[str] = None
    dataset: str = "unknown"
    split: str = "unknown"
    instance: GenSeq2SeqSample


class GenNERSampleWrapper(GenSeq2SeqSampleWrapper):
    instance: GenNERSample
    label_list: list[str] = None
    prediction: Optional[str] = None  # for original GNER version

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


class GenNERSampleEntitySpan(BaseModel):
    entity: str
    span: list[int]


class Message(BaseModel):
    role: str
    content: str


class KGGenerationMessage(BaseModel):
    dataset_name: str
    entity: str
    triples_by_human: List[Tuple[str, str, str]]
    generation_level: int
    generation_messages: List[Message]


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
