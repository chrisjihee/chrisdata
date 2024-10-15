import logging

from chrisbase.data import AppTyper
from pydantic import BaseModel

app = AppTyper()
logger = logging.getLogger(__name__)


class WikipediaCrawlResult(BaseModel):
    _id: int
    query: str
    title: str | None = None
    page_id: int | None = None
    section_list: list = []
    passage_list: list = []


class WikipediaDocument(BaseModel):
    title: str
    length: int
    page_id: int
    sections: list[str]

    @property
    def id(self):
        return self.title


class WikipediaStat(BaseModel):
    title: str
    length: int
    page_id: int

    @property
    def id(self):
        return self.title


from . import convert_wikipedia
