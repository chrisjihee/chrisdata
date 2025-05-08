from __future__ import annotations

import logging

from pydantic import BaseModel

from chrisbase.data import AppTyper

app = AppTyper()
logger = logging.getLogger(__name__)


class WikipediaCrawlResult(BaseModel):
    query: str
    _id: int | None = None
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
from . import crawl_wikipedia
from . import parse_wikipedia
