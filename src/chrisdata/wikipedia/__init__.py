import logging
from dataclasses import field

from pydantic import BaseModel

from chrisbase.data import AppTyper

app = AppTyper()
logger = logging.getLogger(__name__)


class WikipediaProcessResult(BaseModel):
    _id: int
    query: str
    title: str | None = None
    page_id: int | None = None
    section_list: list = field(default_factory=list)
    passage_list: list = field(default_factory=list)


from . import convert_wikipedia
