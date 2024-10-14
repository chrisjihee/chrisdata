from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin


@dataclass
class WikipediaProcessResult(DataClassJsonMixin):
    _id: int
    query: str
    title: str | None = None
    page_id: int | None = None
    section_list: list = field(default_factory=list)
    passage_list: list = field(default_factory=list)
