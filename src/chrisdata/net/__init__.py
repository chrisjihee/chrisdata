from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin

from chrisbase.data import AppTyper

app = AppTyper()


@dataclass
class IPCheckResult(DataClassJsonMixin):
    _id: str
    uri: str
    ip: str
    status: int | None = None
    size: float | None = None
    text: str | None = None
    elapsed: float | None = None


import chrisdata.net.check
