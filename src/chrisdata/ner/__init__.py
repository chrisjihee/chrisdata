import logging

from chrisbase.data import AppTyper
from pydantic import BaseModel

app = AppTyper()
logger = logging.getLogger(__name__)


class WebResponse(BaseModel):
    id: str
    uri: str
    ip: str | None = None
    status: int | None = None
    size: float | None = None
    text: str | None = None
    elapsed: float | None = None


from . import convert_ner
