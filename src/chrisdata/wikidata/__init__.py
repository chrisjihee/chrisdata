import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel
from qwikidata.claim import WikidataClaim
from qwikidata.datavalue import _DATAVALUE_TYPE_TO_CLASS, WikidataDatavalue, Time, Quantity, WikibaseEntityId, String, MonolingualText
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme, ClaimsMixin
from qwikidata.snak import WikidataSnak
from qwikidata.typedefs import LanguageCode

from chrisbase.data import AppTyper, TypedData, IOArguments
from chrisbase.data import MongoStreamer
from chrisbase.util import SP, US

app = AppTyper()
logger = logging.getLogger(__name__)


def datavalue_dict_to_obj(x: dict) -> WikidataDatavalue:
    return _DATAVALUE_TYPE_TO_CLASS[x['type']](x)


def datavalue_dict(x: WikidataSnak):
    return x.datavalue._datavalue_dict


class ClaimMixinEx(ClaimsMixin):
    def get_claims(self, args: IOArguments, truthy: bool) -> list[dict]:
        claims = list()
        claim_groups = self.get_claim_groups() if not truthy else self.get_truthy_claim_groups()
        for claim_group in claim_groups.values():
            for claim in claim_group:
                claim: WikidataClaim = claim
                if args.env.debugging:
                    logger.info(f"+ {claim.mainsnak.property_id} = {datavalue_dict(claim.mainsnak)}")  # claim.mainsnak.datavalue.value == datavalue_dict_to_obj(datavalue_dict(claim.mainsnak))
                if claim.mainsnak.snaktype == "value" and claim.mainsnak.datavalue is not None:
                    qualifiers = list()
                    for qualifier_group in claim.qualifiers.values():
                        for qualifier in qualifier_group:
                            if qualifier.snak.snaktype == "value" and qualifier.snak.datavalue is not None:
                                if args.env.debugging:
                                    logger.info(f"  - {qualifier.snak.property_id} = {datavalue_dict(qualifier.snak)}")  # qualifier.snak.datavalue == datavalue_dict_to_obj(datavalue_dict(qualifier.snak))
                                qualifiers.append({"property": qualifier.snak.property_id, "datavalue": datavalue_dict(qualifier.snak)})
                    claims.append({"property": claim.mainsnak.property_id, "datavalue": datavalue_dict(claim.mainsnak), "qualifiers": qualifiers})
                if args.env.debugging:
                    logger.info('')
        return claims


class WikidataPropertyEx(WikidataProperty, ClaimMixinEx):
    pass


class WikidataItemEx(WikidataItem, ClaimMixinEx):
    def get_wiki_title(self, lang: LanguageCode) -> str:
        wikiname = f"{lang.lower()}wiki"
        if (
                isinstance(self._entity_dict["sitelinks"], dict)
                and wikiname in self._entity_dict["sitelinks"]
        ):
            return self._entity_dict["sitelinks"][wikiname]["title"]
        else:
            return ""


class WikidataLexemeEx(WikidataLexeme, ClaimMixinEx):
    def get_gloss(self, lang: LanguageCode):
        res = list()
        for sense in self.get_senses():
            res.append({"sense_id": sense.sense_id, "gloss": sense.get_gloss(lang)})
        return res


WIKIDATA_ID_PATTERN = re.compile(r"^([A-Z])([0-9]+)$")


def split_wikidata_id(x: str):
    match = WIKIDATA_ID_PATTERN.fullmatch(x)
    if match:
        try:
            return match.group(1), int(match.group(2))
        except Exception as e:
            raise ValueError(f"No numeric part: x={x}: [{type(e).__name__}] {e}")
    else:
        raise ValueError(f"Not matched Wikidata ID: x={x}")


def norm_wikidata_id(x: str):
    try:
        pre, post = split_wikidata_id(x)
        return f"{pre}{post:09d}"
    except Exception as e:
        logger.error(f"Error on to_normalized_id(x={x}): [{type(e).__name__}] {e}")
        return None


@dataclass
class WikidataUnit(TypedData):
    _id: str
    id: str
    ns: int
    type: str
    time: str
    label1: str | None = None
    label2: str | None = None
    title1: str | None = None
    title2: str | None = None
    alias1: list[str] = field(default_factory=list)
    alias2: list[str] = field(default_factory=list)
    descr1: str | None = None
    descr2: str | None = None
    claims: list[dict] = field(default_factory=list)


class Entity(BaseModel):
    id: str
    label1: str
    label2: str
    # alias1: list = field(default_factory=list)
    # alias2: list = field(default_factory=list)
    # descr1: str | None = None
    # descr2: str | None = None
    title1: str | None = None
    title2: str | None = None

    @property
    def title(self) -> str:
        return self.title1 or self.title2

    @property
    def label(self) -> str:
        return self.label1 or self.label2 or self.id

    @property
    def source(self) -> str:
        return f"https://www.wikidata.org/wiki/{self.id}"

    @property
    def document(self) -> str:
        if self.title1:
            return f"https://ko.wikipedia.org/wiki/{self.title1.replace(SP, US)}"
        elif self.title2:
            return f"https://en.wikipedia.org/wiki/{self.title2.replace(SP, US)}"
        else:
            return self.source

    @staticmethod
    def from_wikidata_unit(unit: WikidataUnit) -> "Entity":
        return Entity.model_validate(unit.to_dict())


entity_cache: dict[str, Entity | None] = dict()


class Relation(BaseModel):
    id: str
    label1: str
    label2: str
    # alias1: list[str] = field(default_factory=list)
    # alias2: list[str] = field(default_factory=list)
    # descr1: str
    # descr2: str
    datatype: str | None = None
    property_count: int = -1
    qualifier_count: int = -1
    reference_count: int = -1

    @property
    def label(self) -> str:
        return self.label1 or self.label2 or self.id

    @property
    def source(self) -> str:
        return f"https://www.wikidata.org/wiki/Property:{self.id}"


class DataValue(BaseModel):
    type: str
    string: str
    entity: Entity | None = None
    # raw_data: str | dict | None = None
    # entity_link: str | None = None


QUANTITY_UNIT_PATTERN = re.compile(
    r"""
    http://www.wikidata.org/entity/(?P<id>[A-Z]\d+)
    """,
    re.VERBOSE,
)


def get_entity(_id: str, reader: MongoStreamer) -> Entity | None:
    if _id not in entity_cache:
        row: dict | None = reader.table.find_one({'_id': _id})
        if not row:
            entity_cache[_id] = None
        else:
            entity_cache[_id] = Entity.model_validate(row)
    return entity_cache[_id]


def get_wikidata_entity(datavalue: WikibaseEntityId, reader: MongoStreamer) -> DataValue:
    entity: Entity | None = get_entity(norm_wikidata_id(datavalue.value['id']), reader)
    if not entity:
        return DataValue(
            type=type(datavalue).__name__,
            string=datavalue.value['id'],
            # raw_data=datavalue.value,
        )
    else:
        return DataValue(
            type=type(datavalue).__name__,
            string=entity.title or entity.label,
            entity=entity,
            # raw_data=datavalue.value,
        )


def get_quantity(datavalue: Quantity, reader: MongoStreamer) -> DataValue:
    if datavalue.value['unit'] == '1':
        return DataValue(
            type=type(datavalue).__name__,
            string=f"{datavalue.value['amount']}",
            # raw_data=datavalue.value,
        )
    else:
        match = QUANTITY_UNIT_PATTERN.fullmatch(datavalue.value['unit'])
        if not match:
            return DataValue(
                type=type(datavalue).__name__,
                string=f"{datavalue.value['amount']} {datavalue.value['unit']}",
                # raw_data=datavalue.value,
            )
        else:
            unit: Entity | None = get_entity(norm_wikidata_id(match.group('id')), reader)
            return DataValue(
                type=type(datavalue).__name__,
                string=f"{datavalue.value['amount']} {unit.label}",
                entity=unit,
                # raw_data=datavalue.value,
            )


def get_time(datavalue: Time) -> DataValue:
    parts = datavalue.get_parsed_datetime_dict()
    if datavalue.value['precision'] <= 9:
        return DataValue(
            type=type(datavalue).__name__,
            string=f"{parts['year']:04d}년",
            # raw_data=datavalue.value,
        )
    elif datavalue.value['precision'] == 10:
        return DataValue(
            type=type(datavalue).__name__,
            string=f"{parts['year']:04d}-{parts['month']:02d}",
            # raw_data=datavalue.value,
        )
    elif datavalue.value['precision'] == 11:
        return DataValue(
            type=type(datavalue).__name__,
            string=f"{parts['year']:04d}-{parts['month']:02d}-{parts['day']:02d}",
            # raw_data=datavalue.value,
        )
    else:
        raise ValueError(f"Unknown precision: {datavalue.value['precision']}")


def get_monolingual_text(datavalue: MonolingualText) -> DataValue:
    return DataValue(
        type=type(datavalue).__name__,
        string=f"{datavalue.value['text']}({datavalue.value['language']})",
        # raw_data=datavalue.value,
    )


def datavalue_to_object(datavalue: dict, reader: MongoStreamer) -> DataValue:
    datavalue: WikidataDatavalue = datavalue_dict_to_obj(datavalue)
    if isinstance(datavalue, WikibaseEntityId):
        return get_wikidata_entity(datavalue, reader)
    elif isinstance(datavalue, Quantity):
        return get_quantity(datavalue, reader)
    elif isinstance(datavalue, Time):
        return get_time(datavalue)
    elif isinstance(datavalue, MonolingualText):
        return get_monolingual_text(datavalue)
    elif isinstance(datavalue, String):
        return DataValue(
            type=type(datavalue).__name__,
            string=datavalue.value,
            # raw_data=datavalue.value,
        )
    else:
        return DataValue(
            type=type(datavalue).__name__,
            string=str(datavalue),
            # raw_data=datavalue.value,
        )


class StatementValue(BaseModel):
    value: DataValue
    qualifiers: dict[str, str]


class Statement(BaseModel):
    relation: Relation
    values: list[StatementValue]


class SubjectInfo(BaseModel):
    subject: Entity
    num_statements: int
    num_qualifiers: int
    document_length: int


class SubjectStatements(SubjectInfo):
    statements: list[Statement]


@dataclass
class EntityInWiki(TypedData):
    entity: str
    hits: int
    score: float


@dataclass
class TripleInWiki(TypedData):
    entity1: EntityInWiki
    entity2: EntityInWiki
    relation: Relation
    hits: int
    score: float = field(default=0.0)
    pmi: float = field(default=0.0)

    @staticmethod
    def calc_pmi(h_xy: float, h_x: float, h_y: float, n: int = 10000, e: float = 0.0000001) -> float:
        p_xy = h_xy / n
        p_x = h_x / n
        p_y = h_y / n
        return math.log2((p_xy + e) / ((p_x * p_y) + e))

    def __post_init__(self):
        if self.score is None:
            self.score = 0.0
        self.pmi = self.calc_pmi(self.hits, self.entity1.hits, self.entity2.hits)


@dataclass
class SingleTriple(TypedData):
    entity1: EntityInWiki
    entity2: EntityInWiki
    relation: Relation
    hits: int = field(default=0)
    pmi: float = field(default=0.0)
    score: float = field(default=0.0)

    @staticmethod
    def calc_pmi(h_xy: float, h_x: float, h_y: float, n: int = 10000, e: float = 0.0000001) -> float:
        p_xy = h_xy / n
        p_x = h_x / n
        p_y = h_y / n
        return math.log2((p_xy + e) / ((p_x * p_y) + e))

    def __post_init__(self):
        if self.score is None:
            self.score = 0.0
        self.pmi = self.calc_pmi(self.hits, self.entity1.hits, self.entity2.hits)


@dataclass
class DoubleTriple(TypedData):
    entity1: EntityInWiki
    entity2: EntityInWiki
    entity3: EntityInWiki
    relation1: Relation
    relation2: Relation
    hits1: int = field(default=0)
    hits2: int = field(default=0)
    pmi1: float = field(default=0.0)
    pmi2: float = field(default=0.0)
    score1: float = field(default=0.0)
    score2: float = field(default=0.0)

    @staticmethod
    def from_triples(
            triple1: "SingleTriple",
            triple2: "SingleTriple"
    ) -> Optional["DoubleTriple"]:
        if (triple2.entity2.entity != triple1.entity1.entity
                and triple2.relation.id != triple1.relation.id):
            return DoubleTriple(
                entity1=triple1.entity1,
                entity2=triple1.entity2,
                entity3=triple2.entity2,
                relation1=triple1.relation,
                relation2=triple2.relation,
                hits1=triple1.hits,
                hits2=triple2.hits,
                score1=triple1.score,
                score2=triple2.score,
                pmi1=triple1.pmi,
                pmi2=triple2.pmi,
            )
        else:
            return None


from . import parse_wikidata, filter_wikidata, convert_wikidata, restore_wikidata
