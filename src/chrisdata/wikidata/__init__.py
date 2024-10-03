import re
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from qwikidata.claim import WikidataClaim
from qwikidata.datavalue import _DATAVALUE_TYPE_TO_CLASS, WikidataDatavalue
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme, ClaimsMixin
from qwikidata.snak import WikidataSnak
from qwikidata.typedefs import LanguageCode

from chrisbase.data import AppTyper, TypedData, IOArguments

app = AppTyper()
logger = logging.getLogger(__name__)


def datavalue_dict_to_obj(x: dict) -> WikidataDatavalue:
    return _DATAVALUE_TYPE_TO_CLASS[x['type']](x)


def datavalue_dict(x: WikidataSnak):
    return x.datavalue._datavalue_dict


class ClaimMixinEx(ClaimsMixin):
    def get_claims(self, args: IOArguments) -> list[dict]:
        claims = list()
        claim_groups = self.get_claim_groups() if not args.option.truthy else self.get_truthy_claim_groups()
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


wikidata_unit_id = re.compile(r"^([A-Z])([0-9]+)$")


def split_wikidata_id(x: str):
    match = wikidata_unit_id.fullmatch(x)
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


@dataclass
class Entity(TypedData):
    id: str
    label1: str
    label2: str
    title1: str | None = None
    title2: str | None = None

    # alias1: list = field(default_factory=list)
    # alias2: list = field(default_factory=list)
    # descr1: str | None = None
    # descr2: str | None = None

    @staticmethod
    def from_wikidata_unit(unit: WikidataUnit) -> "Entity":
        return Entity.from_dict(unit.to_dict())

    def __str__(self):
        return f"{self.id}[{self.title1 or self.title2}]"


@dataclass
class Relation(TypedData):
    id: str
    label1: str
    label2: str
    descr1: str
    descr2: str
    alias1: list[str] = field(default_factory=list)
    alias2: list[str] = field(default_factory=list)
    datatype: str | None = None
    property_count: int = -1
    qualifier_count: int = -1
    reference_count: int = -1

    def __str__(self):
        return f"{self.id}[{self.label2}]"


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


import chrisdata.wikidata.parse_wikidata
import chrisdata.wikidata.convert_wikidata
import chrisdata.wikidata.restore_wikidata
