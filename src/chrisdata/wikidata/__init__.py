import logging
import math
from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin
from qwikidata.claim import WikidataClaim
from qwikidata.datavalue import _DATAVALUE_TYPE_TO_CLASS
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme, ClaimsMixin
from qwikidata.snak import WikidataSnak
from qwikidata.typedefs import LanguageCode

from chrisbase.data import AppTyper, TypedData

app = AppTyper()
logger = logging.getLogger(__name__)


def datavalue_dict_to_obj(x: dict):
    return _DATAVALUE_TYPE_TO_CLASS[x['type']](x)


def datavalue_dict(x: WikidataSnak):
    return x.datavalue._datavalue_dict


class ClaimMixinEx(ClaimsMixin):
    def get_claims(self, args: "ParseArguments") -> list[dict]:
        claims = list()
        claim_groups = self.get_claim_groups() if not args.filter.truthy else self.get_truthy_claim_groups()
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


@dataclass
class WikidataUnit(DataClassJsonMixin):
    _id: str
    ns: int
    type: str
    time: str
    label1: str | None = None
    label2: str | None = None
    title1: str | None = None
    title2: str | None = None
    alias1: list = field(default_factory=list)
    alias2: list = field(default_factory=list)
    descr1: str | None = None
    descr2: str | None = None
    claims: list[dict] = field(default_factory=list)


@dataclass
class EntityInWiki(TypedData):
    entity: str
    hits: int
    score: float


@dataclass
class Relation(TypedData):
    id: str
    label1: str
    label2: str

    def __str__(self):
        return f"{self.id}[{self.label2}]"


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


import chrisdata.wikidata.parse_wikidata
import chrisdata.wikidata.restore_wikidata