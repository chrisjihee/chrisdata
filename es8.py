from datetime import datetime
from elasticsearch import Elasticsearch

# Create a connection
es = Elasticsearch(hosts="http://localhost:9810",
                   basic_auth=("elastic", "cIrEP5OCwTLn0QIQwnsA"))
if not es.ping():
    raise ValueError("Connection failed")
print(es.info())

# Create an index
index_name = "es8_test"
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
es.indices.create(index=index_name)
print(es.indices.get_alias(index="e*"))

# Index a document
doc = {
    "agency": "연합뉴스",
    "text": "보리스 옐친(68) 러시아 대통령은 31일 사임을 발표하고 블라디미르 푸틴 총리를 대통령 직무대행으로 임명했다.",
    "date": "1999.12.31.",
}
print(es.index(index=index_name, document=doc))
es.indices.refresh(index=index_name)

# Search for the document
res = es.search(index=index_name, query={"match_all": {}})
print("Got %d Hits:" % res['hits']['total']['value'])
for hit in res['hits']['hits']:
    print(hit["_source"])
