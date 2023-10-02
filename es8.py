from datetime import datetime
from elasticsearch import Elasticsearch

# Create a connection
es = Elasticsearch("http://localhost:9717", basic_auth=("elastic", "RGzYkwogSi2jg9oE1oI6"))

# Ping the Elasticsearch Cluster
if not es.ping():
    raise ValueError("Connection failed")

# Create an index
index_name = "my_index"
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
es.indices.create(index=index_name)

# Index a document
doc = {
    'author': 'john doe',
    'text': 'Elasticsearch: cool. bonsai cool.',
    'timestamp': datetime.now(),
}
print(es.index(index=index_name, document=doc))
es.indices.refresh(index=index_name)

# Search for the document
res = es.search(index=index_name, query={"match_all": {}})
print("Got %d Hits:" % res['hits']['total']['value'])
for hit in res['hits']['hits']:
    print(hit["_source"])
