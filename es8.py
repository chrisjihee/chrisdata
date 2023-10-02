from datetime import datetime
from elasticsearch import Elasticsearch

# Create a connection
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Ping the Elasticsearch Cluster
if not es.ping():
    raise ValueError("Connection failed")

# Create an index
index_name = "my_index"
if not es.indices.exists(index_name):
    es.indices.create(index=index_name)

# Index a document
doc = {
    'author': 'john doe',
    'text': 'Elasticsearch: cool. bonsai cool.',
    'timestamp': datetime.now(),
}
res = es.index(index=index_name, doc_type='_doc', body=doc)
print(res['result'])

# Search for the document
res = es.search(index=index_name, body={"query": {"match_all": {}}})
print("Got %d Hits:" % res['hits']['total']['value'])
for hit in res['hits']['hits']:
    print(hit["_source"])
