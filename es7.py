from elasticsearch import Elasticsearch, helpers

# Establish a connection to the Elasticsearch cluster.
# If Elasticsearch is running on a different host or port, modify the line below accordingly.
es = Elasticsearch("http://localhost:9200"
                   # , basic_auth=("elastic", "viygZnjDVBegPHgFXk2Y")
                   )

# Define an index name
index_name = "test_index"

# Delete the index if it exists
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# Create an index
es.indices.create(index=index_name)

# Index a document
doc = {
    "name": "John Doe",
    "age": 30,
    "interests": ["coding", "hiking"]
}
es.index(index=index_name, id=1, document=doc)

# Refresh the index to make sure the document is searchable.
es.indices.refresh(index=index_name)

# Search for the document
response = es.search(index=index_name, query={
    "match": {
        "interests": "coding"
    }
})

# Output the search results
print("Search Results:")
for hit in response['hits']['hits']:
    print(hit['_source'])
