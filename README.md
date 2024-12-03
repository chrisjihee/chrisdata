# chrisdata

Data processing tools for data analysis


## Installation

1. Install Miniforge
    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

2. Clone the repository
    ```bash
    rm -rf chrisdata*
    git clone git@github.com:chrisjihee/chrisdata.git
    cd chrisdata*
    ```

3. Create a new environment
    ```bash
    mamba create -n chrisdata python=3.11 -y
    mamba activate chrisdata
    ```

4. Install the required packages
    ```bash
    pip install -U -e .
    rm -rf chrisbase*; git clone git@github.com:chrisjihee/chrisbase.git
    pip install -U -e chrisbase*
    pip list | grep -E "mongo|search|Wiki|wiki|json|pydantic|chris|Flask"
    ```

5. Install MongoDB
    ```bash
    mkdir mongodb; cd mongodb; mkdir data log
    if [ "$(uname)" = "Linux" ]; then
      aria2c https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-8.0.0.tgz
    elif [ "$(uname)" = "Darwin" ]; then
      aria2c https://fastdl.mongodb.org/osx/mongodb-macos-arm64-8.0.0.tgz
    fi
    tar zxvf mongodb-*.tgz --strip-components=1
    cd ..
    ```

6. Run MongoDB
    ```bash
    cd mongodb
    bin/mongod --config ../cfg/mongod-8800.yaml
    cd ..
    ```
    ```bash
    cd mongodb
    bin/mongod --config ../cfg/mongod-8801.yaml
    cd ..
    ```

7. Install Elasticsearch
    ```bash
    mkdir elasticsearch7; cd elasticsearch7
    if [ "$(uname)" = "Linux" ]; then
      aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-linux-x86_64.tar.gz
    elif [ "$(uname)" = "Darwin" ]; then
      aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-darwin-aarch64.tar.gz
    fi
    tar zxf elasticsearch-*.tar.gz --strip-components 1
    sed -i '' 's/#http.port: 9200/http.port: 9717/g' ./config/elasticsearch.yml
    echo "xpack.security.enabled: true" >> ./config/elasticsearch.yml
    cd ..
    ```

8. Link input data
    ```bash
    cd input
    ln -s /mnt/geo/data/wikidata .
    ln -s /mnt/geo/data/wikipedia .
    cd ..
    ```


## Execution

1. Show help
    ```bash
    python -m chrisdata.cli --help
    ```

    ```bash
    python -m chrisdata.cli wikipedia --help
    ```

    ```bash
    python -m chrisdata.cli wikidata --help
    ```

2. Run command
  * To convert Wikipedia articles
    ```bash
    python -m chrisdata.cli wikipedia convert
    ```

  * To parse Wikidata dump
    ```bash
    python -m chrisdata.cli wikidata parse
    ```

  * To filter Wikidata entities
    ```bash
    python -m chrisdata.cli wikidata filter
    ```

  * To convert Wikidata entities
    ```bash
    python -m chrisdata.cli wikidata convert
    ```


## Reference

* https://pypi.org/project/chrisdata/
* https://github.com/chrisjihee/chrisdata/
* https://github.com/chrisjihee/chrisbase/
