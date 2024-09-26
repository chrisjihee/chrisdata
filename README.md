# WiseData

Data processing tools for data analysis


## Installation

1. Install Miniforge and create a new environment
    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    mamba create -n WiseData python=3.11 -y; mamba activate WiseData
    ```

2. Clone the repository
    ```bash
    rm -rf WiseData*
    git clone git@github.com:chrisjihee/WiseData-2023.08.git
    cd WiseData*
    ```

3. Install the required packages
    ```bash
    pip install -U -e .
    rm -rf chrisbase*; git clone git@github.com:chrisjihee/chrisbase.git
    pip install -U -e chrisbase*
    ```

4. List installed packages
    ```bash
    pip list | grep -E "search|Wiki|wiki|json|chris|Data"
    ```

5. Install MongoDB
    ```bash
    mkdir mongodb; cd mongodb; mkdir data log
    if [ "$(uname)" = "Linux" ]; then
      aria2c https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-7.0.14.tgz
    elif [ "$(uname)" = "Darwin" ]; then
      aria2c https://fastdl.mongodb.org/osx/mongodb-macos-arm64-7.0.14.tgz
    fi
    tar zxvf mongodb-*.tgz --strip-components 1
    cd..
    ```

6. Run MongoDB
    ```bash
    mongodb/bin/mongod --config cfg/mongod.yaml
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


## Execution

1. Show help
    ```bash
    chrisdata --help
    ```

2. Run command
  * To check local IP addresses
    ```bash
    chrisdata hello-chrisdata
    ```

  * To crawl Wikipedia articles
    ```bash
    chrisdata hello-chrisdata
    ```

  * To parse Wikipedia articles
    ```bash
    chrisdata hello-chrisdata
    ```

  * To parse Wikidata dump
    ```bash
    chrisdata hello-chrisdata
    ```


## Reference

* https://github.com/chrisjihee/WiseData-2023.08/
* https://github.com/chrisjihee/chrisbase/
