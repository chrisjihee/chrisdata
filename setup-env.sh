#!/bin/bash
conda create -n WiseData-2023.07 python=3.10 -y
conda activate WiseData-2023.07
pip install -r requirements.txt
pip list

rm -rf chrisbase chrislab
git clone git@github.com:chrisjihee/chrisbase.git
git clone git@github.com:chrisjihee/chrislab.git
pip install --editable chrisbase
pip install --editable chrislab

rm -rf Wikipedia-API-*
pip download --no-binary :all: --no-deps Wikipedia-API==0.6.0
tar zxf *.tar.gz
rm *.tar.gz
pip install --editable Wikipedia-API-0.6.0

wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2004-6.0.8.tgz
tar zxf mongodb-linux-x86_64-ubuntu2004-6.0.8.tgz
mv mongodb-linux-x86_64-ubuntu2004-6.0.8 mongodb
mkdir -p mongodb/data
