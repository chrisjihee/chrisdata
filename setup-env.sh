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

pip download --no-binary :all: --no-deps Wikipedia-API==0.6.0
rm -rf Wikipedia-API-*
tar zxf *.tar.gz
rm *.tar.gz
pip install --editable Wikipedia-API-0.6.0
