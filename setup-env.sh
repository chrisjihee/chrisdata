#!/bin/bash
# for running program
conda create -n WiseData-2023.08 python=3.10 -y
conda activate WiseData-2023.08
pip install -r requirements.txt
pip list

# for developing library
rm -rf chrisbase* chrislab*
git clone https://github.com/chrisjihee/chrisbase.git
git clone https://github.com/chrisjihee/chrislab.git
pip install --editable chrisbase
pip install --editable chrislab

# MongoDB
mkdir mongodb
cd mongodb || exit
mkdir data log
if [ "$(uname)" = "Darwin" ]; then
  wget https://fastdl.mongodb.org/osx/mongodb-macos-arm64-7.0.1.tgz
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2004-7.0.1.tgz
fi
tar zxvf mongodb-*.tgz --strip-components 1
cd ..
