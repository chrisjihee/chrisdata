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

# for mongodb
mkdir mongodb
cd mongodb || exit
mkdir data log
if [ "$(uname)" = "Linux" ]; then
  wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2004-7.0.1.tgz
elif [ "$(uname)" = "Darwin" ]; then
  wget https://fastdl.mongodb.org/osx/mongodb-macos-arm64-7.0.1.tgz
fi
tar zxvf mongodb-*.tgz --strip-components 1
cd ..

# for elasticsearch
mkdir elasticsearch
cd elasticsearch || exit
if [ "$(uname)" = "Linux" ]; then
  wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.10.1-linux-x86_64.tar.gz
elif [ "$(uname)" = "Darwin" ]; then
  wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.10.1-darwin-aarch64.tar.gz
fi
tar zxvf elasticsearch-*.tar.gz --strip-components 1
cd ..
