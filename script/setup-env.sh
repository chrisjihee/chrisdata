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
  aria2c https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2004-7.0.1.tgz
elif [ "$(uname)" = "Darwin" ]; then
  aria2c https://fastdl.mongodb.org/osx/mongodb-macos-arm64-7.0.1.tgz
fi
tar zxvf mongodb-*.tgz --strip-components 1
cd ..

# for elasticsearch8
mkdir elasticsearch8
cd elasticsearch8 || exit
if [ "$(uname)" = "Linux" ]; then
  aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.10.2-linux-x86_64.tar.gz
elif [ "$(uname)" = "Darwin" ]; then
  aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.10.2-darwin-aarch64.tar.gz
fi
tar zxf elasticsearch-*.tar.gz --strip-components 1
sed -i '' 's/#http.port: 9200/http.port: 9810/g' ./config/elasticsearch.yml
echo "xpack.security.enabled: true" >> ./config/elasticsearch.yml
#./bin/elasticsearch
#./bin/elasticsearch-setup-passwords auto
#./bin/elasticsearch-reset-password -u elastic
cd ..

# for elasticsearch7
mkdir elasticsearch7
cd elasticsearch7 || exit
if [ "$(uname)" = "Linux" ]; then
  aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-linux-x86_64.tar.gz
elif [ "$(uname)" = "Darwin" ]; then
  aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-darwin-aarch64.tar.gz
fi
tar zxf elasticsearch-*.tar.gz --strip-components 1
sed -i '' 's/#http.port: 9200/http.port: 9717/g' ./config/elasticsearch.yml
echo "xpack.security.enabled: true" >> ./config/elasticsearch.yml
#./bin/elasticsearch
#./bin/elasticsearch-setup-passwords auto
cd ..
