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
cd ..

#bash script/run-elasticsearch7.sh
#bash script/setup-elasticsearch7-pw.sh
