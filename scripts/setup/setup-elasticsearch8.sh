# for elasticsearch8
mkdir elasticsearch8; cd elasticsearch8
if [ "$(uname)" = "Linux" ]; then
  aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.10.2-linux-x86_64.tar.gz
elif [ "$(uname)" = "Darwin" ]; then
  aria2c https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.10.2-darwin-aarch64.tar.gz
fi
tar zxf elasticsearch-*.tar.gz --strip-components 1
sed -i '' 's/#http.port: 9200/http.port: 9810/g' ./config/elasticsearch.yml
echo "xpack.security.enabled: true" >> ./config/elasticsearch.yml
cd ..

#bash script/run-elasticsearch8.sh
#bash script/setup-elasticsearch8-pw.sh
