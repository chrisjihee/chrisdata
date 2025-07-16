# for mongodb
mkdir mongodb; cd mongodb; mkdir data log
if [ "$(uname)" = "Linux" ]; then
  wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-8.0.9.tgz
elif [ "$(uname)" = "Darwin" ]; then
  wget https://fastdl.mongodb.org/osx/mongodb-macos-arm64-8.0.9.tgz
fi
tar zxvf mongodb-*.tgz --strip-components 1
cd ..
