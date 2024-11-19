# for mongodb
mkdir mongodb; cd mongodb; mkdir data log
if [ "$(uname)" = "Linux" ]; then
  aria2c https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-7.0.14.tgz
elif [ "$(uname)" = "Darwin" ]; then
  aria2c https://fastdl.mongodb.org/osx/mongodb-macos-arm64-7.0.14.tgz
fi
tar zxvf mongodb-*.tgz --strip-components 1
cd ..
