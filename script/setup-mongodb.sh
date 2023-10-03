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
