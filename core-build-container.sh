export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

docker build \
    --build-arg HOST_UID=$HOST_UID \
    --build-arg HOST_GID=$HOST_GID \
    -t core .
