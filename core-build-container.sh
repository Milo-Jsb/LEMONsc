export UID=$(id -u)
export GID=$(id -g)

docker build \
    --build-arg HOST_UID=$UID \
    --build-arg HOST_GID=$GID \
    -t core .