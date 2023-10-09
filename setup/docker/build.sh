#!/bin/bash

echo 'Building Dockerfile with image name jaxrl'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    --build-arg REQS="$(cat ../requirements-base.txt ../requirements-gpu.txt | tr '\n' ' ')" \
    -t jaxrl \
    .
