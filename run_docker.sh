#!/bin/bash
WANDB_API_KEY=$(cat ./docker/wandb_key)
git pull

script_and_args="${@:2}"
gpu=$1
echo "Launching container diffrl_$gpu on GPU $gpu"
docker run \
    --env CUDA_VISIBLE_DEVICES=$gpu \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/rl-jax-template \
    --name jaxrl_$gpu \
    --user $(id -u) \
    --rm \
    -d \
    -t jaxrl \
    /bin/bash -c "$script_and_args"
