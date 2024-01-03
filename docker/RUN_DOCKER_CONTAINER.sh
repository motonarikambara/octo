#!/bin/sh
xhost +

if [ ! -z $1 ]; then
  TAG_NAME=$1
else
  TAG_NAME="latest"
fi

export WANDB_API_KEY=`cat wandb_key.txt`

docker-compose -p octo up ${TAG_NAME} &
