#!/bin/sh

if [ ! -z $1 ]; then
  TAG_NAME=$1
else
  TAG_NAME="latest"
fi

docker-compose -p octo build ${TAG_NAME}
