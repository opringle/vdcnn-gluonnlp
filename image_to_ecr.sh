#!/usr/bin/env bash

usage="
USAGE: $(basename "$0") [-h] profile region
program to retrieve and split classifier data for customer model
positional arguments:
    profile:   AWS profile
    region:    AWS region

optional positional arguments:
    -h:         show this help text"

while getopts ':hs-:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
  esac
done



registry_url=$(aws ecr get-authorization-token --profile "$1" --region "$2" | \
               python -c "import sys, json; print(json.load(sys.stdin)['authorizationData'][0]['proxyEndpoint'][8:])")


# log into docker
eval $(aws ecr get-login --profile "$1" --region "$2" --no-include-email)

# build image
docker build . -t "$registry_url"/gluonnlp-cuda9.2

# push to ECR
docker push "$registry_url"/gluonnlp-cuda9.2
