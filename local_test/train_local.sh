#!/bin/sh

image=$1

# mount the input folder as if sagemaker was running the image
docker run -v $(pwd)/test_dir/input:/opt/ml/input --rm ${image} train
# --train input/data --val input/data