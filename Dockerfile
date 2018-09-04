FROM nvidia/cuda:9.2-devel-ubuntu16.04

WORKDIR /tmp

RUN apt-get update && \
    apt-get -y install build-essential libopencv-dev libatlas-base-dev libcurl4-openssl-dev libgtest-dev \
        libjemalloc-dev python3-dev unzip git wget curl nginx

# install pip
RUN apt-get update && apt-get install -y python-pip

# install requirements
ADD requirements.txt ./
RUN pip install -r requirements.txt

# add all python code to workdir
ADD *.py ./


# This is here to make our installed version of OpenCV work.
# https://stackoverflow.com/questions/29274638/opencv-libdc1394-error-failed-to-initialize-libdc1394
# TODO: Should we be installing OpenCV in our image like this? Is there another way we can fix this?
RUN ln -s /dev/null /dev/raw1394

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    MXNET_CUDNN_AUTOTUNE_DEFAULT=0

# Entrypoint script comes from sagemaker_container_support
ENTRYPOINT ["/usr/bin/python", "/usr/local/bin/entry.py"]