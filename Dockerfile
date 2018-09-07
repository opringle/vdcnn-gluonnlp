FROM nvidia/cuda:9.2-devel-ubuntu16.04

# Install python 3
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# install python dependencies
WORKDIR /
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Set env variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV MXNET_CUDNN_AUTOTUNE_DEFAULT=0
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY vdcnn /opt/program
WORKDIR /opt/program

ENTRYPOINT ["entry.py"]