From registry.docker-cn.com/tensorflow/tensorflow:1.8.0

RUN apt-get update && apt-get install -y python-tk

RUN pip install h5py==2.8.0rc1

WORKDIR /tensorflow-101

ADD . /tensorflow-101