FROM python:2.7

ENV DEBIAN_FRONTEND=noninteractive

RUN set -ex; \
  apt-get update; \
  apt-get -y install python-opencv; \
  apt-get -y install protobuf-compiler; \
  pip install numpy; \
  pip install paho-mqtt; \
  pip install pillow; \
  apt-get -y install libhdf5-dev; \
  rm -rf /root/.cache/; \
  rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/lib/python2.7/dist-packages/cv2.aarch64-linux-gnu.so /usr/local/lib/python2.7/site-packages/cv2.aarch64-linux-gnu.so

RUN curl -OL "https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.14.0-buster/tensorflow-1.14.0-cp27-none-linux_aarch64.whl" && \
    pip install tensorflow-1.14.0-cp27-none-linux_aarch64.whl && \
    rm tensorflow-1.14.0-cp27-none-linux_aarch64.whl


RUN mkdir /models
WORKDIR /models

RUN git init && \
        git remote add origin https://github.com/tensorflow/models.git && \
        git config core.sparsecheckout true && \
        echo "resarch/object_detection" >> .git/info/sparse-checkout && \
        git pull --depth 1 origin master && \
        rm -rf .git research/object_detection/test*

WORKDIR /models/research/object_detection
RUN protoc /models/research/object_detectionobject_detection/protos/*.proto --python_out=.
COPY . /models/research/object_detection


ENTRYPOINT ["/usr/local/bin/python", "od_atlas.py"]

