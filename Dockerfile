FROM braveyuyong/tf_on_atlas:0.2.0

RUN mkdir -p /models/resarch/object_detection
COPY . /models/resarch/object_detection
WORKDIR /models/resarch/object_detection

ENTRYPOINT ["/usr/local/bin/python", "/models/resarch/object_detection/od_atlas.py"]

