FROM braveyuyong/tf_on_atlas:0.2.0

RUN mkdir /models
COPY . /models/
WORKDIR /models

ENTRYPOINT /usr/local/bin/python classify_rtsp_atlas.py --model mobilenet_v1_1.0_224_quant.tflite --labels labels_mobilenet_quant_v1_224.txt

