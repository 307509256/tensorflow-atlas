## Use tensorflow lite on atlas
* parse RSTP stream via opencv
* use tensorflow inference
* pub message via MQTT

Refere to https://github.com/tensorflow/models/tree/master/research/object_detection/object_detection_tutorial.ipynb  
Get models from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

## build docker
docker build -t braveyuyong/tf_on_atlas:0.2.1 .
docker run -d --net host braveyuyong/tf_on_atlas:0.2.1

## use docker
* run eclipse-mosquitto docker first  
* run tensorflow inferenc  
  docker pull braveyuyong/tf_on_atlas:0.2.1  
  docker run -d --net host braveyuyong/tf_on_atlas:0.2.1  
* usr luckyyuyong/visual_rtsp[https://github.com/luckyyuyong/visual_rtsp] check inference

