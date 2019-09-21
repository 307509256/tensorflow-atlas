## Use tensorflow lite on atlas
only use atlas cpu
* parse RSTP stream via opencv
* use tensorflow inference
* pub message via MQTT

refer to https://github.com/tensorflow/examples/raw/master/lite/examples/image_classification/raspberry_pi/classify_picamera.py  
Get models from https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip, and unzip.


## usage
python classify_rtsp_atlas.py --model mobilenet_v1_1.0_224_quant.tflite --labels labels_mobilenet_quant_v1_224.txt


## build docker
docker build -t braveyuyong/tf_on_atlas:0.2.1-lite .   
docker run -d --net host braveyuyong/tf_on_atlas:0.2.1-lite  

## use docker
docker pull braveyuyong/tf_on_atlas:0.2.1-lite  
docker run -d --net host braveyuyong/tf_on_atlas:0.2.1-lite  


