import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import json
import paho.mqtt.client as mqtt

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util

client = mqtt.Client()
client.connect("127.0.0.1", 1883, 600)

cv2.setUseOptimized(True)

MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
print("PATH_TO_LABELS", PATH_TO_LABELS)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    print("PATH_TO_FROZEN_GRAPH", PATH_TO_FROZEN_GRAPH)
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

cap = cv2.VideoCapture("rtsp://192.168.1.164:554/user=admin&password=&channel=1&stream=1.sdp?")
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_FPS, 1)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

num = 0

with detection_graph.as_default():
    with tf.Session() as sess:

        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
 
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, height, width)
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)

            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        while True:
            ret, image_np = cap.read()
            if ret == False:
                continue
            num = num + 1
            print("frame num", num)

            image_np_expanded = np.expand_dims(image_np, axis=0)
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image_np_expanded})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            output = {}
            num_detections = int(output_dict['num_detections'])
            for i in range(num_detections):
            	detection_class = int(output_dict['detection_classes'][i])
            	class_name = category_index[detection_class]['name']
            	score = output_dict['detection_scores'][i]
            	scores = "scores %.f%%" % (score * 100)
            	output[class_name] = scores
            	
            print("output ", output)
            if num_detections > 0 :
            	obj_ret = json.dumps(output)
            	client.publish('object_detection', obj_ret, qos=0)
            else:
            	client.publish('object_detection', "", qos=0)

cap.release()
