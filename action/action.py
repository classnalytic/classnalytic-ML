import cv2
import numpy as np
import tensorflow as tf
# import sys
# sys.path.insert(0, './action')

# from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

w, h = 432, 368
parts = ['nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist', 'r_hip', 'r_knee',
         'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye', 'l_eye', 'r_ear', 'l_ear']

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)

tf_config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)

e = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(w, h), tf_config=tf_config)

def action_classifie(img):
    img_shape = img.shape
    humans = e.inference(img, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    bodys_pos = []

    for human in humans:
        temp = {}
        for i in range(len(parts)):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            temp[parts[i]] = body_part
        bodys_pos += [temp]

    return is_handup(bodys_pos, img_shape)

def is_handup(bodys_pos, img_shape):
    result = []
    print(len(bodys_pos))
    for body_pos in bodys_pos:
        temp = None
        if 'r_wrist' in body_pos and 'r_shoulder' in body_pos:
            if body_pos["r_wrist"].y <= body_pos['r_shoulder'].y:
                temp = {
                    "face_pos": (body_pos['nose'].x * img_shape[1] + 0.5, body_pos['nose'].y * img_shape[0] + 0.5),
                    "action" : "Hand up"
                }

        elif 'l_wrist' in body_pos and 'l_shoulder' in body_pos:
            if body_pos["l_wrist"].y <= body_pos['l_shoulder'].y:
                temp = {
                    "face_pos": (body_pos['nose'].x * img_shape[1] + 0.5, body_pos['nose'].y * img_shape[0] + 0.5),
                    "action" : "Hand up"
                }

        elif 'r_elbow' in body_pos and 'r_shoulder' in body_pos:
            if body_pos["r_elbow"].y <= body_pos['r_shoulder'].y:
                temp = {
                    "face_pos": (body_pos['nose'].x * img_shape[1] + 0.5, body_pos['nose'].y * img_shape[0] + 0.5),
                    "action" : "Hand up"
                }

        elif 'l_elbow' in body_pos and 'l_shoulder' in body_pos:
            if body_pos["l_elbow"].y <= body_pos['l_shoulder'].y:
                temp = {
                    "face_pos": (body_pos['nose'].x * img_shape[1] + 0.5, body_pos['nose'].y * img_shape[0] + 0.5),
                    "action" : "Hand up"
                }


        if temp is not None:
            result += [temp]

    print(result)
    return result