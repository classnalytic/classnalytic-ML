import tensorflow as tf
# import keras

import cv2
import numpy as np
import align.detect_face
import json

import emotion as em

def crop_and_resize(image, position):
    """ Crop Function take path, x1, y1, x2, y2 then give back cropped photo """
    posx1, posy1, posx2, posy2 = position
    cropped = image[posy1: posy1 + abs(posy2 - posy1), posx1: posx1 + abs(posx2 - posx1)]
    cropped = cv2.resize(cropped, (224, 224))
    return cropped

# Detect Face factor
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.8]  # three steps's threshold
factor = 0.709  # scale factor

print('Creating networks and loading parameters')

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0, allow_growth=True)
    tf_config = tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False)
    sess = tf.Session(config=tf_config)
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

# Load emotion
emotion = em.Emotion()
emotion.load_weights("models/emotion/emotion.h5")

def default(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def predict_all(img):
    # print(emotion.model.summary())
    results = []
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    for face_position in bounding_boxes:
        face_position = face_position.astype(int)
        face_pos = (face_position[0], face_position[1], face_position[2], face_position[3])
        # results += [list(face_pos)]

        # Predict emotion
        face = crop_and_resize(img, face_pos)
        
        results += [[emotion.predict(face)]]
        # results += face_position

    print(results)
    response_data = {
        "faces_location": results
    }
    response = json.dumps(response_data)
    print(response)
    return response
