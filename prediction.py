import cv2
import numpy as np
import align.detect_face
import json
from scipy import misc

import tensorflow as tf
import facenet.facenet as facenet
import pickle
import sklearn
import emotion as em
from action import action
import os

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

        facenet.load_model(os.path.abspath("./models/facenet/20180402-114759/"))

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# Load emotion
emotion = em.Emotion()
emotion.load_weights("models/emotion/emotion.h5")

# Load Facenet
classifier_filename_exp = os.path.abspath("./models/facenet/20180402-114759/lfw_classifier.pkl")
with open(classifier_filename_exp, 'rb') as infile:
    (model, class_names) = pickle.load(infile)

def crop_and_resize(image, position, size):
    """ Crop Function take path, x1, y1, x2, y2 then give back cropped photo """
    posx1, posy1, posx2, posy2 = position
    cropped = image[posy1: posy1 + abs(posy2 - posy1), posx1: posx1 + abs(posx2 - posx1)]
    cropped = cv2.resize(cropped, size)
    return cropped

def predict_face(img, bounding_boxes=None, margin=44, image_size=(160, 160)):
    results = []

    if bounding_boxes is None:
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # Facenet Prediction
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces == 0:
        return results

    img_size = np.asarray(img.shape)[0:2]
    img_list = [None] * nrof_faces
    for i in range(nrof_faces):
        det = np.squeeze(bounding_boxes[i, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(
            cropped, image_size, interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)

    feed_dict = {images_placeholder: images,
                phase_train_placeholder: False}
    emb = sess.run(embeddings, feed_dict=feed_dict)

    predictions = model.predict_proba(emb)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(
        len(best_class_indices)), best_class_indices]

    index = 0
    for face_position in bounding_boxes:
        face_position = face_position.astype(int)
        face_pos = (face_position[0], face_position[1], face_position[2], face_position[3])

        result = {}

        if best_class_probabilities[index] > 0.75:
            result["accuracy"] = best_class_probabilities[index]
            result["name"] = class_names[best_class_indices[0]]
            print("results => %s" %class_names[best_class_indices[0]])
        else:
            result["accuracy"] = "null"
            result["name"] = "Unknown"
            print("Unknown")

        index += 1

        results += [result]
    return results

def predict_emotion(img, bounding_boxes=None):
    results = []

    if bounding_boxes is None:
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if bounding_boxes.shape[0] == 0:
        return results

    for face_position in bounding_boxes:
        result = {}
        face_position = face_position.astype(int)
        face_pos = (face_position[0], face_position[1], face_position[2], face_position[3])

        # Predict Emotion
        face = crop_and_resize(img, face_pos, (224, 224))
        result["emotions"] = emotion.predict(face)

        results += [result]
    return results

def face_location(img, bounding_boxes=None):
    results = []

    if bounding_boxes is None:
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if bounding_boxes.shape[0] == 0:
        return results

    for face_position in bounding_boxes:
        result = {}
        face_position = face_position.astype(int)
        face_pos = (face_position[0], face_position[1], face_position[2], face_position[3])

        # Predict Emotion
        result["face_location"] = list(map(int, face_pos))

        results += [result]
    return results

def predict_action(img):
    # print(action.action_classifie(img))
    return action.action_classifie(img)

def predict_all(img):
    results = []
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    face_loc = face_location(img, bounding_boxes)
    face_reg = predict_face(img, bounding_boxes)
    face_emotion = predict_emotion(img, bounding_boxes)
    actions = predict_action(img)

    for i in range(len(face_loc)):
        results += [{**face_loc[i], **face_reg[i], **face_emotion[i]}]
    
    for action in actions:
        for result in results:
            if result["face_location"][0] <= action["face_pos"][0] <= result["face_location"][2] and \
                result["face_location"][1] <= action["face_pos"][1] <= result["face_location"][3]:

                result["action"] = action["action"]

    print(results)
    return results
