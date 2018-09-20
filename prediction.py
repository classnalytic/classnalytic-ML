import tensorflow as tf
# import keras

import cv2
import numpy as np
import align.detect_face
import json
from scipy import misc

import facenet.facenet as facenet
import pickle
import sklearn
import emotion as em
import os

def crop_and_resize(image, position, size):
    """ Crop Function take path, x1, y1, x2, y2 then give back cropped photo """
    posx1, posy1, posx2, posy2 = position
    cropped = image[posy1: posy1 + abs(posy2 - posy1), posx1: posx1 + abs(posx2 - posx1)]
    cropped = cv2.resize(cropped, size)
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

# Load Facenet
classifier_filename_exp = os.path.abspath("./models/facenet/20180402-114759/lfw_classifier.pkl")
with open(classifier_filename_exp, 'rb') as infile:
    (model, class_names) = pickle.load(infile)

def predict_all(img):
    results = []
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    for face_position in bounding_boxes:
        temp_result = {}
        face_position = face_position.astype(int)
        face_pos = (face_position[0], face_position[1], face_position[2], face_position[3])

        # Predict Emotion
        face = crop_and_resize(img, face_pos, (224, 224))
        temp_result["face_location"] = list(map(int, face_pos))
        temp_result["emotions"] = emotion.predict(face)

        results += [temp_result]

    # Facenet Prediction
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(os.path.abspath("./models/facenet/20180402-114759/"))

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            nrof_faces = bounding_boxes.shape[0]
            margin = 44

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
                    cropped, (160, 160), interp='bilinear')
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
                temp_result = results[index]
                face_position = face_position.astype(int)
                face_pos = (face_position[0], face_position[1], face_position[2], face_position[3])

                if best_class_probabilities[index] > 0.75:
                    temp_result["name"] = class_names[best_class_indices[0]]
                    print("results => %s" %class_names[best_class_indices[0]])
                else:
                    temp_result["name"] = "Unknown"
                    print("Unknown")

                results[index] = temp_result
                index += 1

    response = json.dumps(results)
    print(response)
    return response
