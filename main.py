import tensorflow as tf
import keras

import sys
import cv2
import numpy as np
import align.detect_face
# from PIL import Image

import emotion as em

def crop_and_resize(image, position):
    """ Crop Function take path, x1, y1, x2, y2 then give back cropped photo """
    posx1, posy1, posx2, posy2 = position
    cropped = image[posy1: posy1 + abs(posy2 - posy1), posx1: posx1 + abs(posx2 - posx1)]
    cropped = cv2.resize(cropped, (224, 224))
    return cropped

def main():
    # Detect Face factor
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.8]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    # with tf.Graph().as_default():
    #     sess = tf.Session()
    #     with sess.as_default():
    #         pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    # Load emotion
    emotion = em.Emotion()
    emotion.load_weights("emotion/emotion3.h5")

    # Camera
    capture = cv2.VideoCapture(0)

    if not capture:
        print("Can't get image from webcam")
        sys.exit(0)

    frame_interval = 2
    count = 0

    while True:
        cam_status, frame = capture.read()
        if not cam_status:
            print(cam_status)
            continue

        if count == 0:
            img = frame
            bounding_boxes, _ = align.detect_face.detect_face(
                            img, minsize, pnet, rnet, onet, threshold, factor)
            # print(bounding_boxes.shape)

        if bounding_boxes.shape[0] <= 0:
            continue

        # Prediction and Draw rectangle
        for face_position in bounding_boxes:
            face_position = face_position.astype(int)

            # Predict emotion
            face = crop_and_resize(frame, (face_position[0], face_position[1], face_position[2], face_position[3]))

            emotion_result = emotion.predict(face)
            emotion_max = max(emotion_result, key=emotion_result.get)
            face_emotion = "{} {:.5f}".format(emotion_max, emotion_result[emotion_max])

            # Draw rectangle at the face position
            cv2.rectangle(frame, (face_position[0], face_position[1]), (
                            face_position[2], face_position[3]), (0, 255, 0), 2)
            cv2.putText(frame, face_emotion, (face_position[0], face_position[1]-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        cv2.imshow('Webcam', frame)

        count += 1
        if count >= frame_interval:
            count = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
