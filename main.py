import tensorflow as tf
import keras

import cv2
import numpy as np
import align.detect_face

def main():
    # Detect Face factor
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.8]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    # Camera
    capture = cv2.VideoCapture(0)
    frame_interval = 5
    count = 0

    while True:
        cam_status, frame = capture.read()
        if not cam_status:
            print(cam_status)
            break

        if count == 0:
            img = frame
            bounding_boxes, _ = align.detect_face.detect_face(
                            img, minsize, pnet, rnet, onet, threshold, factor)
            print(bounding_boxes.shape)

        # Draw rectangle at the face position
        for face_position in bounding_boxes:
            face_position = face_position.astype(int)
            cv2.rectangle(frame, (face_position[0], face_position[1]), (
                            face_position[2], face_position[3]), (0, 255, 0), 2)

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
