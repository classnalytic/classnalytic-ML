import keras

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import numpy as np
import os
import time

import tensorflow as tf

def crop(image, posx1, posy1, posx2, posy2):
    """ Crop Function take path, x1, y1, x2, y2 then give back cropped photo """

    # data = cv2.imread(path)

    cropped = image[posy1: posy1 + abs(posy2 - posy1), posx1: posx1 + abs(posx2 - posx1)]

    return cropped

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
model_path = os.path.join('./', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

cap = cv2.VideoCapture(os.path.join("../", "action/data", "Analyse ผู้เรียน SRJ03 No.1.mp4"))
# cap = cv2.VideoCapture(os.path.join("../", "action/data", "Tsugi no Season.mp4"))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# out = cv2.VideoWriter('../action/data/out/output1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

count = 0
pic_count = 0

export_path = os.path.abspath("../action/data/out/unlabels")

while 1:
    ret, frame = cap.read()
    if ret:
        if count % 60 == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = preprocess_image(image)
            image, scale = resize_image(image)

            start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            print("processing time: ", time.time() - start)

            boxes /= scale

            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if labels_to_names[label] != "person":
                    break
                
                color = label_color(label)
                b = box.astype(int)

                crop_area = b
                if crop_area[0] - 20 < 0:
                    crop_area[0] = 0
                if crop_area[1] - 20 < 0:
                    crop_area[1] = 0
                if crop_area[2] + 20 > frame_width:
                    crop_area[2] = frame_width
                if crop_area[3] + 20 > frame_height:
                    crop_area[3] = frame_height
                
                cropped_img = crop(frame, *crop_area)
                cv2.imwrite(os.path.join(export_path, 'unlabel-{}.jpg'.format(pic_count)), cropped_img)
                pic_count += 1
                print("Exported {} image(s)".format(pic_count))

                captions = "{} {:.3f}".format(labels_to_names[label], score)
                print(captions, b)

                # cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 5)
                # cv2.putText(frame, captions, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

        # out.write(frame)
        cv2.imshow("Test", frame)

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()