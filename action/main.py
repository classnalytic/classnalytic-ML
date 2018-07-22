import cv2
import numpy as np
import os

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

PRETRAINED_MODEL_PATH = os.path.abspath("mask_rcnn_coco.h5")

class HumanConfig(Config):
    # Give the configuration a recognizable name
    NAME = "human"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # Background + Human

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

class InferenceConfig(HumanConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def main():
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
    config = InferenceConfig()
    config.display()

    cap = cv2.VideoCapture(0)
    success = True

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.abspath("logs"))
    model.load_weights(PRETRAINED_MODEL_PATH, by_name=True)
    colors = visualize.random_colors(len(class_names))

    print("Opening camera")

    while 1:
        success, frame = cap.read()

        predictions = model.detect([frame], verbose=1)
        p = predictions[0]

        output = visualize.display_instances(frame, p['rois'], p['masks'], p['class_ids'],
                                     class_names, p['scores'], colors=colors, real_time=True, figsize=(6, 5))

        cv2.imshow("test", output)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
