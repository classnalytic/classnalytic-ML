import numpy as np
import os

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import utils, optimizers, applications
from keras import backend as k
import tensorflow as tf

global graph, model

def resnet_model(class_num):
    ###################################
    # TensorFlow
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = False

    config.gpu_options.per_process_gpu_memory_fraction = 0.75

    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))
    ###################################

    model = Sequential()
    # model.add(applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3), name='input')))
    model.add(applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))
    graph = tf.get_default_graph()

    return model, graph

class Emotion:
    def __init__(self):
        self.labels = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    def load_weights(self, weight_path="emotion/emotion3.h5"):
        # self.model = resnet_model(8)
        self.model, self.graph = resnet_model(8)
        self.model.load_weights(weight_path)

        print("Loaded weights.")
        return True

    def predict(self, face):
        face = np.expand_dims(face, axis=0)
        face[:] = np.max(face, axis=-1, keepdims=1)/2+np.min(face, axis=-1, keepdims=1)/2

        with self.graph.as_default():
            model_predict = self.model.predict([face])
            result = {}
            for i in model_predict[0].argsort()[::-1]:
                print(' {:.5f} {}'.format(model_predict[0][i], self.labels[i]))
                result[self.labels[i]] = float("{:.8f}".format(model_predict[0][i]))

        return result
