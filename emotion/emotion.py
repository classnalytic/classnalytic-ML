import numpy as np
import os

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import utils, optimizers, applications

def resnet_model(class_num):
    model = Sequential()
    # model.add(applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3), name='input')))
    model.add(applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))

    optimizer = optimizers.Adam(lr=0.01)

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    return model

class Emotion:
    def __init__(self):
        self.labels = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    def load_weights(self, weight_path="emotion/emotion3.h5"):
        self.model = resnet_model(8)
        self.model.load_weights(weight_path)

        return True
    
    def predict(self, face):
        model_predict = self.model.predict([face])
        result = {}
        for i in model_predict[0].argsort()[::-1]:
            # print(' {:.5f} {}'.format(model_predict[0][i], self.labels[i]))
            result[self.labels[i]] = model_predict[0][i]
        
        return result
