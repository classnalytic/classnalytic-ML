import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import Sequential
# from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import utils, optimizers, applications
from collections import OrderedDict

from psutil import virtual_memory

DATA_DIR = os.path.abspath("data/legend.csv")
TRAIN_DIR = os.path.abspath("train")
VALIDATE_DIR = os.path.abspath("validate")

batch_size = 30

def prepare_from_csv(image_dir, csv_dir):
    pic_list = os.listdir(image_dir)
    pic_labels = pd.read_csv(csv_dir)

    pic_labels["emotion"] = pic_labels["emotion"].str.lower()
    pic_labels["int_label"], emotion_labels = pic_labels["emotion"].factorize()

    return emotion_labels, pic_labels

# def load_data(dataset):
#     imgs = np.empty((len(dataset),100,100,3),dtype="float32")
#     mem = virtual_memory()
#     print(mem)
#     j = 0
#     for img_file in dataset["image"]:
#         img_path = IMAGE_DIR +"/"+ img_file
#         img = image.load_img(img_path, target_size=(100, 100))
#         arr = np.asarray(img, dtype="float32")
#         imgs[j,:,:,:] = arr
#         j += 1
#     labels = dataset["int_label"].values
#     return imgs, labels

def dataset(img_width=224, img_height=224):
    train_datagen = image.ImageDataGenerator()

    test_datagen = image.ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        interpolation='bicubic',
        class_mode='categorical')
    
    test_generator = test_datagen.flow_from_directory(
        VALIDATE_DIR,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        interpolation='bicubic',
        class_mode='categorical')
    
    class_dictionary = train_generator.class_indices
    print("[Class dictionary] {}".format(class_dictionary))
    sorted_class_dictionary = OrderedDict(sorted(class_dictionary.items()))
    sorted_class_dictionary = sorted_class_dictionary.values()
    print("[Sorted Class dictionary] {}".format(sorted_class_dictionary))

    return train_generator, test_generator, class_dictionary

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

def train():
    # emotion_labels, pic_labels = prepare_from_csv(IMAGE_DIR, DATA_DIR)

    # train_set = pic_labels[4000:].reset_index()
    # validation_set = pic_labels[:4000].reset_index()

    # train_imgs, train_labels = load_data(train_set)
    # validate_imgs, validate_labels = load_data(validation_set)

    # train_labels = utils.to_categorical(train_labels, 8)
    # validate_labels = utils.to_categorical(validate_labels, 8)

    # Normalize data
    # train_imgs /= 255
    # validate_imgs /= 255


    train_generator, validate_generator, class_dictionary = dataset()

    # nb_train_files = sum([len(os.listdir(TRAIN_DIR + "/" + i)) for i in os.listdir(TRAIN_DIR)])
    # nb_test_files = sum([len(os.listdir(VALIDATE_DIR + "/" + i)) for i in os.listdir(VALIDATE_DIR)])

    # print("Total train files : %d" %nb_train_files)
    # print("Total test files : %d" %nb_test_files)

    class_num = len(class_dictionary)

    model = resnet_model(class_num)

    model.summary()

    nb_epoch = 100
    print("len train_generator %d" %len(train_generator))
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=validate_generator.samples // batch_size,
        epochs=nb_epoch,
        validation_data=validate_generator,
        workers=12)
    # history = model.fit(train_imgs, train_labels, batch_size=batch_size, epochs=n_epochs, validation_data=(validate_imgs, validate_labels))

    model.save_weights('emotion3.h5')

    # score = model.evaluate(validate_imgs, validate_labels, verbose=0)
    score = model.evaluate_generator(validate_generator, len(train_generator)/batch_size, workers=12)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

if __name__ == "__main__":
    train()

