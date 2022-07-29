import cv2
from matplotlib import image, pyplot as plt
import numpy as np
#from keras_squeezenet import SqueezeNet
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn import metrics

IMG_SAVE_PATH = imageSet #insert path to training image dataset

NUMBER_MAP = {
    "Zero": 0,
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

NUM_CLASSES = len(NUMBER_MAP)


def mapper(val):
    return NUMBER_MAP[val]


def get_model():
    model = Sequential([
        VGG19(input_shape=(224, 224, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model


#Load images from the directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        #To make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
        except:
            continue

        dataset.append([img, directory])

data, labels = zip(*dataset)
labels = list(map(mapper, labels))


#One hot encode the labels
labels = np_utils.to_categorical(labels)

#Define the model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Start training
history = model.fit(np.array(data), np.array(labels), epochs=10)

#Plots the accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.legend()
plt.savefig('Accuracy.png')
plt.show()

#Plots the loss
plt.plot(history.history['loss'], label='train loss')
plt.legend()
plt.savefig('Loss.png')
plt.show()


#Building Confusion Matrix
testing = model.predict(np.array(data))


print(testing)

metrics.confusion_matrix()



#Save the model for later use
model.save("model.h5")