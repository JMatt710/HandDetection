from keras.models import load_model
import cv2
import numpy as np
import os
from sklearn import metrics

training_path = imageSet #training image set path
testing_path = imageSet #testing image set path

NUMBER_MAP = {
    0: "Zero",
    1: "One",
    2: "Two", 
    3: "Three", 
    4: "Four", 
    5: "Five" 
}

def mapper(val):
    return NUMBER_MAP[val]


model = load_model("model.h5")

training_list = []
for directory in os.listdir(training_path):
    path = os.path.join(training_path, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):

        img = cv2.imread(os.path.join(path, item))
        
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
        except:
            continue

        pred = model.predict(np.array([img]))
        number = np.argmax(pred[0])
        num = mapper(number)

        training_list.append(num)
        
        #print("Predicted: {}".format(number))

testing_list = []
for directory in os.listdir(testing_path):
    path = os.path.join(testing_path, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):

        img = cv2.imread(os.path.join(path, item))
        
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
        except:
            continue

        pred = model.predict(np.array([img]))
        number = np.argmax(pred[0])
        num = mapper(number)

        testing_list.append(num)

print(metrics.confusion_matrix(training_list, testing_list))
print(metrics.classification_report(training_list, testing_list))

