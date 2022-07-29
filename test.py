from keras.models import load_model
import cv2
import numpy as np
import sys

filepath = sys.argv[1]

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

img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

pred = model.predict(np.array([img]))
number = np.argmax(pred[0])
num = mapper(number)

print("Predicted: {}".format(num))