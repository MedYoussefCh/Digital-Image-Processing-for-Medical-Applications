import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
ap.add_argument("-m", "--model", required = True)
args = vars(ap.parse_args())

INIT_LR = 1e-3
EPOCHS = 25

model = tf.keras.models.load_model(args["model"])
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
image = cv2.imread(args["image"])
imageCpy = cv2.resize(image, (224, 224))

images = [imageCpy]
data =  np.stack(images, axis=0)

print(data.shape)

LABEL = ["covid-positive", "covid-negetive"]

pred =  model.predict(data)
label = LABEL[pred.argmax()]

cv2.putText(image, label, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
cv2.imshow("Image", image)
key = cv2.waitKey(0)