import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.mobilenet_v2 import MobileNetV2
#model = MobileNetV2(weights='imagenet')

#print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'C:\\Users\\YOUSSEF\\Desktop\\COVID19-app\\models\\covid19.model'

# Load your own trained model
model = load_model(MODEL_PATH)

labels=['covid','normal']
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):

    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/ar', methods=['GET'])
def index_ar():
    # Main page
    return render_template('index_ar.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json).convert('RGB')
        #print(request.files['file'])

        # Save the image to ./uploads
        #img.save("uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        if(preds[0][0]>=preds[0][1]):
            pred_class="POSITIVE"
        else:
            pred_class="NEGATIVE"

        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability

        # Serialize the result, you can add additional fields
        return jsonify(result=pred_class, probability=pred_proba)

    return None


if __name__ == '__main__':

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
