#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import requests
from io import BytesIO
import numpy as np
import tflite_runtime.interpreter as tflite

#inputs

#url = 'https://bit.ly/mlbookcamp-pants'

#preparation

interpreter = tflite.Interpreter(model_path = 'clothing_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

def preprocess_manual(x):
    x /= 127.5
    x -= 1.
    
    return x

#run

def predict(url):
    response = requests.get(url)
    with Image.open(BytesIO(response.content)) as img:
        img = img.resize((299,299), Image.NEAREST)

    x = np.array(img, dtype = 'float32')
    X = np.array([x])
    X = preprocess_manual(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    tf_preds = interpreter.get_tensor(output_index)

    float_preds = tf_preds[0].tolist()
    
    return dict(zip(classes,float_preds))

def lambda_handler(event,context):
    
    url = event['url']
    result = predict(url)
    return result
