#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='dragon_model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

import numpy as np
from PIL import Image
from urllib import request
from io import BytesIO
   
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_manual(x):
    x /= 255   
    return x

#url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'

def predict(url):
    
    response = request.urlopen(url)
    with Image.open(BytesIO(response.read())) as img:
        img = img.resize((150,150), Image.NEAREST)
        
    x = np.array(img, dtype = 'float32')
    X = np.array([x])
    X = preprocess_manual(X)
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    
    tf_preds = interpreter.get_tensor(output_index)
    
    dicta = {}

    if tf_preds[0][0] > 0.5:
        animal = 'dragon'
    else:
        animal = 'dino'

    dicta[animal] = tf_preds[0][0]
    
    return dicta

def lambda_handler(event,contex):
    url = event['url']
    pred = predict(url)
    
    return pred