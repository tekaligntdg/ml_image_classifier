# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:40:00 2021

@author: admin
"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
from PIL import Image


IMG_SHAPE = 244

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='./test_images/orange_dahlia.jpg', help = 'Path to image.', type = str)
parser.add_argument('--trained_network', default ='model.h5', help='trained model', type = str)
parser.add_argument('--top_k', default = 5, help = 'Top 5 classes.', type = int)
commands = parser.parse_args()

image_path = commands.image_dir
keras_model = commands.trained_network
top_k = commands.top_k
loadmodel = tf.keras.models.load_model(keras_model,custom_objects={'KerasLayer': hub.KerasLayer})

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

def process_image(img):
    image = np.squeeze(img)
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))/255.0
    return image

def predict(image_path, model, top_k):
    img1 = Image.open(image_path)
    img2 = np.asarray(img1)
    img3 = process_image(img2)
    img4 = np.expand_dims(img3,axis=0)
    prediction = model.predict(img4)
    prob_predictions= prediction[0].tolist()
    probs_final, classes = tf.math.top_k(prob_predictions, k=top_k)

    prob_list = probs_final.numpy().tolist()
    index_temp = classes.numpy()+1
    index = index_temp.tolist()

    return prob_list, index


probs, classes = predict(image_path, loadmodel, top_k)

print("Output of Top K Probabilities and classes of Flowers:\n")
for i in range(len(classes)):
    print(' Class Name:  ',class_names.get(str(classes[i])))
    print(' Probabilties: ', probs[i])
    print(' Class Key: ', classes[i])
