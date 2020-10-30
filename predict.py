import argparse
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_hub as hub
import json

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from PIL import Image


def process_image(image):
    image = tf.image.resize(image,(224,224))/255
    return image.numpy();

def predict(img_path,model,k):
    
    if(k==None):
        k=1; 
    
    with tf.device('/CPU:0'):
        img_path = img_path
        im = Image.open(img_path)
        image = np.asarray(im)
        processed_image = process_image(image)
        processed_image = np.expand_dims(processed_image,axis=0)
        prop,index = tf.math.top_k(model.predict(processed_image),k)
        prop = prop.numpy()[0].tolist()
        index = index.numpy()[0].tolist()
        return prop,index



parser = argparse.ArgumentParser(description = "A command line program that takes as input data and a model and outputs a prediction")

parser.add_argument('image_path', action="store", help = "The file path to the image that is to be predicted.")
parser.add_argument('model_path', action="store", help = "The file path to the model that is to predict the image.")
parser.add_argument('--category_names',action='store', help = "The .json file that includes the label of the image.") 
parser.add_argument('--top_k', action="store", type=int, help = "The number of predictions showcased in terms of most probable to least.")
results = parser.parse_args()
given_model = tf.compat.v1.keras.experimental.load_from_saved_model(results.model_path,custom_objects={'KerasLayer': hub.KerasLayer})
prob,ind = predict(results.image_path,given_model,results.top_k)

if __name__ == '__main__':
    if(results.category_names != None):
        with open(results.category_names, 'r') as f:
            class_names = json.load(f)
    print("\n")
    for i in range(len(prob)):
        if(results.category_names == None):
            label = ind[i]
        else:
            label = class_names[str(ind[i]+1)]
        print("{}) Class: {}, Probability: {}%\n".format(i+1,label,prob[i]*100))