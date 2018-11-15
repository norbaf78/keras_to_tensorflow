# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:09:00 2018

@author: fabio.roncato
"""

from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
import os
import json

keras_h5_filename = "yolov2.h5"
keras_json_filename = "yolov2.json"

model = load_model(keras_h5_filename)
# save as JSON
json_string = model.to_json()
with open('keras_json_filename', 'w') as outfile:
    json.dump(json_string, outfile)

## model reconstruction from JSON:
#with open(keras_json_filename) as json_data:
#    json_string = json.load(json_data)
#    print(json_string)    
#model = model_from_json(json_string)


sess = K.get_session()
K.set_learning_phase(0)
model = load_model(keras_h5_filename) # get keras model
model.summary() 
model.load_weights(keras_h5_filename) # read keras weight

saver = tf.train.Saver()
saver.save(sess, "./TF_Model/tf_model") # convert keras to tensorflow model

tf_model_path = "./TF_Model/tf_model"

fw = tf.summary.FileWriter('logs', sess.graph)
fw.close()

[node.op.name for node in model.outputs]
[node.op.name for node in model.inputs]