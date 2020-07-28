#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on
@author: Shivani Reddy
"""

import h5py
from tensorflow.keras import backend as Ks
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Add
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

trainingData_folder = './input_dataset'
image_length = 320
image_breadth = 320
total_slices = 34742
input_data = np.zeros((total_slices,320,320,1))
groundtruth_data = np.zeros((total_slices,320,320,1))

for i in range(total_slices):
    input_data[i,:,:,:] = np.load(trainingData_folder+'/input'+str(i+1)+'.npy')
    groundtruth_data[i,:,:,:] = np.load(trainingData_folder+'/output'+str(i+1)+'.npy')

input_data = (input_data - np.mean(input_data)) / np.std(input_data)
input_data = (input_data - np.min(input_data))/np.ptp(input_data)
groundtruth_data = (groundtruth_data - np.mean(groundtruth_data)) / np.std(groundtruth_data)
groundtruth_data = (groundtruth_data - np.min(groundtruth_data))/np.ptp(groundtruth_data)

d=34
f=3
n=5
B=32
l=1
LEARNING_RATE = 1e-5
input_shape = (image_length,image_breadth,l)
inputReconstruction = Input(input_shape) # template dimension for the input

print('Building the CS-net model...')
firstLayer = Convolution2D(d, (3, 3), strides = (1, 1), kernel_initializer = RandomUniform(minval=-0.05, maxval=0.05, seed=None), padding='same',
                            input_shape=(image_length,image_breadth,l), use_bias=True, bias_initializer='zeros')(inputReconstruction)
firstLayer = Activation('relu')(firstLayer)
M0Layer = firstLayer
for i in range(5):
    M1Layer = Convolution2D(d, (3, 3), strides = (1, 1), kernel_initializer = RandomUniform(minval=-0.05, maxval=0.05, seed=None), padding='same',
                                input_shape=(image_length,image_breadth,d), use_bias=True, bias_initializer='zeros')(M0Layer)
    M1Layer = Activation('relu')(M1Layer)
    M1Layer = Add()([M0Layer,M1Layer])
    M2Layer = Convolution2D(d, (3, 3), strides = (1, 1), kernel_initializer = RandomUniform(minval=-0.05, maxval=0.05, seed=None), padding='same',
                                input_shape=(image_length,image_breadth,d), use_bias=True, bias_initializer='zeros')(M1Layer)
    M2Layer = Activation('relu')(M2Layer)
    M0Layer = M2Layer
lastLayer = Convolution2D(l, (3, 3), strides = (1, 1), kernel_initializer = RandomUniform(minval=-0.05, maxval=0.05, seed=None), padding='same',
                            input_shape=(image_length,image_breadth,d), use_bias=True, bias_initializer='zeros')(M2Layer)
final_output = Add()([lastLayer,inputReconstruction])

model = Model(inputs=inputReconstruction, outputs = final_output)

checkpoint_path = "./checkPoint_CSNet_30epochs_5layers.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
myadam = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-3, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=myadam)
model.summary()

# # Training the CS-net model
# model.fit(input_data, groundtruth_data, validation_split=0.2, epochs=30, batch_size = 32)

# Testing the model
model.load_weights(checkpoint_path)
weights = model.get_weights()
print(weights)

testData_folder = "./testData"
test_slices = 7135
input_data_test = np.zeros((test_slices,320,320,1))
groundtruth_data_test = np.zeros((test_slices,320,320,1))
for i in range(test_slices):
    input_data_test[i,:,:,:] = np.load(testData_folder+'/input'+str(i+1)+'.npy')
    groundtruth_data_test[i,:,:,:] = np.load(testData_folder+'/output'+str(i+1)+'.npy')

input_data_test = (input_data_test - np.mean(input_data_test)) / np.std(input_data_test)
input_data_test = (input_data_test - np.min(input_data_test))/np.ptp(input_data_test)
groundtruth_data_test = (groundtruth_data_test - np.mean(groundtruth_data_test)) / np.std(groundtruth_data_test)
groundtruth_data_test = (groundtruth_data_test - np.min(groundtruth_data_test))/np.ptp(groundtruth_data_test)

model.compile(loss='mean_squared_error', optimizer='adam')
output_image = model.predict(x=input_data_test,batch_size = 32)
np.save('output_images_name', output_image)
loss = model.evaluate(x=input_data_test, y=groundtruth_data_test, batch_size=32)
