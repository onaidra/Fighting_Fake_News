from re import I
from models import exif
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras import Model
from keras.initializers import RandomNormal
from keras.layers import Dense,Flatten,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from extract_exif import extract_exif, random_list,generate_label,cropping_list,get_np_arrays
from matplotlib import image
from lib.utils import benchmark_utils, util,io
import cv2
import numpy as np
import keras
import pickle

def datagenerator(images, labels, batchsize, mode="train"):
    ssad = 1
    while True:
        start = 0
        end = batchsize
        while start  < len(images):
            #if(len(images)-start < batchsize):
            #    break
            # load your images from numpy arrays or read from directory
            #else:
            x = images[start:end] 
            y = labels[start:end]
            if (ssad == 1):
                print(x[0].shape)
                ssad = 0
            yield x, y

            start += batchsize
            end += batchsize

def create_base_model(image_shape, dropout_rate, suffix=''):
    I1 = Input(image_shape)
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=I1, pooling=None)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1]._outbound_nodes = []

    for layer in model.layers:
        layer._name = layer.name + str(suffix)
        layer._trainable = False

    flatten_name = 'flatten' + str(suffix)

    x = model.output
    x = Flatten(name=flatten_name)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    #x = Dense(512, activation='relu')(x)
    #x = Dropout(dropout_rate)(x)

    return x, model.input


def create_siamese_model(image_shape, dropout_rate):

    output_left, input_left = create_base_model(image_shape, dropout_rate)
    output_right, input_right = create_base_model(image_shape, dropout_rate, suffix="_2")
    print("------------------------------------------------------------------------------")
    print(output_left)
    print("------------------------------------------------------------------------------")
    #output = tf.concat([output_left,output_right],0)
    
    L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([output_left, output_right])
    L1_prediction = Dense(1, use_bias=True,
                          activation='sigmoid',
                          input_shape = image_shape,
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                          name='weighted-average')(L1_distance)

    prediction = Dropout(0.2)(L1_prediction)

    siamese_model = Model(inputs=[input_left, input_right], outputs=prediction)

    return siamese_model
"""
siamese_model = create_siamese_model(image_shape=(128,128, 3),
                                         dropout_rate=0.2)

siamese_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['binary_crossentropy', 'acc'])

imagexs =cv2.imread('D01_img_orig_0001.jpg')[:,:,[2,1,0]]

imagexs = np.array(imagexs,np.float32)
imagexs = util.random_crop(imagexs,[128,128])
imagexs = np.expand_dims(imagexs,axis=0)
siamese_model.summary()

tmp1 = np.empty((5, 128, 128, 3), dtype=np.uint8)

for i in range(len(tmp1)):
    tmp1[i] = imagexs

x  = (tmp1,tmp1)

siamese_model.fit(x = (imagexs,imagexs),y=(imagexs),batch_size = 32,epochs=10)
                            #verbose=1,
                            #callbacks=[checkpoint, tensor_board_callback, lr_reducer, early_stopper, csv_logger],
                            #validation_data=(imagexs,imagexs))
                            #max_q_size=3)


#siamese_model.save('siamese_model.h5')


# and the my prediction
siamese_net = load_model('siamese_model.h5', custom_objects={"tf": tf})
X_1 = [image, ] * len(markers)
batch = [markers, X_1]
result = siamese_net.predict_on_batch(batch)


# I've tried also to check identical images 
markers = [image]
X_1 = [image, ] * len(markers)
batch = [markers, X_1]
result = siamese_net.predict_on_batch(batch)

############################################################################################### FINE

siamese_model = create_siamese_model(image_shape=(128,128, 3),
                                         dropout_rate=0.2)

siamese_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['binary_crossentropy', 'acc'])

with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	exif_lbl = pickle.load(fp)
fp.close()

#######################################################################################à
#crop images to 128x128
#######################################################################################à
list1,list2 = get_np_arrays('cropped_arrays.npy')

x_train = datagenerator(list1,exif_lbl,32)

siamese_model.fit_generator(datagenerator(list1,exif_lbl,32),steps_per_epoch=32,epochs=10,verbose=1)
                            #callbacks=[checkpoint, tensor_board_callback, lr_reducer, early_stopper, csv_logger],
                            #validation_data=x_train)
                            #max_q_size=3)"""

import numpy as np
import pandas as pd

data = np.random.rand(200,2)
expected = np.random.randint(2, size=200).reshape(-1,1)

dataFrame = pd.DataFrame(data, columns = ['a','b'])
expectedFrame = pd.DataFrame(expected, columns = ['expected'])

dataFrameTrain, dataFrameTest = dataFrame[:100],dataFrame[-100:]
expectedFrameTrain, expectedFrameTest = expectedFrame[:100],expectedFrame[-100:]

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.utils import np_utils


model = Sequential()
model.add(Dense(12, activation='relu', input_dim=dataFrame.shape[1]))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#Train the model using generator vs using the full batch
batch_size = 8

#without generator
model.fit(
    x = np.array(dataFrame),
    y = np.array(expected),
    batch_size = batch_size,
    epochs = 3)