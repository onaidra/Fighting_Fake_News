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

EPOCHS = 100

def datagenerator(images,images2, labels, batchsize, mode="train"):
    while True:
        start = 0
        end = batchsize
        while start  < len(images):
            x = images[start:end] 
            y = labels[start:end]
            x2 = images2[start:end]
            yield (x,x2),y

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
    output = tf.concat([output_left,output_right],0)
    """
    L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([output_left, output_right])
    L1_prediction = Dense(1, use_bias=True,
                          activation='sigmoid',
                          input_shape = image_shape,
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                          name='weighted-average')(L1_distance)

    prediction = Dropout(0.2)(L1_prediction)"""

    siamese_model = Model(inputs=[input_left, input_right], outputs=output)

    return siamese_model


############################################################################################### FINE

###########################################################################################################
#EXTRACTION#
###########################################################################################################
#extract exif data
#dict,image_list,dict_keys = extract_exif()

#############################################SAVE DICT##############################################
#with open("dict.pkl", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(dict,fp)
#fp.close()

with open("dict.pkl", "rb") as fp:   #Picklingpickle.dump(l, fp)
	dict = pickle.load(fp)
fp.close()
#############################################SAVE IMAGE LIST##############################################
#with open("list_img.pkl", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(image_list,fp)
#fp.close()

with open("list_img.pkl", "rb") as fp:   #Picklingpickle.dump(l, fp)
	image_list = pickle.load(fp)
fp.close()

#############################################SAVE DICT_KEYS##############################################
#with open("dict_keys.pkl", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(dict_keys,fp)
#fp.close()

with open("dict_keys.pkl", "rb") as fp:   #Picklingpickle.dump(l, fp)
	dict_keys = pickle.load(fp)
fp.close()

#----------------------------------------------------------------------------------------------------------------------------------------

#generate second random list
#second_image_list = random_list(image_list)

#generate lab els for each pair of images

#exif_lbl = generate_label(dict_keys,image_list,second_image_list)

#with open("exif_lbl.txt", "wb") as fp:   #Picklingpickle.dump(l, fp)#
#	pickle.dump(exif_lbl,fp)
#fp.close()

#list1,list2 = cropping_list(image_list,second_image_list)

with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
	exif_lbl = pickle.load(fp)
fp.close()

for i in range(len(exif_lbl)):
    exif_lbl[i] = np.array(exif_lbl[i])
exif_lbl = np.array(exif_lbl)

list1,list2 = get_np_arrays('cropped_arrays.npy')

###########################################################################################################
#MODEL#
###########################################################################################################


siamese_model = create_siamese_model(image_shape=(128,128, 3),
                                         dropout_rate=0.2)

siamese_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['binary_crossentropy', 'accuracy'])


x_train = datagenerator(list1,list2,exif_lbl,32)

steps = int(len(list1)/EPOCHS)

siamese_model.fit(x_train,epochs=EPOCHS,steps_per_epoch=steps)
siamese_model.save('siamese_model.h5')
