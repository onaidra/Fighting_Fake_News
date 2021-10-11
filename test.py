from re import I
from models import exif
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Input
from keras import Model,Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense,Flatten,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from extract_exif import extract_exif,generate_label,cropping_list,get_np_arrays,remove_elements,create_batch_samples
from matplotlib import image
from lib.utils import benchmark_utils, util,io
import cv2
import os
import numpy as np
import keras
import pickle
from keras.engine import keras_tensor
from google.colab.patches import cv2_imshow

def datagenerator(images,images2, batchsize, mode="train"):
    while True:
        start = 0
        end = batchsize
        while start  < len(images):
            x = images[start:end] 
            #y = labels[start:end]
            x2 = images2[start:end]
            yield (x,x2)

            start += batchsize
            end += batchsize


print("[INFO] starting test")
"""
#--------------------------------------------------------------- EXTRACT 
dict,image_list,dict_keys = extract_exif()
#--------------------------------------------------------------- REMOVE ELEMENTS
dict = remove_elements(dict)

print("[INFO] number of keys: ", len(dict_keys))
#--------------------------------------------------------------- CREATE SAMPLES
list1_img,list2_img = create_batch_samples(dict,image_list)
#--------------------------------------------------------------- GENERATE LABELS
exif_lbl = generate_label(dict_keys,list1_img,list2_img)

with open("exif_lbl.txt", "wb") as fp:   #Picklingpickle.dump(l, fp)#
	pickle.dump(exif_lbl,fp)
fp.close()
#--------------------------------------------------------------- CROP IMAGES
list1,list2 = cropping_list(list1_img,list2_img)


#--------------------------------------------------------------- GET ELEMENTS
#list1,list2 = get_np_arrays('test_cropped_arrays.npy')
#with open("exif_lbl.txt", "rb") as fp:   #Picklingpickle.dump(l, fp)
#	exif_lbl = pickle.load(fp)
#fp.close()
#--------------------------------------------------------------- RUN MODEL
x_train = datagenerator(list1,list2,exif_lbl,32)
"""
#prova
path = r"/content/drive/MyDrive/foto/test/images/2217.jpg"
#dir = os.listdir(path)
foto1 = cv2.imread(path)[:,:,[2,1,0]]
foto2 = cv2.imread(path)[:,:,[2,1,0]]
patch1 = util.random_crop(foto1,[128,128])
patch2 = util.random_crop(foto2,[128,128])
print(foto1.shape)
cv2_imshow(patch1)
cv2_imshow(patch2)
"""
#length = len(dir)
tmp1 = np.empty((length, 128, 128, 3), dtype=np.uint8)
tmp2 = np.empty((length, 128, 128, 3), dtype=np.uint8)
i = 0
for elem in dir:
    elem = os.path.join(path,elem)
    foto1 = cv2.imread(elem)[:,:,[2,1,0]]
    foto2 = cv2.imread(elem)[:,:,[2,1,0]]
    patch1 = util.random_crop(foto1,[128,128])
    patch2 = util.random_crop(foto2,[128,128])
    tmp1[i] = patch1
    tmp2[i] = patch2
    i+=1

model = tf.keras.models.load_model('siameseMLP.h5')


x = model.predict((tmp1,tmp2))
print(x)
print(model.metrics_names)
"""