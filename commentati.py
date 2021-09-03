"""
im1 = cv2.imread(r"C:\Users\Adri\Desktop\VISIOPE\prova\foto\D01_Motorola_E3_1\orig\D01_img_orig_0001.jpg")[:,:,[2,1,0]]
im1  = util.random_crop(im1,[128,128])
list1 = []
list2 = []
for i in range(10):
    list1.append(im1)
    list2.append(im1)
exif_lbl = np.ones((2,83))
exif_lbl[1] = np.random.randint(0,2,(1,83))
cls_lbl = np.ones((1,1))
cls_lbl[0][0] = 1
"""
"""
#------------------------------------------------------------------------------
im1 = [image_list[0]]
im2 = [image_list[1]]
exif_lbl = generate_label(im1,im2)
list1,list2 = cropping_list(im1,im2)
#-------------------------------------------------------------------------------


second_image_list = random_list(image_list)
exif_lbl = generate_label(image_list,second_image_list)
list1 = []
list2 = []
#tmp1 = np.empty((N, 128, 128, 3), dtype=np.uint8)
#tmp2 = np.empty((N, 128, 128, 3), dtype=np.uint8)
for i in range(10):
    print(second_image_list[i])
    x = cv2.imread(image_list[i])[:,:,[2,1,0]]
    y = cv2.imread(second_image_list[i])[:,:,[2,1,0]]
    patch1 = util.random_crop(x,[128,128])
    patch2 = util.random_crop(y,[128,128])

    list1.append(patch1)
    list2.append(patch2)



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
    epochs = 100)

"""


"""