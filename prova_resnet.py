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
import pickle

import numpy as np

def create_base_model(image_shape, dropout_rate, suffix=''):
    I1 = Input(shape=image_shape)
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
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    return x, model.input


def create_siamese_model(image_shape, dropout_rate):

    output_left, input_left = create_base_model(image_shape, dropout_rate)
    output_right, input_right = create_base_model(image_shape, dropout_rate, suffix="_2")

    L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([output_left, output_right])
    L1_prediction = Dense(1, use_bias=True,
                          activation='sigmoid',
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                          name='weighted-average')(L1_distance)

    prediction = Dropout(0.2)(L1_prediction)

    siamese_model = Model(inputs=[input_left, input_right], outputs=prediction)

    return siamese_model

siamese_model = create_siamese_model(image_shape=(64, 64, 3),
                                         dropout_rate=0.2)

siamese_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['binary_crossentropy', 'acc'])
siamese_model.fit_generator(["https://drive.google.com/drive/u/0/folders/1qtAc7_R9DVIf8-fqI5BqTQ8yC2nRbPgq/D01_img_orig_0143","https://drive.google.com/drive/u/0/folders/1qtAc7_R9DVIf8-fqI5BqTQ8yC2nRbPgq/D01_img_orig_0143"],
                            steps_per_epoch=1000,
                            epochs=10,
                            verbose=1,
                            #callbacks=[checkpoint, tensor_board_callback, lr_reducer, early_stopper, csv_logger],
                            validation_data=(["https://drive.google.com/drive/u/0/folders/1qtAc7_R9DVIf8-fqI5BqTQ8yC2nRbPgq/D01_img_orig_0143","https://drive.google.com/drive/u/0/folders/1qtAc7_R9DVIf8-fqI5BqTQ8yC2nRbPgq/D01_img_orig_0143"]))
                            #max_q_size=3)

siamese_model.save('siamese_model.h5')


"""
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
"""
