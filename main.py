from models import exif
import random
from models.exif import exif_solver,exif_net
from load_models import initialize_exif
from extract_exif import extract_exif, random_list,generate_label,cropping_list
from lib.utils import benchmark_utils, util,io
from demo import Demo
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

#extract exif data
dict,image_list,dict_keys = extract_exif()
print(len(dict_keys))
#generate second random list
second_image_list = random_list(image_list)

#crop images to 128x128
list1,list2 = cropping_list(image_list,second_image_list)

#generate labels for each pair of images

exif_lbl = generate_label(dict_keys,image_list,image_list)

#start initialization

solver = initialize_exif()
solver.sess.run(tf.compat.v1.global_variables_initializer())
if solver.net.use_tf_threading:
    solver.coord = tf.train.Coordinator()
    solver.net.train_runner.start_p_threads(solver.sess)
    tf.train.start_queue_runners(sess=solver.sess, coord=solver.coord)

cls_lbl = np.ones((1,1))
cls_lbl[0][0] = len(dict_keys)


im1_merge = {'im_a':list1,'im_b':list2,'exif_lbl': exif_lbl,'cls_lbl': cls_lbl}
exif_solver.ExifSolver.setup_data(solver,list1,im1_merge)
exif_solver.ExifSolver.train(solver)

"""
im = np.zeros((256, 256, 3))

bu = benchmark_utils.EfficientBenchmark(solver, nc, params, im, auto_close_sess=False, 
                                                     mirror_pred=False, dense_compute=False, stride=None, n_anchors=10,
                                                    patch_size=128, num_per_dim=30)
"""