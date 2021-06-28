from models.exif import exif_solver,exif_net
from load_models import initialize_exif
from extract_exif import extract_exif
from lib.utils import benchmark_utils, util,io
from demo import Demo
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
#dict = extract_exif()
#salvati = dict.keys()
#print(salvati)
im1 = cv2.imread("D01_img_orig_0001.jpg")[:,:,[2,1,0]]
print("---------------------------------------------------------------------")
print(im1.shape())
print("---------------------------------------------------------------------")
solver = initialize_exif()
solver.sess.run(tf.compat.v1.global_variables_initializer())
if solver.net.use_tf_threading:
    solver.coord = tf.train.Coordinator()
    solver.net.train_runner.start_p_threads(solver.sess)
    tf.train.start_queue_runners(sess=solver.sess, coord=solver.coord)

net_args = {'num_classes':80+3,
                'is_training':False,
                'train_classifcation':True,
                'freeze_base': True,
                'im_size':128,
                'batch_size':64,
                'use_gpu':[0],
                'use_tf_threading':False,
                'learning_rate':1e-4}

benchmark_utils.EfficientBenchmark(solver,exif_net,net_args,im1)
#exif_solver.ExifSolver.train(solver)
"""
im = np.zeros((256, 256, 3))

bu = benchmark_utils.EfficientBenchmark(solver, nc, params, im, auto_close_sess=False, 
                                                     mirror_pred=False, dense_compute=False, stride=None, n_anchors=10,
                                                    patch_size=128, num_per_dim=30)
"""