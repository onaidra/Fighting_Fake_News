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
def run_vote_no_threads(image, solver, exif_to_use, n_anchors=1, num_per_dim=None,
             patch_size=None, batch_size=None, sample_ratio=3.0, override_anchor=False):
    """
    solver: exif_solver module. Must be initialized and have a network connected.
    exif_to_use: exif to extract responses from. A list. If exif_to_use is None
                 extract result from classification output cls_pred
    n_anchors: number of anchors to use.
    num_per_dim: number of patches to use along the largest dimension.
    patch_size: size of the patch. If None, uses the one specified in solver.net
    batch_size: size of the batch. If None, uses the one specified in solver.net
    sample_ratio: The ratio of overlap between patches. num_per_dim must be None
                  to be useful.
    """

    h, w = np.shape(image)[:2]

    if patch_size is None:
        patch_size = solver.net.im_size

    if batch_size is None:
        batch_size = solver.net.batch_size

    if num_per_dim is None:
        num_per_dim = int(np.ceil(sample_ratio * (max(h,w)/float(patch_size))))

    if exif_to_use is None:
        not_exif = True
        exif_to_use = ['out']
    else:
        not_exif = False
        exif_map = {e: np.squeeze(np.argwhere(np.array(solver.net.train_runner.tags) == e)) for e in exif_to_use}

    responses   = {e:np.zeros((n_anchors, h, w)) for e in exif_to_use}
    vote_counts = {e:1e-6 * np.ones((n_anchors, h, w)) for e in exif_to_use}

    if np.min(image) < 0.0:
        # already preprocessed
        processed_image = image
    else:
        processed_image = util.process_im(image)
    ones = np.ones((patch_size, patch_size))

    anchor_indices = []
    # select n anchors
    for anchor_idx in range(n_anchors):
        if override_anchor is False:
            _h, _w  = np.random.randint(0, h - patch_size), np.random.randint(0, w - patch_size)
        else:
            assert len(override_anchor) == 2, override_anchor
            _h, _w = override_anchor

        anchor_indices.append((_h, _w))
        anchor_patch = processed_image[_h:_h+patch_size, _w:_w+patch_size, :]

        batch_a = np.tile([anchor_patch], [batch_size, 1, 1, 1])
        batch_b, batch_b_coord = [], []

        prev_batch = None
        for i in np.linspace(0, h - patch_size, num_per_dim).astype(int):
            for j in np.linspace(0, w - patch_size, num_per_dim).astype(int):
                compare_patch = processed_image[i:i+patch_size, j:j+patch_size]
                batch_b.append(compare_patch)
                batch_b_coord.append((i,j))

                if len(batch_b) == batch_size:
                    if not_exif:
                        pred = solver.sess.run(solver.net.cls_pred,
                                 feed_dict={solver.net.im_a:batch_a,
                                            solver.net.im_b:batch_b,
                                            solver.net.is_training:False})
                    else:
                        pred = solver.sess.run(solver.net.pred,
                                 feed_dict={solver.net.im_a:batch_a,
                                            solver.net.im_b:batch_b,
                                            solver.net.is_training:False})

                    for p_vec, (_i, _j) in zip(pred, batch_b_coord):
                        for e in exif_to_use:
                            if not_exif:
                                p = p_vec[0]
                            else:
                                p = p_vec[int(exif_map[e])]
                            responses[e][anchor_idx, _i:_i+patch_size, _j:_j+patch_size] += (p * ones)
                            vote_counts[e][anchor_idx, _i:_i+patch_size, _j:_j+patch_size] += ones
                    prev_batch = batch_b
                    batch_b, batch_b_coord = [], []

        if len(batch_b) > 0:
            batch_b_len = len(batch_b)
            to_pad = np.array(prev_batch)[:batch_size - batch_b_len]
            batch_b = np.concatenate([batch_b, to_pad], axis=0)

            if not_exif:
                pred = solver.sess.run(solver.net.cls_pred,
                                     feed_dict={solver.net.im_a:batch_a,
                                                solver.net.im_b:batch_b,
                                                solver.net.is_training:False})
            else:
                pred = solver.sess.run(solver.net.pred,
                                     feed_dict={solver.net.im_a:batch_a,
                                                solver.net.im_b:batch_b,
                                                solver.net.is_training:False})

            for p_vec, (_i, _j) in zip(pred, batch_b_coord):
                for e in exif_to_use:
                    if not_exif:
                        p = p_vec[0]
                    else:
                        p = p_vec[int(exif_map[e])]
                    responses[e][anchor_idx, _i:_i+patch_size, _j:_j+patch_size] += (p * ones)
                    vote_counts[e][anchor_idx, _i:_i+patch_size, _j:_j+patch_size] += ones

    return {e: {'responses':(responses[e] / vote_counts[e]), 'anchors':anchor_indices} for e in exif_to_use}

net_args = {'num_classes':80+3,
                'is_training':False,
                'train_classifcation':True,
                'freeze_base': True,
                'im_size':128,
                'batch_size':64,
                'use_gpu':[0],
                'use_tf_threading':False,
                'learning_rate':1e-4}

solver = initialize_exif()
solver.sess.run(tf.compat.v1.global_variables_initializer())
if solver.net.use_tf_threading:
    solver.coord = tf.train.Coordinator()
    solver.net.train_runner.start_p_threads(solver.sess)
    tf.train.start_queue_runners(sess=solver.sess, coord=solver.coord)

x=run_vote_no_threads(im1,solver,None)
benchmark_utils.EfficientBenchmark(solver,exif_net.EXIFNet,net_args,im1)
#exif_solver.ExifSolver.train(solver)
"""
im = np.zeros((256, 256, 3))

bu = benchmark_utils.EfficientBenchmark(solver, nc, params, im, auto_close_sess=False, 
                                                     mirror_pred=False, dense_compute=False, stride=None, n_anchors=10,
                                                    patch_size=128, num_per_dim=30)
"""