from models.exif import exif_solver,exif_net
from load_models import initialize_exif
from extract_exif import extract_exif
from lib.utils import benchmark_utils, util,io
import numpy as np
dict = extract_exif()
salvati = dict.keys()
print(salvati)
"""
solver,nc,params = initialize_exif()
exif_solver.ExifSolver.setup_data(solver,data=dict,data_fn=dict)
im = np.zeros((256, 256, 3))

bu = benchmark_utils.EfficientBenchmark(solver, nc, params, im, auto_close_sess=False, 
                                                     mirror_pred=False, dense_compute=False, stride=None, n_anchors=10,
                                                    patch_size=128, num_per_dim=30)
"""