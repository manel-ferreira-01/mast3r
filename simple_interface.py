import os
import torch
import numpy as np
import tempfile
from contextlib import nullcontext

from mast3r.demo import get_args_parser, main_demo, get_reconstructed_scene, set_scenegraph_options
from dust3r.utils.device import to_numpy


from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp

import matplotlib.pyplot as pl

import os
from scipy.io import savemat
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='mast3r Inference')
parser.add_argument('--outdir', type=str, default='./output', help='Output directory e.g. ./output')
parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
parser.add_argument('--filelist', type=str, default='./images_in', help='Path to the filelist containing input images, e.g  ./images_in')
args = parser.parse_args()

outdir = args.outdir
device = args.device
filelist = args.filelist

#make filelist be a list of all the files in the directory
filelist = [os.path.join(filelist, f) for f in os.listdir(filelist)]
print(filelist)

# check whats this
# torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

# TODO: THIS NEEDS TO BE LAUNCHED AS SOON AS THE CONTAINER STARTS
model = AsymmetricMASt3R.from_pretrained("./docker/files/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").to(device)

#scenegraphy
win_col, winsize, win_cyclic, refid = set_scenegraph_options(filelist, False, 0, "complete")

# Get the 3D model from the scene - one function, full pipeline
scene, _ = get_reconstructed_scene(outdir=outdir,
                                   gradio_delete_cache=False,
                                   model=model,
                                   device=device,
                                   filelist=filelist,
                                   shared_intrinsics=True,
                                   optim_level="coarse",
                                   winsize=winsize,
                                   win_cyclic=win_cyclic, 
                                   refid=refid)
#unpack scene
scene = scene.sparse_ga

mask = to_numpy(scene.get_masks())
pts3d, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
pts3d_clr = to_numpy(scene.get_pts3d_colors())
#confs work for single camera

rgbimgs = scene.imgs
focals = scene.get_focals().cpu()
cams2world = scene.get_im_poses().cpu()

# world frame
pts3d = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
pts3d_clr = np.concatenate([p[m] for p, m in zip(rgbimgs, mask)])
pts3d_clr = pts3d_clr.reshape(-1,3)
confs_wrld = np.concatenate([p[m] for p, m in zip(confs, mask)])
confs_wrld = confs_wrld.reshape(-1,1) / np.max(confs_wrld)

# create a dict about world info
wrld_info = {
    'wrld': {
        'xyz': pts3d,
        'color': pts3d_clr,
        'conf': confs_wrld, 
    }
}
savemat(outdir + '/wrld_info.mat', wrld_info)

cell_array = np.empty((len(cams2world), 1), dtype=object)
for i in range(0,cell_array.shape[0]):
    cell_array[i,0] = {
        'extrinsics': to_numpy(cams2world[i]),
        'intrinsics': to_numpy(scene.get_focals().cpu()[i]),
        'rgb': to_numpy(rgbimgs[i]),
        'depth': np.reshape(depthmaps[i], (to_numpy(rgbimgs[i]).shape[0], to_numpy(rgbimgs[i]).shape[1])),
        'conf': confs[i]
    }
data = {
    'cams_info': cell_array  # Saving the cell array as 'myCellArray'
}
savemat(outdir + '/cam_info.mat', data)
