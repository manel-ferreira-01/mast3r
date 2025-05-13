import cv2 as cv
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R
import rerun as rr
from auxiliar.rerun import log_points, log_camera_pose
import time

import rerun as r
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from dust3r.utils.device import to_numpy


def tensor_to_cv2(img):
    return (((img.squeeze().permute(1, 2, 0).detach().cpu().numpy() + 1 )/2)*255).astype('uint8')

def rerun_wlrd(scene, imgs, scale_rerun=0.7, mode="dust3r", conf_thresh = 2):

    # Initialize Rerun logging
    rr.init("3D Points and Camera Visualization")
    rr.notebook_show(width=int(1920*scale_rerun), height=int(1080*scale_rerun))

    if mode=="dust3r":
        cam_poses = scene.get_im_poses()
        colors = [im["img"].squeeze().permute(1,2,0).reshape(-1, 3).detach().cpu().numpy() for im in imgs]

        pts3d = scene.get_pts3d()
        confs = scene.get_conf()
        depthmaps = scene.get_depthmaps()
        #confs = cleaned_confs

        # load all 3d points and colors
        cat_3d_wrld = np.concatenate([pts3d[timestamp].detach().cpu().numpy().reshape(-1, 3)[(confs[timestamp].detach().cpu().reshape(-1,1)>conf_thresh).squeeze(),:] \
                                    for timestamp in range(0, cam_poses.shape[0])])
        cat_3d_clr = np.concatenate([((colors[timestamp][(confs[timestamp].detach().cpu().reshape(-1,1)>conf_thresh).squeeze(),:]+1)/2*255).astype(np.uint32).tolist()\
                                    for timestamp in range(0, cam_poses.shape[0])])

        # downsample points
        cat_3d_wrld = cat_3d_wrld[::5]
        cat_3d_clr = cat_3d_clr[::5]

    else:
        focals = scene.get_focals()
        cam_poses = scene.get_im_poses()
        colors = [im["img"].squeeze().permute(1,2,0).reshape(-1, 3).detach().cpu().numpy() for im in imgs]

        pts3d, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
        #cat_3d_wrld = np.concatenate([p[m] for p, m in zip(pts3d, scene.get_masks())])
        cat_3d_wrld = np.concatenate(
            [pts[(confs[idx].reshape(-1,1)>conf_thresh).squeeze(),:]for idx, pts in enumerate(pts3d)] 
       )
        
        #cat_3d_clr = (np.concatenate([p[m] for p, m in zip(scene.imgs, scene.get_masks())]).reshape(-1, 1)*255).astype(np.uint32).tolist()
        cat_3d_clr = np.concatenate([((colors[timestamp][(confs[timestamp].reshape(-1,1)>conf_thresh).squeeze(),:]+1)/2*255).astype(np.uint32).tolist() \
                                    for timestamp in range(0, cam_poses.shape[0])])

        #downsample
        cat_3d_wrld = cat_3d_wrld[::10]
        cat_3d_clr = cat_3d_clr[::10]

    # set 3d points as static in the frame_idx timeline
    rr.log("world/3d", rr.Points3D(cat_3d_wrld, colors=cat_3d_clr), static=True)


    # update all transforms over the timestamps
    for timestamp in range(0, cam_poses.shape[0]):

        rr.set_time_sequence("frame_idx", timestamp)

        # wrld to camera TF
        rr.log("world/camera", rr.Transform3D(translation=cam_poses[timestamp][:3, 3].detach().cpu().numpy(), \
                                            rotation=rr.Quaternion(xyzw=R.from_matrix(cam_poses[timestamp][:3, :3].detach().cpu()).as_quat())))
        
        rr.log(f"world/camera{timestamp}", rr.Transform3D(translation=cam_poses[timestamp][:3, 3].detach().cpu().numpy(), \
                                            rotation=rr.Quaternion(xyzw=R.from_matrix(cam_poses[timestamp][:3, :3].detach().cpu()).as_quat())), static=True)
        # pinhole definition
        rr.log("world/camera", rr.Pinhole(width=imgs[timestamp]["img"].shape[3], height=imgs[timestamp]["img"].shape[2], focal_length=float(focals[timestamp].detach().cpu().numpy().squeeze())))

        # image in the pinhole
        rr.log("world/camera/rgb", rr.Image((imgs[timestamp]["img"].detach().cpu().squeeze().permute(1,2,0).numpy()+1)/2))

        # log depthmaps with confidence
        #rr.log("world/camera/depth", rr.DepthImage(depthmaps[timestamp].detach().cpu().numpy()))

    return cat_3d_clr
    

def rerun_pairs(pairs, tfs, output, scale_rerun=0.7):

    rr.init("3D Points and Camera Visualization")

    # use rerun to see transformation in an inference pair
    rr.notebook_show(width=int(1920*scale_rerun_w), height=int(1080*scale_rerun_w))

    for idx_pair, pair in tqdm(enumerate(pairs),total=len(pairs), ncols=100):

        if not (idx_pair % 10 == 0):
            continue

        #print("between img", pair[0]["idx"], pair[1]["idx"])

        downsample_factor = 4
        rr.set_time_sequence("idx_pair", idx_pair)

        rr.log(f"pair_0", rr.Transform3D(translation=np.zeros((1,3)), \
                                            rotation=rr.Quaternion(xyzw=np.array([0,0,0,1]))))
            
        rr.log(f"pair_1", rr.Transform3D(translation=tfs[pairs[idx_pair][0]["idx"]][pairs[idx_pair][1]["idx"]][:3, 3], \
                                            rotation=rr.Quaternion(xyzw=R.from_matrix(tfs[pairs[idx_pair][0]["idx"]][pairs[idx_pair][1]["idx"]][:3, :3]).as_quat())))
        
        
        rr.log(f"pair_0/3d", rr.Points3D(torch.reshape(output["pred1"]["pts3d"][idx_pair,:,:,:], (-1,3)).detach().cpu().numpy()[::downsample_factor],\
                                                colors =((torch.reshape(pairs[idx_pair][0]["img"].squeeze().permute(1,2,0),(-1,3))+1)/2*255).detach().cpu().numpy().astype(np.uint32).tolist()[::downsample_factor]))
        
        rr.log(f"pair_0/3d_other", rr.Points3D(torch.reshape(output["pred2"]["pts3d_in_other_view"][idx_pair,:,:,:], (-1,3)).detach().cpu().numpy()[::downsample_factor],\
                                                colors =((torch.reshape(pairs[idx_pair][1]["img"].squeeze().permute(1,2,0),(-1,3))+1)/2*255).detach().cpu().numpy().astype(np.uint32).tolist()[::downsample_factor]))    
        