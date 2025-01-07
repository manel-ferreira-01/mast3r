import cv2 as cv
import alphashape
import shapely
import torch
import time
import numpy as np

def is_near_motion_mask(motion_mask, x, y, distance=5):
    x = int(x)
    y = int(y)
    for i in range(-distance, distance):
        for j in range(-distance, distance):
            if motion_mask[y + i, x + j]:
                return True
    return False

def find_mask(cotracker_model, device, img1, img2, grid_size=45, epipolarReproj=1.0, alpha= 0.05,
              raft_model=None, model="homography"):

    if raft_model is None:
        two_frame_tensor = torch.cat([img1["img"], img2["img"]], dim=0).unsqueeze(0).to(device)

        pred_tracks, pred_visibility = cotracker_model(two_frame_tensor, grid_size=grid_size)

        pts1 = pred_tracks[0,0,:,:].cpu().numpy()
        pts2 = pred_tracks[0,1,:,:].cpu().numpy()

    else:
        start = time.time()
        flow = raft_model(img1["img"].to(device), img2["img"].to(device))[-1]
        H, W = torch.tensor(img1["img"].shape[2:] )

        pts1 = np.mgrid[:int(W), :int(H)].T.astype(np.float32)

        pts2 = pts1 + flow.detach().squeeze().permute(1,2,0).cpu().numpy()

        pts1 = pts1.reshape(-1, 2)
        pts2 = pts2.reshape(-1, 2)

        #print("Time to compute flow: ", time.time()-start)

    if not "homography":

        # Check for visibility here
        #visibility_mask = pred_visibility.cpu().numpy().squeeze()[0,:]
        #pts1 = pts1[visibility_mask == 1]
        #pts2 = pts2[visibility_mask == 1]

        F = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, epipolarReproj)

        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv.dilate(F[1].astype(np.uint8), kernel, iterations=5)
        
        if raft_model is None:
            return 1-cv.resize(np.array(F[1]==0).reshape(grid_size, grid_size).astype(np.uint8), (img2["img"].shape[3], img2["img"].shape[2]))
        else:
            return 1-np.array(F[1]==0).reshape(img2["img"].shape[2], img2["img"].shape[3])

    else:

        flow = pts2 - pts1
        
        H, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

        coords = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        transformed_coords = H @ coords.T
        transformed_coords /= transformed_coords[2]  # Normalize homogeneous coords

        global_flow_x = (transformed_coords[0] - coords[:, 0])
        global_flow_y = (transformed_coords[1] - coords[:, 1])
        global_flow = np.stack((global_flow_x, global_flow_y), axis=-1)

        residual_flow = global_flow - flow

        magnitude = np.sqrt(residual_flow[..., 0]**2 + residual_flow[..., 1]**2)

        #normalize the magnitude
        magnitude /= np.max(magnitude)

        motion_mask = magnitude > 0.15
        
        #grid_resized = cv.resize(motion_mask.reshape(grid_size, grid_size).astype(np.uint8),\
        #                          (img2["img"].shape[3], img2["img"].shape[2]))

        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv.dilate(motion_mask.astype(np.uint8), kernel, iterations=5)

        if raft_model is not None:
            return 1-motion_mask.reshape(img2["img"].shape[2], img2["img"].shape[3]).astype(np.int64)
        else:
            return 1-cv.resize(motion_mask.reshape(grid_size, grid_size).astype(np.uint8),\
                                  (img2["img"].shape[3], img2["img"].shape[2]))
