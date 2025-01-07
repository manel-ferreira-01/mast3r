import rerun as rr
import numpy as np
import time
from scipy.spatial.transform import Rotation as R


# Function to add 3D points to the visualization
def log_points(timestamp, points, colors):
    print(len(points))
    rr.log("world/points", rr.Points3D(points, colors =colors))

# Function to add the camera pose to the visualization
def log_camera_pose(timestamp, pose):
    translation = pose[:3, 3]
    rot = pose[:3, :3]

    rot = R.from_matrix(rot)

    rr.log(
            "world/camera/depthmpas",
            rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=rot.as_quat())),
    )
