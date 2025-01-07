import numpy as np
from dust3r.utils.device import to_numpy
from scipy.io import savemat
from pypcd4 import PointCloud

class SceneToData:
    def __init__(self, scene):
        self.scene = scene
        self.pts3d, self.pts3d_clr, self.confs_wrld, self.focals, self.cams2world, self.rgbimgs, self.depthmaps, self.confs = self.scene_extractor()

    def scene_extractor(self):
        mask = to_numpy(self.scene.get_masks())
        try:
            pts3d, depthmaps, confs = to_numpy(self.scene.get_dense_pts3d(clean_depth=True))
            pts3d_clr = to_numpy(self.scene.get_pts3d_colors())
        except:
            print("its scene form dust3r")
            pts3d = self.scene.depth_to_pts3d()
            depthmaps = self.scene.get_depthmaps()

        rgbimgs = self.scene.imgs
        focals = self.scene.get_focals().cpu()
        cams2world = self.scene.get_im_poses().cpu()

        pts3d = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        pts3d_clr = np.concatenate([p[m] for p, m in zip(rgbimgs, mask)])
        pts3d_clr = pts3d_clr.reshape(-1, 3)

        confs_wrld = np.concatenate([p[m] for p, m in zip(confs, mask)])
        confs_wrld = confs_wrld.reshape(-1, 1) / np.max(confs_wrld)

        return pts3d, pts3d_clr, confs_wrld, focals, cams2world, rgbimgs, depthmaps, confs

    def save_mat_file(self, outdir):
        # create a dict about world info
        wrld_info = {
            'wrld': {
                'xyz': self.pts3d,
                'color': self.pts3d_clr,
                'conf': self.confs_wrld,
            }
        }
        savemat(outdir + '/wrld_info.mat', wrld_info)

        cell_array = np.empty((len(self.cams2world), 1), dtype=object)
        for i in range(cell_array.shape[0]):
            cell_array[i, 0] = {
                'extrinsics': to_numpy(self.cams2world[i]),
                'intrinsics': to_numpy(self.focals[i]),
                'rgb': to_numpy(self.rgbimgs[i]),
                'depth': np.reshape(self.depthmaps[i], (to_numpy(self.rgbimgs[i]).shape[0], to_numpy(self.rgbimgs[i]).shape[1])),
                'conf': self.confs[i]
            }
        data = {
            'cams_info': cell_array  # Saving the cell array as 'myCellArray'
        }
        savemat(outdir + '/cams_info.mat', data)

    def save_pcd_file(self, outdir, conf_threshold=0.01):
        rgb_unit8 = np.array([PointCloud.encode_rgb(np.array(self.pts3d_clr * 255).astype(np.uint8))]).T
        pc = PointCloud.from_xyzrgb_points(np.hstack([self.pts3d[(self.confs_wrld > conf_threshold).squeeze(), :], rgb_unit8[(self.confs_wrld > conf_threshold).squeeze(), :]]))

        pc.save(outdir + '/scene.pcd')
