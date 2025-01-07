import torch
import torch.nn as nn
from dust3r.post_process import estimate_focal_knowing_depth 

## Global Adam optimizer to calibrate multiple cameras with corresponding depthmaps
class adam_alignner():
    def __init__(self, device, imgs, pts3d):

        assert len(imgs) == len(pts3d)

        self.scales, self.quats, self.trans, self.focals, self.pps = [], [], [], [], []

        # define parameters to quaternitions, translations and scales
        for i in range(0, len(imgs)):
            self.scales.append(nn.Parameter((torch.tensor([1.0], device=device))))
            self.quats.append(nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)))
            self.trans.append(nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device)))

            #calculate focals
            H, W = torch.tensor(imgs[0]["img"].shape[2:] )
            self.pps.append(torch.tensor([W / 2, H / 2]).to(device))
            self.focals.append(estimate_focal_knowing_depth(pts3d[i], self.pps[i], focal_mode='weiszfeld'))

        # parameters that require gradient
        self.params = self.scales + self.quats + self.trans

        # define optimizer
        self.optimizer = torch.optim.Adam(self.params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)

    def optimize(self):
        self.optimizer.zero_grad()
        self.lr_schedule.step()

        loss = self.loss_3d()

        loss.backward()
        self.optimizer.step()

        # normalize quaternions
        for i in range(0, len(self.imgs)):
            self.quats[i].data = self.quats[i].data / self.quats[i].data.norm()