import torch
from torch import nn
import torch.nn.functional as F

from .utils import PI, SQRT2, deg2rad
from .filters import RampFilter

class Radon(nn.Module):
    def __init__(self, in_size, theta=None, circle=True):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.all_grids = self._create_grids(self.theta, in_size if circle else int((SQRT2*in_size).ceil()))

    def forward(self, x):
        N, C, W, H = x.shape
        assert(W==H)
        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, C, W, len(self.theta), device=x.device)

        for i in range(len(self.theta)):
            rotated = F.grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            out[...,i] = rotated.sum(2)

        return out

    def _create_grids(self, angles, grid_size):
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta)
            R = torch.tensor([[
                    [ theta.cos(), theta.sin(), 0],
                    [-theta.sin(), theta.cos(), 0],
                ]])
            all_grids.append(F.affine_grid(R, torch.Size([1, 1, grid_size, grid_size])))
        return all_grids

class IRadon(nn.Module):
    def __init__(self, out_size, theta=None, circle=True, use_filter=RampFilter()):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size = out_size if circle else int((SQRT2*out_size).ceil())
        self.xgrid = torch.arange(in_size).float().view(1,-1).repeat(in_size, 1)*2/(in_size-1)-1
        self.ygrid = torch.arange(in_size).float().view(-1,1).repeat(1, in_size)*2/(in_size-1)-1
        self.all_grids = self._create_grids(self.theta, in_size)
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        x = self.filter(x)

        reco = torch.zeros(x.shape[0], ch_size, it_size, it_size).to(x.device)
        for i_theta in range(len(self.theta)):
            reco += F.grid_sample(x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1).to(x.device))

        if not self.circle:
            W = self.out_size
            diagonal = self.in_size
            pad = int(torch.tensor(diagonal - W).float().ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]))

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.
        
        return reco*PI.item()/(2*len(self.theta))

    def _XYtoT(self, theta):
        T = self.xgrid*(deg2rad(theta)).cos() - self.ygrid*(deg2rad(theta)).sin()
        return T

    def _create_grids(self, angles, grid_size):
        all_grids = []
        for i_theta in range(len(angles)):
            X = torch.ones(grid_size).float().view(-1,1).repeat(1, grid_size)*i_theta*2./(len(angles)-1)-1.
            Y = self._XYtoT(angles[i_theta])
            all_grids.append(torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1).unsqueeze(0))
        return all_grids