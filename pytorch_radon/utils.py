import torch
import torch.nn.functional as F

if torch.__version__ > '1.2.0':
    def affine_grid(theta, size):
        return F.affine_grid(theta, size, align_corners=True)

    def grid_sample(x, grid, mode='bilinear'):
        return F.grid_sample(x, grid, align_corners=True, mode=mode)
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample

# constants
PI = 4*torch.ones(1, dtype=torch.double).atan()
SQRT2 = (2*torch.ones(1, dtype=torch.double)).sqrt()


def deg2rad(x, dtype=torch.float):
    return (x*PI/180).to(dtype)
