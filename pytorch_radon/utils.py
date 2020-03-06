import torch
import torch.nn.functional as F

if torch.__version__ > '1.2.0':
    affine_grid = lambda theta, size: F.affine_grid(theta, size, align_corners=True)
    grid_sample = lambda input, grid, mode='bilinear': F.grid_sample(input, grid, align_corners=True, mode=mode)
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample

# constants
PI = 4*torch.ones(1).atan()
SQRT2 = (2*torch.ones(1)).sqrt()

def deg2rad(x):
    return x*PI/180

def rfft(tensor, axis=-1, normalized=False, onesided=True):
    ndim = tensor.ndim
    if axis < 0:
        axis %= ndim
    tensor = tensor.transpose(axis, ndim-1)
    fft_tensor = torch.rfft(tensor, 1, normalized=normalized, onesided=onesided)
    return fft_tensor.transpose(axis, ndim-1)

def irfft(tensor, axis, normalized=False, onesided=True):
    assert 0 <= axis < tensor.ndim
    tensor = tensor.transpose(axis, tensor.ndim-2)
    ifft_tensor = torch.irfft(
        tensor,
        1,
        normalized=normalized,
        onesided=onesided)
    return ifft_tensor.transpose(axis, tensor.ndim-2)
