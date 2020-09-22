import torch
import torch.nn.functional as F

if torch.__version__ > '1.2.0':
    affine_grid = lambda theta, size: F.affine_grid(theta, size, align_corners=True)
    grid_sample = lambda input, grid, mode='bilinear': F.grid_sample(input, grid, align_corners=True, mode=mode)
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample

# constants
PI = 4*torch.ones(1, dtype=torch.double).atan()
SQRT2 = (2*torch.ones(1, dtype=torch.double)).sqrt()

def deg2rad(x, dtype=torch.float):
    return (x*PI/180).to(dtype)

def rfft(tensor, axis=-1):
    ndim = tensor.ndim
    if axis < 0:
        axis %= ndim
    tensor = tensor.transpose(axis, ndim-1)
    fft_tensor = torch.rfft(
        tensor,
        1,
        normalized=False,
        onesided=False,
    )
    return fft_tensor.transpose(axis, ndim-1)

def irfft(tensor, axis):
    assert 0 <= axis < tensor.ndim
    tensor = tensor.transpose(axis, tensor.ndim-2)
    ifft_tensor = torch.ifft(
        tensor,
        1,
        normalized=False,
    )[..., 0]
    return ifft_tensor.transpose(axis, tensor.ndim-2)
