import torch

# constants
PI = 4*torch.ones(1, dtype=torch.double).atan()
SQRT2 = (2*torch.ones(1, dtype=torch.double)).sqrt()


def deg2rad(x, dtype=torch.float):
    return (x*PI/180).to(dtype)
