import torch

# constants
PI = 4*torch.ones(1).atan()
SQRT2 = (2*torch.ones(1)).sqrt()

def fftfreq(n):
    val = 1.0/n
    results = torch.zeros(n)
    N = (n-1)//2 + 1
    p1 = torch.arange(0, N)
    results[:N] = p1
    p2 = torch.arange(-(n//2), 0)
    results[N:] = p2
    return results*val

def deg2rad(x):
    return x*PI/180
