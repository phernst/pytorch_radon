import torch
from torch import nn
import torch.nn.functional as F

from .utils import PI, fftfreq

class AbstractFilter(nn.Module):
    def __init__(self):
        super(AbstractFilter, self).__init__()

    def forward(self, x):
        input_size = x.shape[2]
        projection_size_padded = \
            max(64, int(2 ** (2 * torch.tensor(input_size)).float().log2().ceil()))
        pad_width = projection_size_padded - input_size
        padded_tensor = F.pad(x, (0,0,0,pad_width))
        f = fftfreq(padded_tensor.shape[2]).view(-1, 1).to(x.device)
        fourier_filter = self.create_filter(f)
        fourier_filter = fourier_filter.unsqueeze(-1).repeat(1,1,2)
        projection = torch.rfft(padded_tensor.transpose(2,3), 1, onesided=False).transpose(2,3) * fourier_filter
        return torch.irfft(projection.transpose(2,3), 1, onesided=False).transpose(2,3)[:,:,:input_size,:]

    def create_filter(self, f):
        raise NotImplementedError

class RampFilter(AbstractFilter):
    def __init__(self):
        super(RampFilter, self).__init__()

    def create_filter(self, f):
        return 2 * f.abs()

class HannFilter(AbstractFilter):
    def __init__(self):
        super(HannFilter, self).__init__()

    def create_filter(self, f):
        fourier_filter = 2 * f.abs()
        omega = 2*PI*f
        fourier_filter *= (1 + (omega / 2).cos()) / 2
        return fourier_filter

class LearnableFilter(AbstractFilter):
    def __init__(self, filter_size):
        super(LearnableFilter, self).__init__()
        self.filter = nn.Parameter(2*fftfreq(filter_size).abs().view(-1, 1))

    def forward(self, x):
        fourier_filter = self.filter.unsqueeze(-1).repeat(1,1,2).to(x.device)
        projection = torch.rfft(x.transpose(2,3), 1, onesided=False).transpose(2,3) * fourier_filter
        return torch.irfft(projection.transpose(2,3), 1, onesided=False).transpose(2,3)