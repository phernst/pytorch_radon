import torch
from torch import nn
import torch.nn.functional as F

from .utils import PI, rfft, irfft

def ramp_filter(size):
    image_n = torch.cat([
        torch.arange(1, size / 2 + 1, 2),
        torch.arange(size / 2 - 1, 0, -2)
    ])

    image_filter = torch.zeros(size)
    image_filter[0] = 0.25
    image_filter[1::2] = -1 / (PI * image_n) ** 2

    fourier_filter = torch.rfft(image_filter, 1, onesided=False)
    fourier_filter[:, 1] = fourier_filter[:, 0]

    return 2*fourier_filter

class AbstractFilter(nn.Module):
    def forward(self, x):
        input_size = x.shape[2]
        projection_size_padded = \
            max(64, int(2**(2*torch.tensor(input_size)).float().log2().ceil()))
        pad_width = projection_size_padded - input_size
        padded_tensor = F.pad(x, (0, 0, 0, pad_width))
        fourier_filter = ramp_filter(padded_tensor.shape[2]).to(x.device)
        fourier_filter = self.create_filter(fourier_filter)
        fourier_filter = fourier_filter.unsqueeze(-2)
        projection = rfft(padded_tensor, axis=2, onesided=False)*fourier_filter
        return irfft(projection, axis=2, onesided=False)[:, :, :input_size, :]

    def create_filter(self, fourier_ramp):
        raise NotImplementedError

class RampFilter(AbstractFilter):
    def create_filter(self, fourier_ramp):
        return fourier_ramp

class HannFilter(AbstractFilter):
    def create_filter(self, fourier_ramp):
        n = torch.arange(0, fourier_ramp.shape[0])
        hann = (0.5 - 0.5*(2.0*PI*n/(fourier_ramp.shape[0]-1)).cos()).to(fourier_ramp.device)
        return fourier_ramp*hann.roll(hann.shape[0]//2, 0).unsqueeze(-1)

class LearnableFilter(AbstractFilter):
    def __init__(self, filter_size):
        super(LearnableFilter, self).__init__()
        self.filter = nn.Parameter(ramp_filter(filter_size)[..., 0].view(-1, 1))

    def forward(self, x):
        fourier_filter = self.filter.unsqueeze(-1).repeat(1, 1, 2).to(x.device)
        projection = rfft(x, axis=2, onesided=False) * fourier_filter
        return irfft(projection, axis=2, onesided=False)

    def create_filter(self, fourier_ramp):
        raise NotImplementedError
