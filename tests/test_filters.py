import unittest

import numpy as np
from skimage.transform import radon as sk_radon, iradon as sk_iradon
import torch

from pytorch_radon import Radon, IRadon
from pytorch_radon.filters import RampFilter, HannFilter, LearnableFilter

class TestFilters(unittest.TestCase):
    def test_ramp_filter(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle, use_filter=RampFilter())
        reco = ir(r(img))
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=4)

    def test_hann_filter(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle, use_filter=HannFilter())
        reco = ir(r(img))
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=3)

    def test_learnable_filter(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle, use_filter=LearnableFilter(img.shape[2]))
        reco = ir(r(img))
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=4)

    def test_hann_radon_iradon_double_scikit(self):
        img = torch.zeros(1, 1, 256, 256, dtype=torch.double)
        img[:, :, 120:130, 120:130] = 1
        circle = False
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle, dtype=torch.double)
        ir = IRadon(img.shape[2], theta, circle, dtype=torch.double, use_filter=HannFilter())
        sino = r(img)
        reco = ir(sino)

        sino_sk = sk_radon(img.numpy().squeeze(), theta=theta.numpy(), circle=circle)
        reco_sk = sk_iradon(sino_sk, theta=theta.numpy(), circle=circle, filter_name="hann")

        self.assertAlmostEqual(np.mean(np.abs(sino_sk - sino.numpy().squeeze())), 0, places=14)
        self.assertAlmostEqual(np.mean(np.abs(reco_sk - reco.numpy().squeeze())), 0, places=10)

if __name__ == '__main__':
    unittest.main()
