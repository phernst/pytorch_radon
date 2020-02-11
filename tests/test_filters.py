import unittest
from pytorch_radon import Radon, IRadon
from pytorch_radon.filters import RampFilter, HannFilter, LearnableFilter
import torch

class TestFilters(unittest.TestCase):
    def test_ramp_filter(self):
        img = torch.zeros(1,1,256,256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle, use_filter=RampFilter())
        reco = ir(r(img))
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=4)

    def test_hann_filter(self):
        img = torch.zeros(1,1,256,256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle, use_filter=HannFilter())
        reco = ir(r(img))
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=3)

    def test_learnable_filter(self):
        img = torch.zeros(1,1,256,256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle, use_filter=LearnableFilter(img.shape[2]))
        reco = ir(r(img))
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=4)

if __name__ == '__main__':
    unittest.main()