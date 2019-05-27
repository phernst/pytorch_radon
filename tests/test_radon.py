import unittest
from pytorch_radon import Radon, IRadon
import torch

class TestRadon(unittest.TestCase):
    def test_radon_iradon_circle(self):
        img = torch.zeros(1,1,256,256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=4)

    def test_radon_iradon_not_circle(self):
        img = torch.zeros(1,1,256,256)
        img[:, :, 120:130, 120:130] = 1
        circle = False
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=4)

if __name__ == '__main__':
    unittest.main()