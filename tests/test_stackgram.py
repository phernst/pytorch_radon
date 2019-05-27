import unittest
from pytorch_radon import Radon, IRadon, Stackgram, IStackgram
import torch

class TestStackgram(unittest.TestCase):
    def test_stackgram_istackgram_circle(self):
        img = torch.zeros(1,1,256,256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle)
        sg = Stackgram(img.shape[2], theta, circle)
        isg = IStackgram(img.shape[2], theta, circle)
        reco = ir(isg(sg(r(img))))
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=3)

    def test_stackgram_istackgram_not_circle(self):
        img = torch.zeros(1,1,256,256)
        img[:, :, 120:130, 120:130] = 1
        circle = False
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle)
        sg = Stackgram(img.shape[2], theta, circle)
        isg = IStackgram(img.shape[2], theta, circle)
        reco = ir(isg(sg(r(img))))
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=3)

if __name__ == '__main__':
    unittest.main()