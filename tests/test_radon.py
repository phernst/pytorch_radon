import unittest

import numpy as np
from skimage.transform import radon as sk_radon, iradon as sk_iradon
import torch

from pytorch_radon import Radon, IRadon

class TestRadon(unittest.TestCase):
    def test_radon_iradon_circle(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=4)

    def test_radon_iradon_circle_cut_output(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle, out_size=128)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img[:, :, 64:192, 64:192], reco).item(), 0, places=3)

    def test_radon_iradon_circle_lazy(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(theta=theta, circle=circle)
        ir = IRadon(theta=theta, circle=circle)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=3)

    def test_radon_iradon_circle_lazy_cut_output(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(theta=theta, circle=circle)
        ir = IRadon(theta=theta, circle=circle, out_size=128)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img[:, :, 64:192, 64:192], reco).item(), 0, places=3)

    def test_radon_iradon_not_circle(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = False
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=4)

    def test_radon_iradon_not_circle_cut_output(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = False
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle)
        ir = IRadon(img.shape[2], theta, circle, out_size=128)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img[:, :, 64:192, 64:192], reco).item(), 0, places=3)

    def test_radon_iradon_not_circle_lazy(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = False
        theta = torch.arange(180)
        r = Radon(theta=theta, circle=circle)
        ir = IRadon(theta=theta, circle=circle)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=3)

    def test_radon_iradon_not_circle_lazy_cut_output(self):
        img = torch.zeros(1, 1, 256, 256)
        img[:, :, 120:130, 120:130] = 1
        circle = False
        theta = torch.arange(180)
        r = Radon(theta=theta, circle=circle)
        ir = IRadon(theta=theta, circle=circle, out_size=128)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img[:, :, 64:192, 64:192], reco).item(), 0, places=3)

    def test_radon_iradon_circle_double(self):
        img = torch.zeros(1, 1, 256, 256, dtype=torch.double)
        img[:, :, 120:130, 120:130] = 1
        circle = True
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle, dtype=torch.double)
        ir = IRadon(img.shape[2], theta, circle, dtype=torch.double)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=3)

    def test_radon_iradon_not_circle_double(self):
        img = torch.zeros(1, 1, 256, 256, dtype=torch.double)
        img[:, :, 120:130, 120:130] = 1
        circle = False
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle, dtype=torch.double)
        ir = IRadon(img.shape[2], theta, circle, dtype=torch.double)
        sino = r(img)
        reco = ir(sino)
        self.assertAlmostEqual(torch.nn.MSELoss()(img, reco).item(), 0, places=3)

    def test_radon_iradon_not_circle_double_scikit(self):
        img = torch.zeros(1, 1, 256, 256, dtype=torch.double)
        img[:, :, 120:130, 120:130] = 1
        circle = False
        theta = torch.arange(180)
        r = Radon(img.shape[2], theta, circle, dtype=torch.double)
        ir = IRadon(img.shape[2], theta, circle, dtype=torch.double, use_filter=None)
        sino = r(img)
        reco = ir(sino)

        sino_sk = sk_radon(img.numpy().squeeze(), theta=theta.numpy(), circle=circle)
        reco_sk = sk_iradon(sino_sk, theta=theta.numpy(), circle=circle, filter_name=None)

        self.assertAlmostEqual(np.mean((sino_sk - sino.numpy().squeeze())**2), 0, places=27)
        self.assertAlmostEqual(np.mean((reco_sk - reco.numpy().squeeze())**2), 0, places=28)

if __name__ == '__main__':
    unittest.main()
