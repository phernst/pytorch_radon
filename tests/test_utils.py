import unittest

import numpy as np
import torch

from pytorch_radon.utils import PI, SQRT2, deg2rad

class TestUtils(unittest.TestCase):
    def test_pi(self):
        self.assertAlmostEqual(PI.item(), np.pi)

    def test_sqrt2(self):
        self.assertAlmostEqual(SQRT2.item(), np.sqrt(2))

    def test_deg2rad(self):
        self.assertAlmostEqual(deg2rad(45, dtype=torch.double).item(), np.deg2rad(45))

if __name__ == '__main__':
    unittest.main()
