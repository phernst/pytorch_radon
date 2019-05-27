import unittest
from pytorch_radon.utils import PI, SQRT2, deg2rad
import numpy as np

class TestStackgram(unittest.TestCase):
    def test_pi(self):
        self.assertAlmostEqual(PI.item(), np.pi, places=6)

    def test_sqrt2(self):
        self.assertAlmostEqual(SQRT2.item(), np.sqrt(2), places=6)

    def test_deg2rad(self):
        self.assertAlmostEqual(deg2rad(45).item(), np.deg2rad(45), places=6)

if __name__ == '__main__':
    unittest.main()