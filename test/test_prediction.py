import unittest

import numpy as np


# TODO: download an actual model instead of using random init and then properly evaluate the output.
class TestPrediction(unittest.TestCase):
    def test_run_counting(self):
        from stacc.prediction import run_counting
        from stacc.unet_2d import UNet2d

        model = UNet2d(in_channels=1, out_channels=1)
        image = np.random.rand(256, 256)
        coords = run_counting(model, image)
        self.assertEqual(coords.ndim, 2)
        self.assertEqual(coords.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
