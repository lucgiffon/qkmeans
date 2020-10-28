import unittest
import numpy as np

from qkmeans.core.utils import proj_onto_l1_ball


class MyTestCase(unittest.TestCase):
    def test_something(self):
        vec = np.random.power(1, 100)
        proj_vec = proj_onto_l1_ball(1, 1e-5, vec)
        print(vec, proj_vec)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
