import unittest
import numpy as np
from neural_network import h1

class TestNeuralNetwork(unittest.TestCase):
    def test_h1(self):
        # Test case 1: Second feature greater than first
        X = np.array([[1, 2], [3, 4], [5, 6]])
        expected_output = np.array([1, 1, 1])
        np.testing.assert_array_equal(h1(X), expected_output)

        # Test case 2: Second feature less than first
        X = np.array([[2, 1], [4, 3], [6, 5]])
        expected_output = np.array([0, 0, 0])
        np.testing.assert_array_equal(h1(X), expected_output)

        # Test case 3: Second feature equal to first
        X = np.array([[1, 1], [2, 2], [3, 3]])
        expected_output = np.array([1, 1, 1])
        np.testing.assert_array_equal(h1(X), expected_output)

        # Test case 4: Mixed cases
        X = np.array([[1, 2], [4, 3], [3, 3], [5, 6], [6, 5]])
        expected_output = np.array([1, 0, 1, 1, 0])
        np.testing.assert_array_equal(h1(X), expected_output)

if __name__ == '__main__':
    unittest.main()