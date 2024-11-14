import numpy as np

def h1(X):
    """Classifies whether the second feature is greater or equal to first.

    Input:
         X: N,D design matrix of input features

    Output:
        yy: N, output 1 if second feature greater, 0 otherwise
    """
    X = np.array(X)
    w11 = -1
    w12 = 1
    b1 = 0
    ww1 = np.array([w11, w12])
    # `aa` is a linear combination of the inputs
    aa = np.dot(X, ww1) + b1
    # `yy` is the output after hard thresholding
    yy = np.maximum(aa, np.zeros(X.shape[0]))
    return yy.astype(int)

def main():
    test_case_1 = np.array([[1, 2], [3, 4], [5, 6]])
    assert np.array_equal(h1(test_case_1), np.array([1, 1, 1]))
    print(h1(test_case_1))  # Expected output: [1 1 1]

    test_case_2 = np.array([[2, 1], [4, 3], [6, 5]])
    assert np.array_equal(h1(test_case_2), np.array([0, 0, 0]))
    print(h1(test_case_2))  # Expected output: [0 0 0]

if __name__ == '__main__':
    main()

    
