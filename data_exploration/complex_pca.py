"""Complex-valued PCA compatible with scikit-learn PCA.
Written by: https://datascience.stackexchange.com/users/69793/alex
Original post: https://datascience.stackexchange.com/a/75875
"""

import numpy as np


class ComplexPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.u = self.s = self.components_ = None
        self.mean_ = None

    @property
    def explained_variance_ratio_(self):
        return self.s

    def fit(self, matrix):
        self.mean_ = matrix.mean(axis=0)
        _, self.s, vh = np.linalg.svd(matrix, full_matrices=False)  # full=False ==> num_pc = min(N, M)
        # It would be faster if the SVD was truncated to only n_components instead of min(M, N)
        self.components_ = vh  # already conjugated.
        # Leave those components as rows of matrix so that it is compatible with Sklearn PCA.

    def transform(self, matrix):
        data = matrix - self.mean_
        result = data @ self.components_.T
        return result

    def inverse_transform(self, matrix):
        result = matrix @ np.conj(self.components_)
        return self.mean_ + result
