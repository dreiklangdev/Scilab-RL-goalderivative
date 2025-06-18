
import numpy as np

class EMAWhitening:
    def __init__(self, dim, alpha=0.01, eps=1e-5):
        self.alpha = alpha
        self.eps = eps
        self.mu = np.zeros(dim)
        self.cov = np.eye(dim)
        self.dim = dim

    def update(self, x):
        # Update mean
        self.mu = (1 - self.alpha) * self.mu + self.alpha * x

        # Demeaned input
        x_centered = x - self.mu

        # Update covariance matrix
        self.cov = (1 - self.alpha) * self.cov + self.alpha * np.outer(x_centered, x_centered)

        # Compute whitening transform: inverse sqrt of covariance
        eigvals, eigvecs = np.linalg.eigh(self.cov + self.eps * np.eye(self.dim))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        whitening_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T

        # Whitened vector
        x_whitened = whitening_matrix @ x_centered

        return x_whitened