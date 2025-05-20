import numpy as np
from abc import ABC, abstractmethod

class BaseKernel(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def __str__(self):
        pass

class LinearKernel(BaseKernel):
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return x @ y.T
    def __str__(self):
        return "Linear Kernel"


'''
I stole this piece of code ;)
'''
class GaussianKernel(BaseKernel):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]

        if np.ndim(y) == 1:
            y = y[np.newaxis, :]

        d = np.linalg.norm(x[:, :, np.newaxis] - y.T[np.newaxis, :, :], axis=1) ** 2

        return np.exp(-d / (2 * self.sigma ** 2))
    
    def __str__(self):
        return f"Gaussian Kernel"