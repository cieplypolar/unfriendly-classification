import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -100, 100)
    return 1 / (1 + np.exp(-z))

def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def ReLU_backward(Z_curr):
    return np.where(Z_curr > 0, 1, 0)
