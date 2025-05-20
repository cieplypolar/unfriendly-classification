import numpy as np
from abc import ABC, abstractmethod

class BaseGradient(ABC):
    @abstractmethod
    def calculate_gradient(
        self,
        regularization: float,
        alpha: float,
        X: np.ndarray,
        theta: np.ndarray,
        intercept: np.float64,
        y_pred: np.ndarray,
        y: np.ndarray) -> tuple[np.float64, np.ndarray]:
        pass

class GradientDescent(BaseGradient):
    def calculate_gradient(
        self,
        regularization: float,
        alpha: float,
        X: np.ndarray,
        theta: np.ndarray,
        intercept: np.float64,
        y_pred: np.ndarray,
        y: np.ndarray) -> tuple[np.float64, np.ndarray]:

        m = X.shape[0]
        regularization_term = regularization * (2 * theta)
        error = y_pred - y
        gradient = X.T @ error + regularization_term
        intercept_gradient = np.sum(error)
        return intercept_gradient * (alpha / m), gradient * (alpha / m)

'''
Weighted means *= m :)
'''
class GradientDescentWithWeightedIntercept(BaseGradient):
    def calculate_gradient(
        self,
        regularization: float,
        alpha: float,
        X: np.ndarray,
        theta: np.ndarray,
        intercept: np.float64,
        y_pred: np.ndarray,
        y: np.ndarray) -> tuple[np.float64, np.ndarray]:

        m = X.shape[0]
        regularization_term = regularization * (2 * theta)
        error = y_pred - y
        gradient = X.T @ error + regularization_term
        intercept_gradient = np.sum(error)
        return intercept_gradient * alpha, gradient * (alpha / m)

class NormalizedGradientDescent(BaseGradient):
    def calculate_gradient(
        self,
        regularization: float,
        alpha: float,
        X: np.ndarray,
        theta: np.ndarray,
        intercept: np.float64,
        y_pred: np.ndarray,
        y: np.ndarray) -> tuple[np.float64, np.ndarray]:

        m = X.shape[0]
        regularization_term = regularization * (2 * theta)
        error = y_pred - y
        gradient = (X.T @ error + regularization_term) * alpha / m
        intercept_gradient = np.sum(error) * alpha / m
        norm = np.linalg.norm(np.concatenate(([intercept_gradient], gradient.ravel())))
        if norm == 0:
            print("Why did we not break for loop??")
            norm = 1e-5
        return intercept_gradient / norm, gradient / norm



