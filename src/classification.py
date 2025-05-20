import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def squared_mean_error(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def zero_one_error(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y != y_pred)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    '''
    Hardcoded values for y {-1, 1}
    '''
    def precision(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        true_positives = np.sum((y == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else np.nan

    def sensitivity(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        true_positives = np.sum((y == 1) & (y_pred == 1))
        actual_positives = np.sum(y == 1)
        return true_positives / actual_positives if actual_positives > 0 else np.nan