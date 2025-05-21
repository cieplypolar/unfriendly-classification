import numpy as np
from src.classification import BaseModel

'''
Features should be numbered from 0 to d-1
Labels must be >= 0, I am using np.bincount
Also they must be {0, 1}
'''
class TreeNode:
    def __init__(self, feature: int, threshold: float, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

class DecisionTree(BaseModel):
    def __init__(self, no_features: int, max_depth: int = 8, min_samples_split: int = 2):
        self.no_features = no_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        y_scaled = (y + 1) // 2  # convert -1, 1 to 0, 1
        y_scaled = y_scaled.flatten()
        self.root = self._build_tree(X, y_scaled, 0)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = self.root
            while node.left is not None and node.right is not None:
                if X[i, node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions[i] = node.label
        return ((predictions - 0.5) * 2).astype(int).reshape(-1, 1)  # convert to -1, 1 labels

    def reset(self):
        self.root = None

    def __str__(self):
        return "Decision Tree"

    def _build_tree(self, X, y, depth):
        m, d = X.shape
        no_labels = len(np.unique(y))

        if depth >= self.max_depth or m < self.min_samples_split or no_labels == 1:
            label = np.bincount(y).argmax()
            return TreeNode(feature=None, threshold=None, label=label)

        best_gini = 1.0
        best_feature = None
        best_threshold = None
        best_splits = None

        features = np.random.choice(d, self.no_features, replace=False)

        for feature in features:
            sorted_idx = np.argsort(X[:, feature])
            sorted_X = X[sorted_idx, feature]
            # fancy way for computing means of adjacent elements
            thresholds = (sorted_X[:-1] + sorted_X[1:]) / 2
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    # does not split
                    continue

                gini = self.__gini(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
                    best_splits = (left_mask, right_mask)


        if best_feature is None:
            label = np.bincount(y).argmax()
            return TreeNode(feature=None, threshold=None, label=label)

        left = self._build_tree(X[best_splits[0]], y[best_splits[0]], depth + 1)
        right = self._build_tree(X[best_splits[1]], y[best_splits[1]], depth + 1)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)

    '''
    Hardocoded for {0, 1}
    '''
    def __gini(self, y_left: np, y_right: np.ndarray) -> float:
        gini_left = 1.0 - sum((np.sum(y_left == c) / y_left.shape[0]) ** 2 for c in [0,1])
        gini_right = 1.0 - sum((np.sum(y_right == c) / y_right.shape[0]) ** 2 for c in [0,1])
        gini = (y_left.shape[0] * gini_left + y_right.shape[0] * gini_right) / (y_left.shape[0] + y_right.shape[0])
        return gini