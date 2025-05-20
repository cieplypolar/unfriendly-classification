import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.classification import BaseModel
from scipy.stats import chi2_contingency

def split_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = data[:, :-1]
    y = data[:, -1].reshape((-1, 1))
    return X, y


def divide_dataset(
    data: np.ndarray, fractions_train_val_test: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    assert len(fractions_train_val_test) == 3
    train_ratio = fractions_train_val_test[0]
    val_ratio = fractions_train_val_test[1]
    # test_ratio - the rest

    classes = np.unique(data[:, -1])
    train, val, test = [], [], []

    for c in classes:
        class_data = data[data[:, -1] == c]
        np.random.shuffle(class_data)
        n_class = len(class_data)
        n_train = int(n_class * train_ratio)
        n_val = int(n_class * val_ratio)

        train.append(class_data[:n_train])
        val.append(class_data[n_train : n_train + n_val])
        test.append(class_data[n_train + n_val :])

    train = np.vstack(train)
    val = np.vstack(val)
    test = np.vstack(test)

    np.random.shuffle(train)
    # latter 2 lines are not necessary
    np.random.shuffle(val)
    np.random.shuffle(test)

    return train, val, test


def chisq(df: pd.DataFrame) -> None:
    columns = df.columns

    chi2_matrix = pd.DataFrame(
        np.zeros((len(columns), len(columns))), index=columns, columns=columns
    )

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i != j:
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                chi2_matrix.loc[col1, col2] = chi2

    plt.figure(figsize=(10, 8))
    sns.heatmap(chi2_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Chi-Square Heatmap")
    plt.show()


def get_random_splits(
    data: np.ndarray, fractions_train_val_test: list[float], T: int = 10
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    datasets = []
    for _ in range(T):
        datasets.append(
            divide_dataset(
                data.copy(), fractions_train_val_test=fractions_train_val_test
            )
        )
    return datasets