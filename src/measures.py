import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from src.classification import BaseModel
from src.utilz import split_data


def average_error(
        model: BaseModel,
        datasets: list[np.ndarray]
    ) -> tuple[float, float]:

    t = len(datasets)
    test_error = 0
    train_error = 0
    for train, _, test in datasets:
        X_train, y_train = split_data(train)
        X_test, y_test = split_data(test)

        model.reset()
        model.fit(X_train, y_train)

        test_error += model.zero_one_error(X_test, y_test)
        train_error += model.zero_one_error(X_train, y_train)

    test_error /= t
    train_error /= t
    return test_error, train_error

def average_error_val(
        model: BaseModel,
        datasets: list[np.ndarray]
    ) -> tuple[float, float]:

    t = len(datasets)
    test_error = 0
    train_error = 0
    for train, test, _ in datasets:
        X_train, y_train = split_data(train)
        X_test, y_test = split_data(test)

        model.reset()
        model.fit(X_train, y_train)

        test_error += model.zero_one_error(X_test, y_test)
        train_error += model.zero_one_error(X_train, y_train)

    test_error /= t
    train_error /= t
    return test_error, train_error

def average_acc_prec_sens(
        model: BaseModel,
        datasets: list[np.ndarray]
    ) -> tuple[float, float, float]:

    t = len(datasets)
    test_acc = 0
    test_prec = 0
    test_sens = 0

    for train, _, test in datasets:
        X_train, y_train = split_data(train)
        X_test, y_test = split_data(test)

        model.reset()
        model.fit(X_train, y_train)

        test_acc += model.accuracy(X_test, y_test)
        test_prec += model.precision(X_test, y_test)
        test_sens += model.sensitivity(X_test, y_test)

    test_acc /= t
    test_prec /= t
    test_sens /= t

    return test_acc, test_prec, test_sens

def average_fit_time(
        model: BaseModel,
        datasets: list[np.ndarray]
    ) -> float:

    t = len(datasets)
    fit_time = 0
    for train, _, _ in datasets:
        X_train, y_train = split_data(train)

        model.reset()
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time += time.perf_counter() - start_time

    fit_time /= t
    return fit_time

def learning_curve(
        model: BaseModel,
        datasets: list[np.ndarray],
        training_set_fractions: list[float]
    ) -> list[float]:

    test_errors = []
    for fraction in training_set_fractions:
        cut_dataset = []
        for train, _, test in datasets:
            cut_train = train[:int(len(train) * fraction)]
            cut_dataset.append((cut_train, _, test))
        test_error, _ = average_error(model, cut_dataset)

        test_errors.append(test_error)
    return test_errors

def learning_curve_acc_prec_sens(
        model: BaseModel,
        datasets: list[np.ndarray],
        training_set_fractions: list[float]
    ) -> tuple[list[float], list[float], list[float]]:

    test_accs = []
    test_precs = []
    test_senss = []

    for fraction in training_set_fractions:
        cut_dataset = []
        for train, _, test in datasets:
            cut_train = train[:int(len(train) * fraction)]
            cut_dataset.append((cut_train, _, test))
        test_acc, test_prec, test_sens = average_acc_prec_sens(model, cut_dataset)

        test_accs.append(test_acc)
        test_precs.append(test_prec)
        test_senss.append(test_sens)

    return test_accs, test_precs, test_senss

def draw_learning_curve(
        models: list[BaseModel],
        datasets: list[np.ndarray],
        training_set_fractions: list[float],
        colors: list[str]) -> None:

    fig, ax = plt.subplots()
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    for model, color in zip(models, colors):
        sns.lineplot(
            x = training_set_fractions, y = learning_curve(model, datasets, training_set_fractions),
            ax = ax, label=f'{str(model)} test error', color=color)
        ax.set(xlabel='Training set fraction', ylabel='Error')
    ax.set_title(f'{', '.join(str(model) for model in models)} learning curve{"s" if len(models) > 1 else ""}')
    plt.legend()