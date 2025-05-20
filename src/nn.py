import numpy as np
from src.classification import BaseModel
from src.activation import sigmoid, ReLU, ReLU_backward


# I wrote this code for Artificial Intelligence course at JU
class HardcodedNeuralNetwork(BaseModel):
    """
    Very basic NN for classification
    Minibatch gradient descent
    Only ReLU
    He initialization
    Standardization
    Sigmoid at the end
    Despite hardcoding you need to specify good input dim and output dim ;)
    !!!Important: y should be in the shape (1,m), X in (m,d)!!!
    We need to change y = -1 to y = 0 (hardcoded)
    """

    def __init__(
        self,
        nn_architecture: list[dict[str, int]],
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int = 64,
    ):
        self.nn_architecture = nn_architecture
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.X = None
        self.y = None
        self.param_values = None
        self.loss_history = None
        self.means = None
        self.stds = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.y = np.where(self.y == -1, 0, self.y)

        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)

        self.X = (self.X - self.means) / self.stds

        self.__init_layers()

        self.loss_history = []


        for i in range(self.epochs):
            train_zip = list(zip(self.X, self.y))
            epoch_loss = 0
            for idx in range((X.shape[1] + self.batch_size - 1) // self.batch_size):
                batch = train_zip[idx * self.batch_size:(idx + 1) * self.batch_size]

                X_sample = np.transpose(np.array(list(map(lambda t: t[0], batch))))
                Y_sample = np.transpose(np.array(list(map(lambda t: t[1], batch))))
                Y_out, memory = self.__full_forward_propagation(X_sample)

                L = self.binary_cross_entropy(X_sample, Y_sample)

                grads_values = self.__full_backward_propagation(Y_out, Y_sample, memory)

                self.__update(grads_values)

                epoch_loss += L
            self.loss_history.append(epoch_loss)

    def get_loss_history(self) -> list[float]:
        return self.loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = (X - self.means) / self.stds
        Y_out, _ = self.__full_forward_propagation(X.T)
        return np.where(Y_out > 0.5, 1, -1).reshape(-1, 1)

    def reset(self) -> None:
        self.X = None
        self.y = None
        self.param_values = None
        self.loss_history = None
        self.means = None
        self.stds = None

    def __str__(self) -> str:
        return "NN"

    def binary_cross_entropy(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        m = X.shape[1]
        y_pred, _ = self.__full_forward_propagation(X)
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8) # to avoid log(0)
        return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m

    def __init_layers(self) -> None:
        self.param_values = {}
        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            self.param_values["W" + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size
            ) * np.sqrt(4.0 / layer_output_size + layer_input_size)
            self.param_values["b" + str(layer_idx)] = np.zeros((layer_output_size, 1))

    def __single_layer_forward_propagation(
        self, A_prev: np.ndarray, W_curr: np.ndarray, b_curr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        Z_curr = np.matmul(W_curr, A_prev) + b_curr
        return ReLU(Z_curr), Z_curr

    def __full_forward_propagation(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        memory = {}
        memory['A' + str(0)] = X

        for idx, _ in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            A_prev = memory['A' + str(idx)]
            W_curr = self.param_values['W' + str(layer_idx)]
            b_curr = self.param_values['b' + str(layer_idx)]
            A_curr, Z_curr = self.__single_layer_forward_propagation(A_prev, W_curr, b_curr)

            memory['A' + str(layer_idx)] = A_curr
            memory['Z' + str(layer_idx)] = Z_curr

        A_sigmoid = sigmoid(memory['Z' + str(len(self.nn_architecture))])
        memory['A' + str(len(self.nn_architecture))] = A_sigmoid

        return A_sigmoid, memory

    def __full_backward_propagation(
        self,
        Y_out: np.ndarray,
        Y: np.ndarray,
        memory: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        grads_values = {}

        m = Y.shape[1]
        dZ_curr = Y_out - Y
        A_prev = memory['A' + str(len(self.nn_architecture) - 1)]
        dW_curr = np.matmul(dZ_curr, np.transpose(A_prev)) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m

        memory['dZ' + str(len(self.nn_architecture))]       = dZ_curr
        grads_values['dW' + str(len(self.nn_architecture))] = dW_curr
        grads_values['db' + str(len(self.nn_architecture))] = db_curr

        for idx, _ in reversed(list(enumerate(self.nn_architecture))):
            if (idx == 0): 
                continue

            dZ_next = memory['dZ' + str(idx + 1)]
            A_prev  = memory['A' + str(idx - 1)]
            W_next  = self.param_values['W' + str(idx + 1)]
            Z_curr  = memory['Z' + str(idx)]

            memory['dZ' + str(idx)] = np.multiply(ReLU_backward(Z_curr), np.matmul(np.transpose(W_next), dZ_next))
            dZ_curr = memory['dZ' + str(idx)]

            grads_values['dW' + str(idx)] = np.matmul(dZ_curr, np.transpose(A_prev)) / m
            grads_values['db' + str(idx)] = np.sum(dZ_curr, axis=1, keepdims=True) / m

        return grads_values

    def __update(self, grads_values: dict[str, np.ndarray]) -> None:
        for idx, _ in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            self.param_values['W' + str(layer_idx)] -= self.learning_rate * grads_values['dW' + str(layer_idx)]
            self.param_values['b' + str(layer_idx)] -= self.learning_rate * grads_values['db' + str(layer_idx)]