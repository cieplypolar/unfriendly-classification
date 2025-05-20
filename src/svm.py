import numpy as np
from src.classification import BaseModel
from src.kernels import BaseKernel

class SVM_SMO(BaseModel):
    def __init__(
            self,
            kernel: BaseKernel,
            C: float,
            epochs: int = 1000,
            eps: float = 1e-3,
            silent: bool = False):
        self.kernel = kernel
        self.C = C
        self.epochs = epochs
        self.eps = eps
        self.silent = silent
        self.b = 0
        self.alphas = None
        self.X = None
        self.y = None
        self.m = None
        self.non_kkt = None
        # in my understanding we cannot use it outside i1 heuristic
        self.error_cache = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        m, _ = X.shape
        self.m = m
        self.alphas = np.zeros(m)
        self.b = 0

        self.non_kkt = np.arange(m)
        self.error_cache = (-y).flatten()

        epoch = 0
        if not self.silent:
            print("Training...")
        while epoch < self.epochs:
            i2 = self.__i2_heuristics()

            if i2 == -1:
                if not self.silent:
                    print("All examples are KKT")
                break

            i1 = self.__i1_heuristics(i2)

            self.__take_step(i1, i2)

            epoch += 1

        if not self.silent:
            print("Number of support vectors: ", np.sum(self.alphas > 0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.margin(X))

    def margin(self, X: np.ndarray) -> np.ndarray:
        support_alphas_idx = np.argwhere(self.alphas != 0).reshape((1, -1))[0]
        support_y = self.y[support_alphas_idx,:]
        support_X = self.X[support_alphas_idx,:]

        w = (self.alphas[support_alphas_idx] * support_y.flatten())
        K = self.kernel(support_X, X)
        u = w @ K - self.b

        return u.reshape(-1, 1)

    def reset(self) -> None:
        self.b = 0
        self.alphas = None
        self.X = None
        self.y = None
        self.m = None
        self.non_kkt = None
        self.error_cache = None

    def __str__(self) -> str:
        return f"SVM SMO {str(self.kernel)}"

    def __check_kkt(self, i: int) -> np.ndarray:
        alph = self.alphas[i]
        y = self.y[i,0]
        u = self.margin(self.X[i])
        E = u - y # cannot use error cache, old value possible
        r = E * y

        return ~(((r < -self.eps) & (alph < self.C)) | ((r > self.eps) & (alph > 0)))

    def __i2_heuristics(self) -> tuple[int, np.ndarray]:
        i2 = -1

        for i in self.non_kkt:
            self.non_kkt = np.delete(self.non_kkt, np.argwhere(self.non_kkt == i))

            # could be optimized
            if not self.__check_kkt(i):
                i2 = i
                return i2
        # it can be slow, will be used rarely
        new_non_kkt = np.array([i for i in np.arange(self.m) if not self.__check_kkt(i)])
        if len(new_non_kkt) > 0:
            np.random.shuffle(new_non_kkt)
            i2 = new_non_kkt[0]
            self.non_kkt = new_non_kkt[1:]

        return i2

    def __i1_heuristics(self, i2: int) -> int:
        E2 = self.error_cache[i2]

        non_bound = np.argwhere((self.alphas > 0) & (self.alphas < self.C)).reshape((1, -1))[0]
        if len(non_bound) > 0:
            if E2 >= 0:
                i1 = non_bound[np.argmin(self.error_cache[non_bound])]
            else:
                i1 = non_bound[np.argmax(self.error_cache[non_bound])]
        else:
            i1 = np.argmax(np.abs(self.error_cache - E2))

        return i1


    def __take_step(self, i1: int, i2: int) -> None:
        if i1 == i2:
            return # cannot optimize with constraints

        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = self.y[i1,0]
        y2 = self.y[i2,0]

        u1 = self.margin(self.X[i1])
        u2 = self.margin(self.X[i2])

        s = y1 * y2
        L = None
        H = None
        if s == 1:
            L = max(0, alph2 + alph1 - self.C)
            H = min(self.C, alph2 + alph1)
        else:
            L = max(0, alph2 - alph1)
            H = min(self.C, self.C + alph2 - alph1)

        if L == H:
            return

        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])
        eta = k11 + k22 - 2 * k12

        # cannot use error cache, old value possible
        E1 = u1 - y1
        E2 = u2 - y2

        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta
            a2 = np.clip(a2, L, H)
        else:
            # we can choose different sample :)
            return
            f1 = y1 * (E1 + self.b) - alph1 * k11 - s * alph2 * k12
            f2 = y2 * (E2 + self.b) - s * alph1 * k12 - alph2 * k22
            L1 = alph1 + s * (alph2 - L)
            H1 = alph1 + s * (alph2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * (L1 * L1) * k11 + 0.5 * (L * L) * k22 + s * L1 * L * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * (H1 * H1) * k11 + 0.5 * (H * H) * k22 + s * H1 * H * k12
            if Lobj < Hobj - self.eps:
                a2 = L
            elif Lobj > Hobj + self.eps:
                a2 = H
            else:
                a2 = alph2

        # low chances we will have convergence
        # if abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps): # relative convergence stopping criterion
        #     return

        a1 = alph1 + s * (alph2 - a2)
        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.b

        self.b = (b1 + b2) / 2
        self.alphas[i1] = a1
        self.alphas[i2] = a2
        self.error_cache[i1] = self.margin(self.X[i1]) - y1
        self.error_cache[i2] = self.margin(self.X[i2]) - y2