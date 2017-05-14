import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, X):
        # ベクトルの内積を求める
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # 条件に合致した行列のインデックスを求めてる
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        # 指定さた長さの0の行列を作成する
        self.w_ = np.zeros(1 + X.sharpe[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
