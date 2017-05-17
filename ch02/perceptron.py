# -*- coding: utf-8 -*-

import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        # zeros は全部 0 の配列を作るだけ。要素数は、X の要素数 + 1です。+1 してるのは、x によらないオフセットのようなものの表現（多分）
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            # zip は、配列をまとめて扱えるっていうやつ
            for xi, target in zip(X, y):
                # ココが実際の計算ポイント。
                # xi（2つの実測値）から重み付け値を出して分類してみて、結果が -1-1 で -2 になったり、-1+1 で　0 になったりして、それに応じて 学習率をかけて 重み付けの変動幅を出して、それを使って重み付けを書き換えている。
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # 判定は、単純に + になるかどうかだけを見て判定している。間違っていた場合は 2η か -2η の補正になるので、重み付けの個々の値は マイナスにもなる。
        return np.where(self.net_input(X) >= 0.0, 1, -1)