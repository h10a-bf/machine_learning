# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import iris
from perceptron import Perceptron

X, y = iris.import_data()

# パーセプロトロンのオブジェクトの生成（インスタンス化）
# η（イータ : eta : 学習率）を 0.1、繰り返しを10回でパーセプトロンを生成する。
ppn = Perceptron(eta=0.1, n_iter=10)

# トレーニングデータへのモデルの適合
# ここでは、X は ２つの実測値array 100個、y は正解データ(-1 or 1)１００個です。
ppn.fit(X, y)

# エポックと誤分類誤差の関係の折れ線グラフをプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

# 軸のラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

# 図の表示
plt.show()