# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import iris

X, y = iris.import_data()

###########
# 以下、一旦正解を見せるためのデモ。
###########
#  元のデータは、前半50個が Iris-setosa(-1)、後半50個が Iris-versicolor (1) なので、こういう分け方をしているだけです。
# 品種setosaのプロット (赤の○) :
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# 品種versicolorのプロット (青の×)

plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

# 軸のラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定 (左上に配置)
plt.legend(loc='upper left')
# 図の表示
plt.show()