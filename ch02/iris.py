# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def import_data():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/' 'machine-learning-databases/iris/iris.data', header=None)
    # サンプルを見せてるだけです。
    df.tail()

    # 1-100行目の目的変数の抽出 :
    #   iloc は pandas の機能。多分 index location とかそんなん。先頭100行で、配列の４項目目（0 origin なので）を持ってきて、y に掘り込む
    y = df.iloc[0:100, 4].values

    # Iris-setosaを-1, Iris-virginicaを1に変換 :
    #  ２値分類なので、片側を正解にすればそれでいい、ということです。
    y = np.where(y == 'Iris-setosa', -1, 1)

    # 1-100行目の1、3列目の抽出 : y と同じように、1と3 だけを抜いている
    X = df.iloc[0:100, [0, 2]].values

    return X, y