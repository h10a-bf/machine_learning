from sklearn import datasets
import numpy as np
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_iris(test_size=0.3, random_state=0):
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    sc = StandardScaler()
    sc.fit(X_train)
    # 平均0, 分散1に標準化
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)


    X_combined_std = np.vstack((X_train_std, X_test_std)) # 縦に連結
    y_combined = np.hstack((y_train, y_test)) # 横に連結

    return [X_combined_std, X_train_std, X_train, X_test_std, X_test, y_combined, y_train, y_test]