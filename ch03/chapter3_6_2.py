import lib.dataset as dataset
import lib.plot as plot
import matplotlib.pyplot as plt
import numpy as np

X_combined_std, X_train_std, X_train, X_test_std, X_test, y_combined, y_train, y_test \
    = dataset.load_iris()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot.plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
