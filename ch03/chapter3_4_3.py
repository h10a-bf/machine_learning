import lib.dataset as dataset
import lib.plot as plot
import matplotlib.pyplot as plt

X_combined_std, X_train_std, X_train, X_test_std, X_test, y_combined, y_train, y_test \
    = dataset.load_iris()

#################### SVM ####################
from sklearn.linear_model import SGDClassifier

ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')

svm.fit(X_train_std, y_train)
plot.plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
