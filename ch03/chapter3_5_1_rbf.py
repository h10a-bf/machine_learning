from sklearn.svm import SVC
import lib.xor as xor
import lib.plot as plot
import matplotlib.pyplot as plt

X_xor, y_xor = xor.xor_data()

svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot.plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()