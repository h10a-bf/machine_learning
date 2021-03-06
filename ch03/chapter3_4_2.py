import lib.dataset as dataset
import lib.plot as plot
import matplotlib.pyplot as plt

X_combined_std, X_train_std, X_train, X_test_std, X_test, y_combined, y_train, y_test \
    = dataset.load_iris()

#################### SVM ####################
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=999999999999.0, random_state=0)
svm.fit(X_train_std, y_train)

plot.plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./chapter_3_4_2_C_999999999999.png', dpi=300)
plt.show()

