from LISTA  import solve
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def wine_classification(data, label):
    '''
    classify wine data and its label directly
    '''
    data_train_pre, data_test_pre, label_train, label_test = \
        train_test_split(data, label, test_size=0.4)
    print("Learning classifier for wine data...")
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class = 'multinomial').fit(data_train_pre, label_train)
    return clf.score(data_test_pre, label_test)

print("Starting...")
sparse_code, (data, label) = solve()
data_train_pre, data_test_pre, label_train, label_test = \
    train_test_split(sparse_code, label,
                     test_size=0.4)
# same random state as function wine_classification to ensure same split
# source: https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets
print("Learning classifier for sparse code...")
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(data_train_pre, label_train)

print("LISTA+Logistic Regression ERROR: " + str(clf.score(data_test_pre, label_test)))
print("Logistic Regression ERROR: " + str(wine_classification(data, label)))
