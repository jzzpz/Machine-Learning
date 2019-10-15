# from sklearn.gaussian_process.kernels import RBF
import numpy as np
# import random
# from math import exp
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
# from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
# from sklearn.svm import LinearSVC
# from sklearn.datasets import make_classification
# from sklearn.svm import SVC
# from sklearn.kernel_approximation import (RBFSampler)
# from sklearn.naive_bayes import GaussianNB

from sklearn import svm


def rbf():
    from sklearn.datasets import load_boston

    bostondataset = load_boston()
    boston = pd.DataFrame(bostondataset.data,
                          columns=bostondataset.feature_names)
    X = boston[['RM', 'AGE', 'DIS', 'RAD', 'TAX']]  # take 5 features
    y = bostondataset.target  # target values

    features = ['RM', 'AGE', 'DIS', 'RAD', 'TAX']
    # print(X['RM'])

    X_train, validate_x, y_train, validate_y = train_test_split(X, y,
                                                                test_size=0.2)

    X_train = np.array(X_train)
    validate_y = np.array(validate_y)
    validate_x = np.array(validate_x)
    y_train = np.array(y_train)

    # ________________________________________________________________
    model = KernelRidge(kernel='rbf', alpha=0.1)  # create a model
    model.fit(X_train, y_train)  # train the model

    y_pred = model.predict(validate_x)

    loss = mean_squared_error(validate_y, y_pred)
    print("Accuracy with regularization (ridge regression) is:", loss)

    # ____________________________________________________________________

    model_2 = KernelRidge(kernel='rbf', alpha=0)
    model_2.fit(X_train, y_train)  # train the model
    y_pred_2 = model_2.predict(validate_x)
    loss_2 = (mean_squared_error(validate_y, y_pred_2))
    print("Accuracy WITHOUT regularization (ridge regression) is:", loss_2)

    return loss, loss_2


l1, l2 = rbf()
