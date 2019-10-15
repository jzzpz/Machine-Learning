from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from math import exp


def Loss(X, y, w):
    # return (np.dot(-y, (np.log(Sig(np.dot(-X, w)))))) - (np.dot(1-y,np.log(1-(Sig(np.dot(-X,w))))))

    all_diff = []
    # for each data
    for i in range(len(y)):
        if y[i] == 1:
            if Sig(np.dot(X[i], w)) == 0.:
                all_diff.append(-np.log(1 / (10 ** 100)))
            else:
                all_diff.append(-np.log(Sig(np.dot(X[i], w))))
        elif y[i] == 0:
            if 1 - Sig(np.dot(X[i], w)) == 0.:
                all_diff.append(-np.log(1 / (10 ** 100)))
            else:
                all_diff.append(-np.log(1 - Sig(np.dot(X[i], w))))

    return sum(all_diff)


def Sig(x):
    return (1) / (1 + np.exp(-x))


def GradientDescent(X, y, w):
    return np.dot(X.transpose(), Sig(np.dot(X, w)) - y)


def fitLogReg(X, y, a, T, X_test, y_test):
    weight = []
    X = np.insert(X, 0, 1, axis=1)  # add a column of 1s
    X = X.astype(np.float128)  # to avoid floating erros
    y = np.array(y)
    for i in range(len(X[0])):
        weight.append(random.randrange(-5, 5))

    weight = np.array(weight)
    new_weight = weight - (a * GradientDescent(X, y, weight))
    # print("new_weight",new_weight)
    loss = float("inf")

    iter = 0
    # while not convergence
    while loss > T:
        loss = Loss(X, y, new_weight)
        new_weight = new_weight - (a * GradientDescent(X, y, new_weight))
        # print(new_weight)
        # print("loss", loss)

        iter += 1

    # _______________________________________________________________________
    # test the model with testing data and weight from traning data
    X_test = np.insert(X_test, 0, 1, axis=1)
    first_features = []
    second_features = []
    for i in X_test:
        first_features.append(i[1])
        second_features.append(i[2])

    y_dec_bound = []

    for i in range(len(X_test)):
        y_dec_bound.append(
            abs(((new_weight[0]) + (X_test[i][1] * new_weight[1])) / (
                new_weight[2])))
    plt.scatter(first_features, second_features, c=y_test, cmap=plt.cm.Set3,
                edgecolor='k')
    plt.plot(first_features, y_dec_bound)
    plt.xlabel('X^(1)')
    plt.ylabel('X^(2)')
    plt.title('With a = {} , T = {}'.format(a, T))
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    y_hat = [1 if np.dot(X_test[i], new_weight) >= 0.5 else 0 for i in
             range(len(y_test))]
    for i in range(len(y_test)):
        if y_test[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_test[i] != y_hat[i]:
            FP += 1
        if y_test[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_test[i] != y_hat[i]:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    R = TP / (TP + FN)

    # avoid divide by 0 error
    if TP + FP == 0:
        P = TP / 0.00001
    else:
        P = TP / (TP + FP)

    if (P + R) == 0:
        F1 = 2 * ((P * R) / (0.00001))
    else:
        F1 = 2 * ((P * R) / (P + R))

    print(
        "Number of iterations:{}, a:{}, T:{}  , accuracy:{}, R:{}, P:{}, F1:{}".format(
            iter, a, T, accuracy, R, P, F1))
    plt.show()
    return


# take the first two classes of the dataset i.e., first 100 instances.
iris = datasets.load_iris()
X = iris.data[:100, :]  # 4 features
y = iris.target[:100]  # the labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
a_list = [0.99, 0.5, 0.001]
T_list = [0.9, 0.0001]

# uncommon this to run for a list of a and T values
# for a in a_list:
#     for T in T_list:
#         fitLogReg(X_train, y_train, a, T, X_test, y_test)

fitLogReg(X_train, y_train, 0.8, 0.1, X_test, y_test
          )
