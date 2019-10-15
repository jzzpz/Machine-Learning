import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def FitPolynomialRegression(K, x, y):
    # x is a column vector: [[1]
    #                        [2]]

    #________________________________________________________________________
    #________________________________________________________________________
    if not (isinstance(x[0], np.ndarray) or isinstance(x[0], list)):
        x = np.array([np.array([i]) for i in x])

    X = []
    for i in range(len(x)):
        temp_vec = []
        for j in range(K+1):
            # temp_vec.append(x[i] ** j)
            temp_vec.append(x[i][0]**j)


        X.append(temp_vec)

    x_np = np.array(X)
    y_np = np.array(y)
    # x_t = x_np.transpose()
    # x_inv = np.linalg.inv(np.matmul(x_t, x_np))
    #
    # temp1 = np.matmul(x_inv, x_t)

    # print(np.matmul(temp1,y_np))
    #
    # print(np.matmul(np.linalg.pinv(X), y_np))
    return (np.dot(np.linalg.pinv(X),y_np))
    # return np.matmul(temp1,y_np)


def EvalPolynomial(x, w):
    # if x is not column vec
    if not (isinstance(x[0],np.ndarray) or isinstance(x[0],list)) :
        x= np.array([np.array([i]) for i in x])

    y_vec = []
    for i in range(len(x)):

        res=[]
        for j in range(len(w)):
            res.append(   (w[j])*(x[i][0]**j)  )

        y_vec.append(sum(res))

    # print(y_vec)
    return y_vec



def GetBestPolynomial(xTrain, yTrain, xTest, yTest, h):

    # if the input is not a column vector
    if not (isinstance(xTrain[0],np.ndarray) or isinstance(xTrain[0],list)):
        xTrain = np.array([np.array([i]) for i in xTrain])

    if not (isinstance(xTest[0], np.ndarray) or isinstance(xTest[0],list)):
        xTest = np.array([np.array([i]) for i in xTest])

    test_model_dict = dict()  # test dict
    train_model_dict = dict() # train dict

    for i in range(1,h+1):
        weight = FitPolynomialRegression(i,xTrain,yTrain)
        pred_y_train = EvalPolynomial(xTrain,weight)
        y_train_error = []
        for j in range(len(yTrain)):
            y_train_error.append( (yTrain[j] - pred_y_train[j] )**2)

        # print(pred_y_train_list)

        pred_y_test =  EvalPolynomial(xTest,weight)
        y_test_error = []
        for j in range(len(yTest)):
            y_test_error.append( (yTest[j] - pred_y_test[j] )**2)
        # print(pred_y_test_list)

        test_model_dict[i] = sum(y_test_error)/len(y_test_error)
        train_model_dict[i] = sum(y_train_error)/len(y_train_error)


    # red is the trainning
    # blue is testing
    plt.plot(list(test_model_dict.keys()), list(test_model_dict.values()), label = 'Testing Error')
    plt.plot(list(train_model_dict.keys()), list(train_model_dict.values()) ,label = "Training Error")
    plt.ylabel('residual squared errors')
    plt.xlabel('polynomial degree')
    plt.scatter(list(test_model_dict.keys()), list(test_model_dict.values()))
    plt.scatter(list(train_model_dict.keys()), list(train_model_dict.values()))
    plt.legend(loc='upper left')

    least_test_error = min(test_model_dict, key=test_model_dict.get)

    # return the lowest error poly degree
    print("The lowest reidual squared error for the Polynomial degree of ",least_test_error)
    print("Test Errors: ", list(test_model_dict.values()))
    print("Train Errors: ", list(train_model_dict.values()))

    plt.show()

    # return least_test_error

if __name__== "__main__":
    given_data = pd.read_csv("hw1_polyreg.csv")
    train_data, test_data = train_test_split(given_data, test_size=0.25)
    # print(train_data)
    # print(test_data)
    train_x, train_y = train_data.iloc[:, 1:2].values, train_data.iloc[:,
                                                       2].values
    test_x, test_y = test_data.iloc[:, 1:2].values, test_data.iloc[:, 2].values




    k = GetBestPolynomial(train_x,train_y,test_x,test_y,76)
    # ______________________________________________________________________

