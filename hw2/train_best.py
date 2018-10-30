""" Logistic Regression Model """
# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
import pandas as pd


def sigmoid(x):
    val = np.float64(1 / np.float64(1 + np.float64(np.exp(-x))))
    return val


def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


def read_X(f1, f3):
    Xtrain = pd.read_csv(f1)
    Xtest = pd.read_csv(f3)
    X = Xtrain.append(Xtest, ignore_index=True)
    sex = pd.get_dummies(X["SEX"], prefix="SEX")
    marriage = pd.get_dummies(X["MARRIAGE"], prefix="MARRIAGE")
    pay0 = pd.get_dummies(X["PAY_0"], prefix="PAY_0")
    pay2 = pd.get_dummies(X["PAY_2"], prefix="PAY_2")
    pay3 = pd.get_dummies(X["PAY_3"], prefix="PAY_3")
    pay4 = pd.get_dummies(X["PAY_4"], prefix="PAY_4")
    pay5 = pd.get_dummies(X["PAY_5"], prefix="PAY_5")
    pay6 = pd.get_dummies(X["PAY_6"], prefix="PAY_6")
    tmp = X.drop(['SEX', 'MARRIAGE', "PAY_0", "PAY_2",
                  "PAY_3", "PAY_4", "PAY_5", "PAY_6"], axis=1)
    newX = pd.concat([tmp, sex, marriage, pay0, pay2, pay3, pay4, pay5, pay6],
                     axis=1)
    normX = normalize(newX)
    train = normX[:20000]
    test = normX[20000:]
    return train, test, list(newX)


def read_y(f2):

    filename = f2
    if os.path.isfile(filename):
        y = pd.read_csv(filename)

    return (np.array(y)-0.5)*2


def compute_loss(W, X, y):
    total_error = 0
    N = float(len(X))
    for i in range(len(X)):
        total_error += (-1/N) * np.log(sigmoid(y[i]*np.dot(W.T, X[i])))
    return total_error


def stepGradient(W, X, y, learningRate):
    W_grad = np.zeros(W.shape)
    N = float(len(X))
    for i in range(len(X)):
        W_grad += (1/N) * sigmoid(-y[i]*np.dot(W.T, X[i])) * (-1) * y[i] * X[i]

    new_W = W - learningRate * W_grad
    return new_W


def logistic_regression(X, y, learningRate, steps):
    """ initialize """
    theta = []
    for i in range(len(X[0])):
        if X[0][i] != 0:
            theta.append(np.random.random_sample()/float(X[0][i]))
        else:
            theta.append(np.random.random_sample())
    theta = np.array(theta)
    theta = np.zeros(len(X[0]))

    print ("Dim of x: ", len(X[0]))
    print ("Num of x: ", len(X))
    print ("Initial loss: ", compute_loss(theta, X, y))
    # print ("All 1 loss:", compute_loss(theta, X, [1 for i in range(len(y))]))

    print ("---- start ----")
    """ gradient descent """
    for i in range(steps):
        theta = stepGradient(theta, X, y, learningRate)
        theta = np.array(theta)
        if i % 10 == 0:
            print("Step ", i+1, ", Loss: ", compute_loss(theta, X, y))

    print ("----- end -----")
    return theta


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X-mean)/std
    X = add_intercept(X)
    return X


def test(theta, X, filename):
    f = open(filename, "w")
    f.write("id,value\n")
    for i in range(len(X)):
        y = 0 if sigmoid(np.dot(theta.T, X[i])) < 0.5 else 1
        string = "id_"+str(i)+','+str(y)+'\n'
        f.write(string)


def train(X, y, learningRate, steps):
    theta = logistic_regression(X, y, learningRate, steps)
    return theta


if __name__ == "__main__":
    Xtrain, Xtest, names = read_X(sys.argv[1], sys.argv[3])
    ytrain = read_y(sys.argv[2])
    theta = train(Xtrain, ytrain, 1, 200)
    np.save("model.npy", theta)
    # test(theta, Xtest, "result/result4-5.txt")
