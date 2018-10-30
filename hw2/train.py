""" Generative Model """
# -*- coding: UTF-8 -*-
import os
import numpy as np
import pandas as pd


def sigmoid(x):
    val = np.float64(1 / np.float64(1 + np.float64(np.exp(-x))))
    return val


def read_X(filename):
    if os.path.isfile(filename):
        X = pd.read_csv(filename)
        sex = pd.get_dummies(X["SEX"], prefix="SEX")
        marriage = pd.get_dummies(X["MARRIAGE"], prefix="MARRIAGE")
        tmp = X.drop(['SEX', 'MARRIAGE'], axis=1)
        newX = pd.concat([tmp, sex, marriage], axis=1)
    normX = normalize(np.array(newX))
    return normX, list(newX)


def read_y():

    filename = "train_y.csv"
    if os.path.isfile(filename):
        y = pd.read_csv(filename)

    return (np.array(y)-0.5)*2


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X-mean)/std
    return X


def test(X, mu0, mu1, sigma, N0, N1, filename):
    f = open(filename, "w")
    f.write("id,value\n")
    for i in range(len(X)):
        WT = np.dot((mu0-mu1).reshape(1, -1), np.linalg.inv(sigma))
        # print (WT.shape)
        WTx = np.dot(WT, X[i])
        # print (WTx.shape)
        b0 = np.dot(-0.5 * mu0.reshape(1, -1), np.linalg.inv(sigma))
        b0 = np.dot(b0, mu0.reshape(-1, 1))
        # print (b0.shape)
        b1 = np.dot(0.5 * mu1.reshape(1, -1), np.linalg.inv(sigma))
        b1 = np.dot(b1, mu1.reshape(-1, 1))
        # print (b1.shape)
        b = b0+b1+np.log(N0/N1)
        z = WTx+b
        y = 0 if sigmoid(float(z)) > 0.5 else 1
        string = "id_"+str(i)+','+str(y)+'\n'
        f.write(string)


def train(X, y):
    class0 = []
    class1 = []
    for i in range(len(y)):
        if y[i] == 1:
            class1.append(X[i])
        else:
            class0.append(X[i])
    class0 = np.array(class0)
    class1 = np.array(class1)
    N0 = float(len(class0))
    N1 = float(len(class1))
    mu0 = np.mean(class0, axis=0)
    mu1 = np.mean(class1, axis=0)
    sigma0 = np.zeros((len(class0[0]), len(class0[0])))
    for i in range(len(class0)):
        sigma0 += np.matmul((class0[i]-mu0).reshape(-1, 1),
                            (class0[i]-mu0).reshape(-1, 1).T)
    sigma0 /= N0
    sigma1 = np.zeros((len(class1[0]), len(class1[0])))
    for i in range(len(class1)):
        sigma1 += np.matmul((class1[i]-mu1).reshape(-1, 1),
                            (class1[i]-mu1).reshape(-1, 1).T)
    sigma1 /= N0
    PC0 = N0/float(len(X))
    PC1 = N1/float(len(X))
    sigma = PC0*sigma0 + PC1*sigma1
    return mu0, mu1, sigma, N0, N1


if __name__ == "__main__":
    Xtrain, names = read_X("train_x.csv")
    Xtest, names = read_X("test_x.csv")
    ytrain = read_y()
    mu0, mu1, sigma, N0, N1 = train(Xtrain, ytrain)
    test(Xtest, mu0, mu1, sigma, N0, N1, "result/1gen.txt")
