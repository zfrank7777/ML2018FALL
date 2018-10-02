# -*- coding: UTF-8 -*-
import os
import numpy as np
import math


def read_train_file():
    filename = "train.csv"
    X = []
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            f.readline()
            end = 0
            while end == 0:
                for i in range(18):
                    data = f.readline()
                    if data == b'':
                        end = 1
                        break
                    data = str(data).replace("NR", "-100").split(',')
                    if data[2] != "PM2.5":
                        continue
                    while(data[-1][-1].isdigit() is False):
                        data[-1] = data[-1][:-1]
                    x = data[3:]
                    for each in x:
                        if float(each) <= 100 and float(each) > 2:
                            X.append(each)
    return X


def transform_data(data):
    X = []
    y = []
    for i in range(9, len(data)):
        x = []
        for j in range(i-9, i):
            x.append(float(data[j]))
        X.append(x)
        y.append(float(data[i]))
    return X, y


def read_test_file():
    filename = "test.csv"
    X = []
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            end = 0
            while end == 0:
                for i in range(18):
                    data = f.readline()
                    if data == b'':
                        end = 1
                        break
                    data = str(data).replace("NR", "-100").split(',')
                    if data[1] != "PM2.5":
                        continue
                    while(data[-1][-1].isdigit() is False):
                        data[-1] = data[-1][:-1]
                    x = []
                    for j in range(9):
                        x.append(float(data[j+2]))
                    for i in range(1, len(x)):
                        if x[i] > 80 or x[i] <= 2:
                            if i == 0: x[i] = 30
                            elif i == 1: x[i] = x[i-1]
                            else: x[i] = 0.5*(x[i-1]+x[i-2])
                    X.append(x)

    return X


def compute_loss(b, theta, X, y):
    total_error = 0
    for i in range(len(X)):
        error = (y[i]-sum([t*a for t, a in zip(theta, X[i])]))**2
        total_error += error
    return math.sqrt(total_error/float(len(X)))


# reference:
# https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
def stepGradient(b_current, theta_current, X, y, learningRate):
    b_gradient = 0
    N = float(len(X))
    theta_gradient = []
    for i in range(len(X[0])):
        theta_gradient.append(0)

    for i in range(len(X)):
        b_gradient += -(2/N) * (y[i] - (sum([t*a for t, a in zip(theta_current, X[i])]) + b_current))
        for j in range(len(X[0])):
            theta_gradient[j] += -(2/N) * X[i][j] * (y[i] - (sum([t*a for t, a in zip(theta_current, X[i])]) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_theta = []
    for i in range(len(X[0])):
        new_theta.append(theta_current[i] - (learningRate * theta_gradient[i]))

    return [new_b, new_theta]


def train(X, y, learningRate, steps):
    """ initialize """
    b = np.random.random_sample()
    theta = []
    for i in range(len(X[0])):
        theta.append(np.random.random_sample())
    print ("\nThis program concates the data, and cut off weird data.")
    print ("Dim of x: ", len(X[0]))
    print ("Num of x: ", len(X))
    print ("Initial loss: ", compute_loss(b, theta, X, y))
    print ("---- start ----")
    """ gradient descent """
    for i in range(steps):
        b, theta = stepGradient(b, theta, X, y, learningRate)
        print("step ", i+1, ", loss: ", compute_loss(b, theta, X, y))

    print ("----- end -----")
    return b, theta


def test(b, theta, X, filename):
    f = open(filename, "w")
    f.write("id,value\n")
    for i in range(len(X)):
        y = sum([t*a for t, a in zip(theta, X[i])]) + b
        string = "id_"+str(i)+','+str(y)+'\n'
        f.write(string)


if __name__ == "__main__":
    training_data = read_train_file()
    Xtest = read_test_file()
    Xtrain, ytrain = transform_data(training_data)
    [b, theta] = train(Xtrain, ytrain, 0.0001, 1000)
    test(b, theta, Xtest, "result.csv")
    theta.append(b)
    np.save("model.npy", theta)
