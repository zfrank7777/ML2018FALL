# -*- coding: UTF-8 -*-
import os
import sys
import numpy as np


def read_test_file(filename):
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
                            if i == 0:
                                x[i] = 30
                            elif i == 1:
                                x[i] = x[i-1]
                            else:
                                x[i] = 0.5*(x[i-1]+x[i-2])
                    X.append(x)

    return X


def test(b, theta, X, filename):
    f = open(filename, "w")
    f.write("id,value\n")
    for i in range(len(X)):
        y = sum([t*a for t, a in zip(theta, X[i])]) + b
        string = "id_"+str(i)+','+str(y)+'\n'
        f.write(string)


if __name__ == "__main__":
    Xtest = read_test_file(sys.argv[1])
    arguments = np.load("model.npy")
    b = arguments[-1]
    theta = arguments[:-1]
    test(b, theta, Xtest, sys.argv[2])
