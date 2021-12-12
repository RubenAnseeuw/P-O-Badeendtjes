from mpl_toolkits.mplot3d import axes3d
import random
import numpy as np
import matplotlib.pyplot as plt
from simulaties1TOT8 import xsample,ysample, OLSVoorschriftBeter, bepaalK_opt, Knn_uiteindelijk, vervorm
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def f(x): return 10**x

def LR(xx, yy):
    a, b = OLSVoorschriftBeter(xx, yy)
    xLR = np.linspace(0, 1, 100)
    yLR = [b * xje + a for xje in xLR]
    plt.plot(xLR, yLR, label = "Lineaire Regressie")

def KNN(xx, yy):
    k = bepaalK_opt(20, xx, yy, f)
    xKNN = np.linspace(0, 1, 100)
    yKNN = [Knn_uiteindelijk(xje, xx, yy, k) for xje in xKNN]
    plt.plot(xKNN, yKNN, label = "KNN, Kopt: %s" % k)

def pop():
    xPop = np.linspace(0, 1, 100)
    yPop = [f(xje) for xje in xPop]
    plt.plot(xPop, yPop, label="Populatiefunctie")

def NN():
    xx = xsample(2000, 0, 1)
    random.shuffle(xx)
    y = [f(xsje) for xsje in xx]
    X2 = [[1, xsje] for xsje in xx]

    regr = MLPRegressor(activation = "tanh", hidden_layer_sizes = 20, random_state = 3, max_iter = 1000).fit(X2, y)

    xxx = np.linspace(0, 1, 100)
    XXX2 = [[1, xsje] for xsje in xxx]
    yyy = regr.predict(XXX2)

    plt.plot(xxx, yyy, label = "Neurale Netwerken")


def main():
    plt.figure()
    xx = xsample(20)
    yy = ysample(xx, f)
    LR(xx, yy)
    KNN(xx, yy)
    pop()
    NN()
    plt.legend()
    plt.show()

main()
