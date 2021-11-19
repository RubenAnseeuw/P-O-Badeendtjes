# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:55:58 2021

@author: ansee
"""
from math import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
sp.init_printing(pretty_print=False)
import random
import statistics
from numpy.random import *
from mpl_toolkits.mplot3d import Axes3D

def f(x): return 10**x

def xsample(n, a=0, b=1):
    return np.linspace(a, b, n)

def ysample(x, f, e=1):
    yarray = []
    e_array = np.random.normal(0,e,len(x))
    for index in range(len(x)):
        yarray.append(f(x[index]) + e_array[index])
    return yarray



x = xsample(5, 0,20)
y = ysample(x, f, e=.3)

def OLSVoorschriftBeter(x,y):
    x_streep = statistics.mean(x)
    y_streep = sum(y)/len(y)
    somteller = somnoemer = 0
    for index in range(len(x)):
        somteller += (x[index] - x_streep)*(y[index]+y_streep)
        somnoemer += (x[index]-x_streep)**2
    b = somteller/somnoemer
    a = y_streep -b *x_streep
    return (a,b)



def Knn_stap1(x0, x):
    lijst = []
    for index in range(len(x)):
        lijst.append((np.linalg.norm(x0-x[index]), index))
    return lijst

def Knn_stap2(x0,x,k):
    lijst = Knn_stap1(x0,x)
    gesorteerd = sorted(lijst)
    slice = [gesorteerd[a][1] for a in range(k)]
    return slice


def Knn_uiteindelijk(x0, x, y , k):
    lijst = Knn_stap2(x0, x, k)
    y_hoedje = 1/k * sum([y[index] for index in lijst])
    return y_hoedje

def simulatie1():
    def f(x): return 10**x
    x = xsample(20,0,1)
    y = ysample(x, f, e=1)
    plt.figure()
    plt.plot(x, 10**x, lw = 3)
    plt.plot(x,y, 'o')
    tuple = OLSVoorschriftBeter(x,y)
    plt.plot(x, tuple[0]+tuple[1]*x)
    xx = np.linspace(0,1,100)
    yy1 = [Knn_uiteindelijk(a, x, y, 1) for a in xx]
    yy5 = [Knn_uiteindelijk(a, x, y, 5) for a in xx]
    yy20 = [Knn_uiteindelijk(a, x, y, 20) for a in xx]
    plt.plot(xx, yy1)
    plt.plot(xx, yy5)
    plt.plot(xx, yy20)

    plt.savefig('Verschil tussen OLS en KNN.pdf')
    plt.show()


def simulatie2():
    plt.figure()
    plt.subplot(2,1,1)
    yalgemeen = []
    xx = np.linspace(0,1,100)
    plt.plot(xx, 10 ** xx)
    for time in range(100):
        x = xsample(20, 0, 1)
        y = ysample(x, f, e=0.5)
        yalgemeen+= y
        plt.plot(x, y, 'o')
        tuple = OLSVoorschriftBeter(x, y)
        plt.plot(x, tuple[0] + tuple[1] * x)
    tuple = OLSVoorschriftBeter(x, yalgemeen)
    plt.subplot(2,1,2)
    plt.plot(x, tuple[0] + tuple[1] * x, )
    plt.show()
    plt.savefig("verschilVariantieVertekening.jpeg")


def simulatie3():
    def f(x): return 10 ** x
    x = xsample(20, 0, 1)
    y = ysample(x, f, e=.5)
    plt.figure()
    xx = np.linspace(0, 1, 100)
    minimum = 100
    minparam = 0
    for parameter in range(1,21):
        yy = [Knn_uiteindelijk(a, x, y, parameter) for a in xx]
        plt.plot(xx,yy)
        mse = MSE(xx,yy,f)
        if mse < minimum:
            minimum = mse
            minparam = parameter
    yy = [Knn_uiteindelijk(a, x, y, minparam) for a in xx]
    plt.plot(xx,yy, lw = 3)
    plt.plot(xx,10**xx)
    print(minimum)
    plt.show()
def MSE(x,y,f):
    som = 0
    for index in range(len(y)):
        som+= (y[index]-f(x[index]))**2
    return som/len(y)

# x = xsample(5, 0,20)
#
# def f(x): return 2*x
# y = ysample(x, f, e=.3)
# print(MSE(x,y,f))
X = [[1,2],[3,4]]
Y = [[5,6],[7,8]]

def mls(Y,X):
    beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    kleine_b = Y-np.dot(X,beta)

    return beta, kleine_b


def simulatie4():
    def f(x): return 1+2*x
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    steekproef = genereerSteekproef(20)
    print(steekproef[1], steekproef[2], steekproef[3])
    ax.scatter(steekproef[1], steekproef[2], steekproef[3])
    xx = steekproef[1]
    zz= [1+2*x for x in xx]
    ax.plot(xx,xx,zz)
    # tuple = mls(steekproef[3], [steekproef[1],steekproef[2]])
    # z = steekproef[1]*tuple[0] + tuple[1]
    # ax.plot_surface(xx,xx,z)
    yy1 = [Knn_uiteindelijk(a, steekproef[1], steekproef[3], 1) for a in xx]
    ax.plot(xx,xx,yy1)
    yy20 = [Knn_uiteindelijk(a, steekproef[1], steekproef[3], 20) for a in xx]
    ax.plot(xx,xx,yy20)
    k = bepaalK_opt(20,steekproef[1], steekproef[3],f)
    yyk = [Knn_uiteindelijk(a, steekproef[1], steekproef[3], k) for a in xx]
    ax.plot(xx, xx, yyk)
    plt.show()


def bepaalK_opt(Aantalpunten,x,y,f):
    xx = np.linspace(0, 1, 100)
    minimum = 100
    minparam = 0
    for parameter in range(1, Aantalpunten +1):
        yy = [Knn_uiteindelijk(a, x, y, parameter) for a in xx]
        mse = MSE(xx, yy, f)
        if mse < minimum:
            minimum = mse
            minparam = parameter
    return minparam

def genereerSteekproef(n):
    x1 = list(xsample(n))
    x2 = list(xsample(n))
    e_array = np.random.normal(0, .1, n)
    y = [1+ x1[index] + x2[index] + e_array[index] for index in range(n)]
    steekproef = [[x1[index], x2[index], y[index]] for index in range(n)]
    return steekproef, x1, x2, y

simulatie4()

# steekproef = genereerSteekproef(20)
# print(np.dot(np.transpose(steekproef[1]), steekproef[1]))