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
        try:
            distance = [x[index][i] - x0[i] for i in range(len(x0))]
            lijst.append((np.linalg.norm(distance), index))
        except:lijst.append(((np.linalg.norm(x[index]-x0), index)))
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
def MSE(x,z,f,y):
    som = 0
    for index in range(len(y)):
        som+= (z[index]-f(x[index],y[index]))**2
    return som/len(y)






X = [[1,2],[3,4]]
Y = [[5,6],[7,8]]

def mls(Y,X):
    beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    return beta


def simulatie4():
    #voorbereidend werk
    dim = 20
    def f(x): return 1+2*x
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    steekproef = genereerSteekproef(dim)

    #alle punten plotten
    lijstMetXCoordinaten = [steekproef[3][k][1] for k in range(len(steekproef[3]))]
    lijstMetYCoordinaten = [steekproef[3][k][2] for k in range(len(steekproef[3]))]
    ax.scatter(lijstMetXCoordinaten, lijstMetYCoordinaten, steekproef[2])

    #lijsten maken met alle x en y coordinaten
    x = y = xsample(dim)
    xx,yy = np.meshgrid(x,y)

    #algemene functie plotten
    zz = xx+yy+1
    # ax.plot_surface(xx,yy,zz, color='r', alpha = .5)

    #meervoudige regressie toepassen
    X = steekproef[3]
    Y = np.transpose(steekproef[2])
    beta = mls(Y,X)
    regressionZZ = beta[0] + xx*beta[1] + yy*beta[2]
    ax.plot_surface(xx,yy,regressionZZ, color='g',alpha = .5)


    xx = xx.tolist()


    #Knnregressie 1 buur
    allePunten = allepunten(xx,yy)
    steekproefpunten = [steekproef[3][i][1:]for i in range(len(steekproef[3]))]
    zz1 = [Knn_uiteindelijk(a, steekproefpunten, steekproef[2], 1) for a in allePunten]

    allePuntenX = [allePunten[k][0] for k in range(len(allePunten))]
    allePuntenY = [allePunten[k][1] for k in range(len(allePunten))]

    zz1 = np.array(vervorm(zz1,dim))
    # ax.plot_surface(xx,yy,zz1)

    #Knn 20 buren:
    zz20 = [Knn_uiteindelijk(a, steekproefpunten, steekproef[2], 20) for a in allePunten]
    zz20 = np.array(vervorm(zz20,dim))
    # ax.plot_surface(xx,yy,zz20)

    def functie(x1,x2): return x1 + x2 +1
    k = bepaalK_opt_3d(20,allePunten,steekproefpunten, steekproef[2],functie)
    print(k)
    zzk = [Knn_uiteindelijk(a, steekproefpunten, steekproef[2], k) for a in allePunten]
    print(xx,yy,zzk)
    zzk = np.array(vervorm(zzk,dim))
    print(xx, "\n",yy,"\n", zzk)
    ax.plot_surface(xx, yy, zzk)

    plt.show()

def vervorm(lijst, dimentie):
    """
    vervormt een lijst in een vierkante matrix van d*d
    :param lijst: lijst
    :param dimentie: d
    :return: matrix
    """
    matrix = []
    for time in range(dimentie):
        matrix.append(lijst[:dimentie])
        lijst= lijst[dimentie:]


    return matrix

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

def bepaalK_opt_3d(Aantalpunten,allePunten, steekproefpunten,z_waarden,f):
    xx = np.linspace(0, 1, 100)
    yy = np.linspace(0, 1, 100)

    minimum = 100
    minparam = 0
    for parameter in range(1, Aantalpunten +1):
        zz = [Knn_uiteindelijk(a, steekproefpunten, z_waarden, parameter) for a in allePunten]
        mse = MSE(xx, zz, f, yy)
        if mse < minimum:
            minimum = mse
            minparam = parameter
    return minparam

def genereerSteekproef(n):
    x1 = xsample(n)
    x2 = xsample(n)
    e_array = np.random.normal(0, 0, n)
    def y(x1,x2,e): return 1 + x1 + x2 + e
    # y = [1+ x1[index] + x2[index] + e_array[index] for index in range(n)]
    # steekproef = [[x1[index], x2[index], y[index]] for index in range(n)]
    X ,Y = genereerXenY(x1, x2)

    return x1, x2, Y, X


def genereerXenY(lijst1, lijst2):
    '''

    :param lijst1: x1 lijst
    :param lijst2: x2 lijst
    :param e: normaal verdeelde getallen
    :return:X: een matrix waar elke rij bestaat uit [1, x1 coordinaat, x2 coordinaat] van dimentie 3 x (len(x1)*len(x2)
    '''
    Y = []
    X = []
    for index1 in range(len(lijst1)):
        for index2 in range(len(lijst2)):
            X.append([1,lijst1[index1], lijst2[index2]])
            Y.append(1+lijst1[index1]+lijst2[index2]+np.random.normal(0,.2))
    return X, Y


def simulatie5():
    def f(x): return 10**x
    basisx = xsample(20)
    basisy = ysample(basisx, f, e=.2)
    xx = np.linspace(0,1,100)
    plt.figure()
    plt.scatter(basisx, basisy)
    yy = [f(x) for x in xx]
    plt.plot(xx, yy)




    plt.show()


def allepunten(lijst1, lijst2):
    allepunten = []
    for index1 in range(len(lijst1)):
        for index2 in range(len(lijst1[index1])):
            allepunten.append([lijst1[index1][index2], lijst2[index1][index2]])
    return allepunten




simulatie4()
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# x = np.linspace(0,1,5)
# y = np.linspace(2,3,5)
#
# xx,yy = np.meshgrid(x,y)
# print(xx,yy)
# zz = xx+yy
# zz[2][2] = 2
# ax.plot_surface(xx,yy,zz)
# plt.show()
# print(zz)

# def funcite(x,y): return x+y+5
#
# xlijst = ylijst =[1,2,3,4]
# zlijst= [6,6,7,9]
# print(MSE(xlijst, zlijst, funcite, ylijst))
