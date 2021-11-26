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
    '''
    Maakt een standaard verdeelde lijst aan
    :param n: grootte van de lijst
    :param a: ondergrens
    :param b: bovengrens
    :return: de standaard verdeelde lijst
    '''
    return np.linspace(a, b, n)

def ysample(x, f, e=0.3):
    '''
    Maakt een standaard verdeelde lijst aan met  normaalverdeelde afwijking, vanuit een functie en een lijst x-waarden
    :param x: lijst x-waarden
    :param f: functie op de x-waarden
    :param e: standaardafwijking, standaard 0.3
    :return: standaard verdeelde lijst aan met  normaalverdeelde afwijking
    '''
    yarray = []
    e_array = np.random.normal(0,e,len(x))
    for index in range(len(x)):
        yarray.append(f(x[index]) + e_array[index])
    return yarray



x = xsample(5, 0,20)
y = ysample(x, f, e=.3)

def OLSVoorschriftBeter(x,y):

    '''
    Zoekt een a en b volgens het voorschrift y = bx+a, via gegeven x en y waarden
    :param x: lijst: x-waarden
    :param y: lijst: y-waarden
    :return: tuple(a,b)
    '''
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
    '''
    berekent de norm/afstand tussen een gegeven punt x0 en alle punten van een lijst x
    :param x0: nummer of tuple: gegeven punt
    :param x: lijst of matrix: x-waarden
    :return: afstand
    '''
    lijst = []

    for index in range(len(x)):
        try:
            distance = [x[index][i] - x0[i] for i in range(len(x0))]
            lijst.append((np.linalg.norm(distance), index))
        except:lijst.append(((np.linalg.norm(x[index]-x0), index)))
    return lijst

def Knn_stap2(x0,x,k):
    '''
    kiest de kleinste k termen uit de lijst van Knn_stap1
    :param x0: nummer of tuple: gegeven punt
    :param x: lijst of matrix: x-waarden
    :param k: parameter
    :return: lijst met kleinste k termen
    '''
    lijst = Knn_stap1(x0,x)
    gesorteerd = sorted(lijst)
    slice = [gesorteerd[a][1] for a in range(k)]
    return slice


def Knn_uiteindelijk(x0, x, y , k):
    '''
    Geeft voor een gegen x0 waarde de gemiddelde y-waarde van de k dichtste termen.

    :param x0: getal of tuple: vast punt
    :param x: lijst of matrix met x-waarden
    :param y: lijst: alle y-waarden
    :param k: parameter
    :return: geschatte y-waarde
    '''
    lijst = Knn_stap2(x0, x, k)
    y_hoedje = 1/k * sum([y[index] for index in lijst])
    return y_hoedje

def simulatie1():
    '''
    plot:   -20 steekproefwaarden
            -populatiefunctie
            -liniare regressie
            -1,5 en 20 Knn

    '''
    def f(x): return 10**x
    x = xsample(20,0,1)
    y = ysample(x, f, e=1)
    plt.figure()
    plt.plot(x, 10**x, lw = 3)
    plt.scatter(x,y)
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
    '''
    100 keer een steekproef van 20 punten genereren, en er dan via liniaire regressie
    de regressierechte van plotten. Uiteindelijk met de
    2000 punten een uiteindelijke regressierechte tekenen

    '''
    #voorbereiding
    plt.figure()
    plt.subplot(2,1,1)
    yalgemeen = []
    xx = np.linspace(0,1,100)
    plt.plot(xx, 10 ** xx)
    #100 simulaties
    for time in range(100):
        x = xsample(20, 0, 1)
        y = ysample(x, f, e=0.5)
        yalgemeen+= y
        plt.plot(x, y, 'o')
        tuple = OLSVoorschriftBeter(x, y)
        plt.plot(x, tuple[0] + tuple[1] * x)
    #uiteindelijke regressierechte
    tuple = OLSVoorschriftBeter(x, yalgemeen)
    plt.subplot(2,1,2)
    plt.plot(x, tuple[0] + tuple[1] * x, )
    plt.show()
    plt.savefig("verschilVariantieVertekening.jpeg")


def simulatie3():
    '''
    Bepaalt de optimale k voor KNN-regressie, door alle k's te overlopen
    en diegene met de minste MSE met de populatiefunctie, die kiest hij als optimale.

    '''
    #voorbereiding
    def f(x): return 10 ** x
    x = xsample(20, 0, 1)
    y = ysample(x, f, e=.5)
    plt.figure()
    xx = np.linspace(0, 1, 100)
    minimum = 100
    minparam = 0
    #alle 20 k's overlopen en minimale MSE bijhouden
    for parameter in range(1,21):
        yy = [Knn_uiteindelijk(a, x, y, parameter) for a in xx]
        plt.plot(xx,yy)
        mse = MSE(xx,yy,f)
        if mse < minimum:
            minimum = mse
            minparam = parameter
    #minimale k nogmaals plotten
    yy = [Knn_uiteindelijk(a, x, y, minparam) for a in xx]
    plt.plot(xx,yy, lw = 3)
    plt.plot(xx,10**xx)
    print(minimum)
    plt.show()
def MSE(x,y,f):
    """
    Berekent de Mean Squared Error tussen functie f en waarden y
    :param x: x-waarden
    :param f: populatiefuctie die met de x-waarden y-waarden berekent
    :param y: y-waarden berekent door de benaderde functie
    :return: MSE: getal: gemiddelde kwadratische afwijking
    """
    som = 0
    for index in range(len(y)):
        som+= (y[index]-f(x[index]))**2
    return som/len(y)

def MSE_3d(XxY,z,f):
    """
    Berekent de Mean Squared Error tussen functie f en waarden z
    :param x: x-coordinaten
    :param z: z-waarden berekent door de benaderde functie
    :param f: populatiefuctie die met de x-waarden y-waarden berekent
    :param y: y-coordinaten
    :return: MSE: getal: gemiddelde kwadratische afwijking
    """
    som = 0
    for index in range(len(z)-1):
        som+= (z[index]-f(XxY[index][0],XxY[index][1]))**2
    return som/len(z)


def mls(Y,X):
    '''
    berekent aan de hand van de formule een Beta-matrix
    die gebruikt wordt voor meervoudige regressie. de functie is dan beta[0] + x1*beta[1] + ...
    :param Y: 1 x N matix met de uitgekomen y-waarden
    :param X: K+1 x N matrix, met elke rij = (1, x1, x2, ...)
    :return: Beta-matrix: 1 x N
    '''
    beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    return beta


def simulatie4():
    #voorbereidend werk
    dim = 10
    def f(x): return 1+2*x
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    X,Y = genereerSteekproef(dim)
    #alle steekproefpunten plotten
    lijstMetXCoordinaten = [X[k][1] for k in range(len(X))]
    lijstMetYCoordinaten = [X[k][2] for k in range(len(X))]
    ax.scatter(lijstMetXCoordinaten, lijstMetYCoordinaten, Y)

    #lijsten maken met alle x en y coordinaten
    x = y = xsample(dim)
    xx,yy = np.meshgrid(x,y) #rare functie, maakt van 2 lijsten matrices bv: meshgrid([1,2,3],[4,5,6]) =[[1, 2, 3],[1, 2, 3][1, 2, 3]],[[4,4,4],[5,5,5],[6,6,6]]
    #algemene functie plotten
    zz = xx+yy+1
    ax.plot_surface(xx,yy,zz, color='r', alpha = .2)

    #meervoudige regressie toepassen
    Y = np.transpose(Y)
    beta = mls(Y,X)
    regressionZZ = beta[0] + xx*beta[1] + yy*beta[2]
    ax.plot_surface(xx,yy,regressionZZ, color='g',alpha = .5)


    xx = xx.tolist()


    #Knnregressie 1 buur
    allePunten = allepunten(xx,yy)
    steekproefpunten = [X[i][1:]for i in range(len(X))]
    zz1 = [Knn_uiteindelijk(a, steekproefpunten, Y, 1) for a in allePunten]

    allePuntenX = [allePunten[k][0] for k in range(len(allePunten))]
    allePuntenY = [allePunten[k][1] for k in range(len(allePunten))]

    zz1 = np.array(vervorm(zz1,dim))
    ax.plot_surface(xx,yy,zz1,alpha = .2)

    #Knn 20 buren:
    zz20 = [Knn_uiteindelijk(a, steekproefpunten, Y, 20) for a in allePunten]
    zz20 = np.array(vervorm(zz20,dim))
    ax.plot_surface(xx,yy,zz20,alpha = .2)

    def functie(x1,x2): return x1 + x2 +1
    k = bepaalK_opt_3d(30,steekproefpunten, Y,functie)
    print("optimale k = ",k)
    zzk = [Knn_uiteindelijk(a, steekproefpunten, Y, k) for a in allePunten]

    zzk = np.array(vervorm(zzk,dim))

    ax.plot_surface(xx, yy, zzk,alpha = .5,  color='b')

    plt.show()

def vervorm(lijst, dimentie):
    """
    vervormt een vlakke lijst in een vierkante matrix van d*d
    :param lijst: lijst
    :param dimentie: d
    :return: vierkante matrix
    """
    matrix = []
    for time in range(dimentie):
        matrix.append(lijst[:dimentie])
        lijst= lijst[dimentie:]
    return matrix

def bepaalK_opt(Aantalpunten,x,y,f):
    """
    Doet zoals in simulatie 3: alle k's voor KNN-regressie overlopen en
    diegene met de minste MSE returnen

    :param Aantalpunten: Grootte van de steekproef die ingegeven wordt
    :param x: alle x-coordinaten van de steekproef
    :param y: alle y-coordinaten van de steekproef
    :param f: functie van de populatiefunctie
    :return: optimale k voor KNN-regressie
    """
    #voorbereiding
    xx = np.linspace(0, 1, 100)
    minimum = 100
    minparam = 0
    #alle k's berekenen
    for parameter in range(1, Aantalpunten +1):
        yy = [Knn_uiteindelijk(a, x, y, parameter) for a in xx]
        mse = MSE(xx, yy, f)
        if mse < minimum:
            minimum = mse
            minparam = parameter
    return minparam

def bepaalK_opt_3d(Aantalpunten, steekproefpunten,z_waarden,f):
    """
    berekent alle k's tot aan Aantalpunten, en geeft de optimale terug.

    :param Aantalpunten: Grootte van de steekproef
    :param steekproefpunten: alle x,y koppels uit de steekproef: matrix van 2*N^2
    :param z_waarden: alle z waarden uit de steekproef: lijst van N^2
    :param f: voorschrift van de populatiefunctie
    :return: optimale k
    """
    #voorbereiding
    xx = np.linspace(0, 1, 20)
    yy = np.linspace(0, 1, 20)
    XX,YY = np.meshgrid(xx,yy)
    XXxYY = allepunten(XX,YY) #is een 2 x N^2 matrix
    minimum = 100
    minparam = 0
    #berekent alle k's
    for parameter in range(1, Aantalpunten +1):
        zz = [Knn_uiteindelijk(a, steekproefpunten, z_waarden, parameter) for a in XXxYY]
        mse = MSE_3d(XXxYY, zz, f)
        if mse < minimum:
            minimum = mse
            minparam = parameter
    return minparam



def genereerSteekproef(n):
    '''

    :param: n: grootte van de steekproef
    :return:X: een matrix waar elke rij bestaat uit [1, x1 coordinaat, x2 coordinaat] van dimentie 3 x (len(x1)*len(x2)
            Y: een 1 * n matrix met de y-waarde (x1+x2+1) van elke rij uit X
    '''
    x1 = xsample(n)
    x2 = xsample(n)
    Y = []
    X = []
    for index1 in range(len(x1)):
        for index2 in range(len(x2)):
            X.append([1,x1[index1], x2[index2]])
            Y.append(1+x1[index1]+x2[index2]+np.random.normal(0,.2))
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
    """
    maakt een matrix met koppels van dim 2 x N^2
    :param lijst1: matrix van n*n met bv. x-coordinaten
    :param lijst2: matrix van n*n bv. y-coordinaten van dimentie m
    :return: matrix: met koppels van dimentie 2 x N^2
    """
    allepunten = []
    for index1 in range(len(lijst1)):
        for index2 in range(len(lijst1[index1])):
            allepunten.append([lijst1[index1][index2], lijst2[index1][index2]])
    return allepunten




simulatie4()

