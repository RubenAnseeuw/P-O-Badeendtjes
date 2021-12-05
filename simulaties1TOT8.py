# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:55:58 2021

@author: ansee
"""
import warnings
from math import *
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
sp.init_printing(pretty_print=False)
import random
import statistics
from numpy.random import *
from mpl_toolkits.mplot3d import Axes3D
import sklearn as skl
from sklearn.linear_model import LinearRegression
import pandas as pd
def f(x): return 10**x

def xsample(n, a=0, b=1):
    '''
    Maakt een standaard verdeelde lijst aan
    :param n: grootte van de lijst
    :param a: ondergrens
    :param b: bovengrens
    :return: de standaard verdeelde lijst
    '''
    return np.random.uniform(a, b, n)

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
    xx = np.linspace(0, 10, 50)
    yy = np.linspace(0, 10, 50)
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
    #voorbereiding
    def f(x): return 10**x
    basisx = xsample(20,0,1)
    basisy = ysample(basisx, f, e=.2)
    #steekproef en populatiefunctie
    xx = np.linspace(0,1,100)
    plt.figure()
    plt.scatter(basisx, basisy)
    yy = [f(x) for x in xx]
    plt.plot(xx, yy)
    #polynomale regressie
    opt_parameter, betalijst = optimale_exponent(basisx, basisy, f)
    beta1 = betalijst[0]
    beta19 = betalijst[18]
    betaopt = betalijst[opt_parameter-1]
    yy1 = [sum([beta1[i] * (x1 ** i) for i in range(len(beta1))]) for x1 in xx]
    yy19 = [sum([beta19[i] * (x1 ** i) for i in range(len(beta19))]) for x1 in xx]
    yyopt = [sum([betaopt[i] * (x1 ** i) for i in range(len(betaopt))]) for x1 in xx]
    plt.plot(xx, yy1, label="p=1")
    plt.plot(xx,yy19,label="p=19")
    plt.plot(xx,yyopt,label= "optimaal")


    plt.legend()
    plt.show()

def optimale_exponent(x,y,f):
    xx = np.linspace(0, 1, 1000)
    minimum = 10000
    minparam = 500
    betalijst = []
    ylijst = []
    for p in range(1,20):
        X = []
        #X = np.array([[1, x1, x1**2] for x1 in x])
        for x1 in x:
            rij = []
            for power in range(p+1):
                rij.append(x1**power)
            X.append(rij)

        beta = mls(y, X)
        betalijst.append(beta)
        yy = [sum([beta[i] * (x1 ** i) for i in range(len(beta))]) for x1 in xx]
        mse = MSE(xx, yy, f)
        if mse < minimum:
            minimum = mse
            minparam = p

    return minparam, betalijst



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


def simulatie6():
    def f(x): return 10**x
    plt.figure()
    x = xsample(20)
    y = ysample(x, f)


    xx = np.linspace(0,1,100)
    plt.plot(xx, f(xx), label="populatiefunctie")

    Knn_param = bepaalK_opt(20,x,y,f)
    yy_KNN = [Knn_uiteindelijk(a, x, y, Knn_param) for a in xx]
    plt.plot(xx, yy_KNN, label = "KNN")


    Polynom_param, betalijst = optimale_exponent(x,y,f)
    betaopt = betalijst[Polynom_param - 1]
    yy_pol_opt = [sum([betaopt[i] * (x1 ** i) for i in range(len(betaopt))]) for x1 in xx]
    plt.plot(xx,yy_pol_opt, label = "Polynom")

    mse_knn= MSE(xx,yy_KNN,f)
    mse_pol = MSE(xx,yy_pol_opt,f)
    if mse_pol< mse_knn:
        print("Polynomiale regressie is beter")
    else:
        print("KNN-regressie is beter")
    plt.legend()
    plt.show()



def simulatie7():
    k=10
    warnings.filterwarnings("ignore")
    def f(x): return np.sin(x) * np.cos(x)
    plt.figure()
    plt.subplot(4,1,3)
    Knnlijst = []
    Polylijst = []
    xx = np.linspace(0, 10, 1000)
    # plt.scatter(xx, f(xx))
    plt.plot(xx, f(xx), label="populatiefunctie")
    for grootte in range(k):
        x = xsample(20*2**grootte,0,10)
        y = ysample(x, f)

        plt.subplot(4,1,1)
        try: Knn_param = bepaalK_opt(40, x, y, f)
        except: Knn_param = bepaalK_opt(20*2**grootte, x, y, f)
        Knnlijst.append(Knn_param)
        yy_KNN = [Knn_uiteindelijk(a, x, y, Knn_param) for a in xx]
        plt.plot(xx, yy_KNN, label=str("KNN"+str(grootte)))
        plt.legend()


        plt.subplot(4,1, 2)
        Polynom_param, betalijst = optimale_exponent(x, y, f)
        Polylijst.append(Polynom_param)
        betaopt = betalijst[Polynom_param - 1]
        yy_pol_opt = [sum([betaopt[i] * (x1 ** i) for i in range(len(betaopt))]) for x1 in xx]
        plt.plot(xx, yy_pol_opt, label=str("Polynom"+str(grootte)))
        plt.legend()

        mse_knn = MSE(xx, yy_KNN, f)
        mse_pol = MSE(xx, yy_pol_opt, f)
        if mse_pol < mse_knn:
            print("Polynomiale regressie is beter bij",grootte)
        else:
            print("KNN-regressie is beter bij", grootte)

    plt.subplot(4,1,4)
    plt.plot(range(k),Knnlijst, label = "knn")
    plt.plot(range(k),Polylijst,label = "pol")
    plt.legend()
    plt.show()

def simulatie8():
    def f(x): return 10**x
    plt.figure()
    x = xsample(20)
    y = ysample(x, f)
    z= [np.log10(yy) for yy in y]
    a,b = OLSVoorschriftBeter(x,z)

    xx = np.linspace(0,1,20)
    yy = [b*x1+a for x1 in xx]
    plt.plot(xx,np.log10(f(x)),label = "popul")
    plt.plot(xx, yy, label = "regressie")


    def g(x): return np.log10(f(x))
    print(MSE(x,z,g))

    Polynom_param, betalijst = optimale_exponent(x, y, f)

    betaopt = betalijst[Polynom_param - 1]
    yy_pol_opt = [sum([betaopt[i] * (x1 ** i) for i in range(len(betaopt))]) for x1 in xx]
    plt.plot(xx,yy_pol_opt)
    print(MSE(xx,yy_pol_opt,f))
    plt.plot(xx,f(xx))
    plt.legend()
    plt.show()


def simulatie9():
    # fig =   plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # model = LinearRegression()
    # Xtrain, ytrain = genereerSteekproef(20)
    # Xtest = genereerSteekproef(20)[0]
    # model.fit(Xtrain, ytrain)
    # model.intercept_
    # coaf = list(model.coef_)
    # print(coaf)
    # yhat = model.predict(Xtest)
    # print("dit",yhat)
    # print(Xtest)
    # xx = [Xtest[k][1]for k in range(len(Xtest))]
    # yy = [Xtest[k][2] for k in range(len(Xtest))]
    # print(xx,yy)
    #
    #
    # ax.scatter(xx,yy, np.array(yhat))
    #
    # plt.show()

    data = pd.read_csv('new1.csv')
    data.head()  # eerste lijnen van dataset tonen
    data.info()  # informatie over dataset tonen
    data[['dimensie/0/_name','dimensie/0/__text','dimensie/1/_name','dimensie/1/__text','dimensie/2/_name','dimensie/2/__text','dimensie/3/_name','dimensie/3/__text']]
    data = data.values.tolist()

    x_lijst=[]
    y_lijst=[]
    xx = []
    yy =[]
    xxx= []
    yyy = []
    for row in data:

        if row[-1] == "motor":

            if row[5] == "snelwegen":
                x_lijst.append(float(row[3]))
                y_lijst.append((row[1]).replace(',',"."))
            elif row[5] == "gewestwegen":
                xx.append(float(row[3]) )
                yy.append((row[1]).replace(',', "."))
            else:
                xxx.append(float(row[3]) )
                yyy.append((row[1]).replace(',', "."))
    print(x_lijst,y_lijst,xx,yy)
    plt.figure()
    plt.scatter(x_lijst[14],y_lijst[14],c='y')
    # plt.scatter(xx,yy)
    # plt.scatter(xxx, yyy)



    Xtrain = [x_lijst, xx, xxx]
    ytrain = [y_lijst,yy,yyy]
    model = LinearRegression()
    for i in range(len(Xtrain)):
        Xxxx= np.array([[1,Xtrain[i][k]]for k in range(len(Xtrain[i]))])
        XX = map(lambda x: float(x), Xxxx)
        yy = [round(float(y),2) for y in ytrain[i]]
        print(type(Xxxx),"\n",type(ytrain[i]))
        Xas = np.linspace(1975,2023,1000)
        model.fit(Xxxx, ytrain[i])
        model.intercept_
        coef = list(model.coef_)
        print(coef)
        print(mls(np.array(np.transpose(np.array(yy))),Xxxx))
        yhat = model.predict([[1,2000]])
        print("dit",yhat, yy[14])
        plt.scatter(2000,yhat[0],c='r')
        plt.scatter(2000,yy[14],c='b')
        # plt.plot(Xas,(Xas-1900)*coef[0])
    plt.show()
# simulatie9()



#neural networks in sklearn:

