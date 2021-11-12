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

def f(x): return 9*x-3

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
    y_streep = statistics.mean(y)
    somteller = somnoemer = 0
    for index in range(len(x)):
        somteller += (x[index] - x_streep)*(y[index]+y_streep)
        somnoemer += (x[index]-x_streep)**2
    b = somteller/somnoemer
    a = y_streep -b *x_streep
    return a, b



def Knn_stap1(x0, x):
    lijst = []
    for index in range(len(x)):
        lijst.append((abs(x0-x[index]), index))
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

