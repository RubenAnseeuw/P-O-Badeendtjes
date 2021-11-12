# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:55:58 2021

@author: ansee
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
sp.init_printing(pretty_print=False)
import random

from numpy.random import *

def f(x): return x**2

def xsample(n, a=0, b=1):
    return np.linspace(a, b, n)

def ysample(x, f, e=1):
    yarray = []
    e_array = np.random.normal(0,e,len(x))
    for index in range(len(x)):
        yarray.append(f(x[index]) + e_array[index])
    return yarray



xsample(5, a=0, b=1)
print(ysample([1,2,3,4,5], f, e=1))
# print(np.random.normal(2,3,5))
