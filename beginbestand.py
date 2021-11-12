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


def xsample(n, a=0, b=1):
    return np.linspace(a, b, n)


xsample(5, a=0, b=1)


