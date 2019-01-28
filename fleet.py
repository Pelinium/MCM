
import math
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy as sp
import scipy.stats as st
import pickle as pkl
import csv as csv
import database as dt
import task1 as t1
import routinePlanning as rp
import random


eHos = t1.Location(1, 22)
dHos = t1.Location(4, 48)
cHos = t1.Location(2, 53)
bHos = t1.Location(13, 54)
aHos = t1.Location(7, 71)
lastP = t1.Location(3, 34)
locBest1 = t1.Location(3, 50)
locBest2 = t1.Location(8, 54)
locBest3 = t1.Location(3, 34)

print(locBest1.getDistance(aHos))
print(locBest2.getDistance(aHos))

print(locBest1.getDistance(dHos))
print(locBest2.getDistance(dHos))


print(locBest1.getDistance(bHos))
print(locBest1.getDistance(cHos))
print(locBest1.getDistance(dHos))
print(locBest1.getDistance(eHos))



# From locBest1 to
# aHos: use B drone
# bHos: all
# cHos: all
# dHos: B D F
# eHos: not possible
'''
print(locBest2.getDistance(aHos))
print(locBest2.getDistance(bHos))
print(locBest2.getDistance(cHos))
print(locBest2.getDistance(dHos))
print(locBest2.getDistance(eHos))
'''
# From locBest2 to
# aHos: B
# bHos: all
# cHos: all
# dHos: all except E
# eHos: not possible

'''
print(locBest3.getDistance(aHos))
print(locBest3.getDistance(bHos))
print(locBest3.getDistance(cHos))
print(locBest3.getDistance(dHos))
print(locBest3.getDistance(eHos))
'''

# From locBest2 to
# aHos: not possible
# bHos: B
# cHos: B
# dHos: B D
# eHos: B D F
