import numpy as np
import matplotlib.pyplot as plt
import math

class Drone:
    def __init__(self, _type):
        self.type = _type
    def getVolume(self):
        if self.type == 'A':
            return 45*45*45
        elif self.type == 'B':
            return 30*30*22
        elif self.type == 'C':
            return 60*50*30
        elif self.type == 'D':
            return 25*20*25
        elif self.type == 'E':
            return 25*20*27
        elif self.type == 'F':
            return 40*40*25
        elif self.type == 'G':
            return 32*32*17

class Location:
    def __init__(self, positionx, positiony):
        self.x = positionx
        self.y = positiony

    def getDistance(self,other):
        return 2.3*math.sqrt((self.x-other.x)**2.0+(self.y-other.y)**2.0)

    def getPointsAtDistance(self, distance):
        listAllPointsAtDistance = []
        firstx = self.x - distance
        while firstx < self.x + distance:
            firsty = self.y - distance
            while firsty < self.y + distance:
                testDistance = round((math.sqrt((self.x-firstx)**2.0+(self.y-firsty)**2.0)), 2)
                if testDistance == distance:
                    listAllPointsAtDistance.append([firstx, firsty])
                firsty += 0.02
            firstx += 0.02
        return listAllPointsAtDistance

def getIntersection(x,y):
    for i in x:
        for j in y:
            if round(i[0],1) == round(j[0],1) and round(i[1],1) == round(j[1],1):
                print (i)

# Calculated location of hospital
A = Location(36.58, -20.64)
B = Location(0, 0)
C = Location(15, 20)
D = Location(5.57, 23.34)
E = Location(-29.42, 73.31)

# Task1: Given we have x drone A in each place
# Given 3 locations, can we finish the task?
# Assume x < 5 and x > 1
def task1(alpha, beta, delta, x):
    distanceFromA = [A.getDistance(alpha), A.getDistance(beta), A.getDistance(delta)]
    distanceFromA.sort()
    distanceFromB = [B.getDistance(alpha), B.getDistance(beta), B.getDistance(delta)]
    distanceFromB.sort()
    distanceFromC = [C.getDistance(alpha), C.getDistance(beta), C.getDistance(delta)]
    distanceFromC.sort()


#plt.plot([A.x,B.x,C.x,D.x,E.x],[A.y,B.y,C.y,D.y,E.y],'ro')
#plt.show()
'''
x = np.array(C.getPointsAtDistance(46.00))
y = np.array(B.getPointsAtDistance(42.00))
'''
'''
plt.plot(x[:,0],x[:,1],'ro')
plt.plot(y[:,0],y[:,1],'ro')
plt.show()
'''
'''
a = Location(1, 22)
b = Location(4, 48)
d = Location(13, 54)
e = Location(7, 71)

print(e.getDistance(d))
'''
