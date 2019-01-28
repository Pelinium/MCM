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



class Drone():
    def __init__(self, typenum):
        self.type = typenum
        if self.type == 'A':
            self.l = 45
            self.w = 45
            self.h = 25
            self.speed = 40
            self.fTime = 35
            self.bayType = 1
            self.maxCapa = 3.5
            self.videoCapa = 1
            self.maxDis = 23.3
        elif self.type == 'B':
            self.l = 30
            self.w = 30
            self.h = 22
            self.speed = 79
            self.fTime = 40
            self.bayType = 1
            self.maxCapa = 8
            self.videoCapa = 1
            self.maxDis = 52.7
        elif self.type == 'C':
            self.l = 60
            self.w = 50
            self.h = 30
            self.speed = 64
            self.fTime = 35
            self.bayType = 2
            self.maxCapa = 14
            self.videoCapa = 1
            self.maxDis = 37.3
        elif self.type == 'D':
            self.l = 25
            self.w = 20
            self.h = 25
            self.speed = 60
            self.fTime = 18
            self.bayType = 1
            self.maxCapa = 11
            self.videoCapa = 1
            self.maxDis = 18
        elif self.type == 'E':
            self.l = 25
            self.w = 20
            self.h = 27
            self.speed = 60
            self.fTime = 15
            self.bayType = 2
            self.maxCapa = 15
            self.videoCapa = 1
            self.maxDis = 15
        elif self.type == 'F':
            self.l = 40
            self.w = 40
            self.h = 25
            self.speed = 79
            self.fTime = 24
            self.bayType = 2
            self.maxCapa = 22
            self.videoCapa = 0
            self.maxDis = 31.6
        else:
            self.l = 32
            self.w = 32
            self.h = 17
            self.speed = 64
            self.fTime = 16
            self.bayType = 2
            self.maxCapa = 20
            self.videoCapa = 1
            self.maxDis = 17.1

class Location:
    def __init__(self, positionx, positiony):
        self.x = positionx
        self.y = positiony

    def getDistance(self,other):
        return math.sqrt((self.x-other.x)**2.0+(self.y-other.y)**2.0)

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

def get_rainfall_map():
    results = []
    with open("rainfall_matrix.csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            results.append(row)
    results = np.array(results)
    return results


def get_resistance_map(map1,map2,weight1,weight2):
    candidate1 = get_weighted_map(map1,weight1)
    candidate2 = get_weighted_map(map2,weight2)
    result = np.zeros((25,72))
    for i in range(0, 25):
        for j in range(0, 72):
            result[i][j] = candidate1[i][j]+candidate2[i][j]
    return result

def get_weighted_map(map,weight):
    temp_map = map
    max = temp_map[0][0]
    min = temp_map[0][0]

    for i in range(0,25):
        for j in range(0,72):
            if temp_map[i][j] != 1:
                if temp_map[i][j] >= max:
                    max = temp_map[i][j]
                if temp_map[i][j] <= min:
                    min = temp_map[i][j]
    difference = max - min
    scale_factor = weight/difference
    for i in range(0,25):
        for j in range(0,72):
            if temp_map[i][j] != 1:
                temp_map[i][j] -= min
                temp_map[i][j] *= scale_factor
    return temp_map


def getMax(dict):
    max = 0
    for i in dict:
        if dict[i] > max:
            key = i
            max = dict[i]
    return key

def pickEhosTiles(map, percentile):
    size = int(percentile/100 * 72 * 25)
    dict = {}
    for i in range(0, 25):
        for j in range(0, 72):
            loc = t1.Location(i, j)
            if (loc.getDistance(eHos) <= 52.7):
                if(loc.getDistance(dHos) <= 52.7 or loc.getDistance(cHos) <= 52.7):
                    if (len(dict) < size):
                        dict[i*100+j] = map[i][j]
                    else:
                        if (map[i][j] < dict[getMax(dict)]):
                            del dict[getMax(dict)]
                            dict[i*100+j] = map[i][j]

    return dict


def pickTiles(map1, percentile):
    size = int(percentile/100 * 72 * 25)
    dict = {}
    for i in range(0, 25):
        for j in range(0, 72):
            if (len(dict) < size):
                dict[i*100+j] = map1[i][j]
            else:
                if (map1[i][j] < dict[getMax(dict)]):
                    del dict[getMax(dict)]
                    dict[i*100+j] = map1[i][j]

    return dict

def crop(map, l1, l2):
    if (l1.x < l2.x):
        output = np.zeros((l2.x - l1.x + 1,l2.y - l1.y + 1))
        for i in range (l1.x, l2.x + 1):
            for j in range (l1.y, l2.y + 1):
                output[i - l1.x][j - l1.y] = map[i][j]
        return output
    else:
        # print(l1.x - l2.x)
        output = np.zeros((l1.x - l2.x + 1,l1.y - l2.y + 1))
        for i in range (l2.x, l1.x + 1):
            # print (i)
            for j in range (l2.y, l1.y + 1):
                # print (l1.y)
                output[i - l2.x][j - l2.y] = map[i][j]
        # print (output)
        return output

def flipCrop(map, l1, l2):
    if (l1.x < l2.x) or (l1.y > l2.y):
        output = np.zeros((l2.x - l1.x + 1,l1.y - l2.y + 1))
        for i in range (l1.x, l2.x + 1):
            for j in range (l2.y, l1.y + 1):
                output[i - l1.x][j - l2.y] = map[l2.x - l1.x - i][l1.y - l2.y - j]
        return output
    else:
        # print (l1.x - l2.x, l2.y - l1.y)
        output = np.zeros((l1.x - l2.x + 1, l2.y - l1.y + 1))
        for i in range (l2.x, l1.x + 1):
            for j in range (l1.y, l2.y + 1):
                output[i - l2.x][j - l1.y] = map[l1.x - l2.x - i][l2.y - l1.y - j]
        return output

def getMinWeight(map, l1, l2):
    if ((l1.x == l2.x) and (l1.y == l2.y)):
        return 0
    elif ((l1.x >= l2.x) and (l1.y <= l2.y)):
        return rp.minCost(flipCrop(map, l1, l2), l1.x - l2.x, l2.y - l1.y)

    elif ((l1.x <= l2.x) and (l1.y >= l2.y)):
        return rp.minCost(flipCrop(map, l1, l2), l2.x - l1.x, l1.y - l2.y)

    elif ((l1.x <= l2.x) and (l1.y <= l2.y)):
        return rp.minCost(crop(map, l1, l2), l2.x - l1.x, l2.y - l1.y)

    elif ((l1.x >= l2.x) and (l1.y >= l2.y)):
        return rp.minCost(crop(map, l1, l2), l1.x - l2.x, l1.y - l2.y)

def printTiles(tilemap):
    result = np.zeros((25,72))
    for i in tilemap:
        y = i % 100
        x = int(i/100)
        #print (x)
        result[x][y] = tilemap[i]
    plt.imshow(result,cmap = "Greens")
    plt.colorbar()
    plt.show()

def adjImpedence(adjPar, loc):
    for i in adjPar:
        y = i % 100
        x = int(i/100)
        tarTile = t1.Location(x, y)
        adjPar[i] = adjPar[i] * (np.exp(-tarTile.getDistance(loc)) + 1)
    return adjPar

def removeImpedence(adjPar, loc):
    for i in adjPar:
        y = i % 100
        x = int(i/100)
        tarTile = t1.Location(x, y)
        adjPar[i] = adjPar[i] / (np.exp(-tarTile.getDistance(loc)) + 1)
    return adjPar

def avaiDists(loc):
    retVal = []
    if (loc.getDistance(aHos) <= 52.7):
        retVal.append('a')
    if (loc.getDistance(bHos) <= 52.7):
        retVal.append('b')
    if (loc.getDistance(cHos) <= 52.7):
        retVal.append('c')
    if (loc.getDistance(dHos) <= 52.7):
        retVal.append('d')
    # if (loc.getDistance(eHos) <= 52.7):
    #     retVal.append('e')
    return retVal

def getAddedImped(tarLoc):
    addedImped = 0
    if ('a' in avaiDists(tarLoc)):
        addedImped += getMinWeight(dt.destruction_wind_rainfall_analysis(), aHos, tarLoc)
    if ('b' in avaiDists(tarLoc)):
        # print (tarLoc.x, tarLoc.y)
        # print (getMinWeight(dt.destruction_wind_rainfall_analysis(), bHos, tarLoc))
        addedImped += getMinWeight(dt.destruction_wind_rainfall_analysis(), bHos, tarLoc)
    if ('c' in avaiDists(tarLoc)):
        addedImped += getMinWeight(dt.destruction_wind_rainfall_analysis(), cHos, tarLoc)
    if ('d' in avaiDists(tarLoc)):
        addedImped += getMinWeight(dt.destruction_wind_rainfall_analysis(), dHos, tarLoc)
    #if ('e' not in avaiDists(tarLoc)):
        # return 10000000
        #addedImped += getMinWeight(dt.destruction_wind_rainfall_analysis(), eHos, tarLoc)
    if len(avaiDists(tarLoc)) == 0:
        return 10000000
    return addedImped/len(avaiDists(tarLoc))/np.sqrt(len(avaiDists(tarLoc)))

def adjedTile(addedImped, adjImped):
    adjdImped = {}
    for i in addedImped:
        adjdImped[i] = addedImped[i] * adjImped[i]
    return adjdImped

def downToFifty(tiles):
    adjPar = {}   #initialize adjust parameters
    lastFifty = {}
    counter = 1
    for i in tiles:
        adjPar[i] = 1

    keys = list(tiles.keys())
    random.shuffle(keys)
    for i in keys:
        print (counter)

        '''
        if(counter == 135):
            print ('1/4 DONE!!')
            printTiles(lastFifty)
        if(counter == 270):
            print ('HALF DONE!!!')
            printTiles(lastFifty)
        if(counter == 405):
            print ('3/4 DONE!!!!')
            printTiles(lastFifty)
        '''

        counter += 1
        y = i % 100
        x = int(i/100)
        tarLoc = t1.Location(x, y)

        if(len(lastFifty) < 50):
            lastFifty[i] = getAddedImped(tarLoc)
            adjPar = adjImpedence(adjPar, tarLoc)

        else:
            temp = adjedTile(lastFifty, adjPar)
            #print (i)
            ff = getAddedImped(tarLoc)
            if (temp[getMax(temp)] > ff*adjPar[i]):
                adx = getMax(temp) % 100
                ady = int(getMax(temp)/100)
                delLoc = t1.Location(adx, ady)
                del lastFifty[getMax(temp)]
                lastFifty[i] = ff
                adjPar = adjImpedence(adjPar, tarLoc)
                adjPar = removeImpedence(adjPar, delLoc)
    return lastFifty

def pickOneEtile(tiles):
    lastOne = {}
    for i in tiles:
        y = i % 100
        x = int(i/100)
        tarLoc = t1.Location(x, y)
        curMin = getMinWeight(dt.destruction_wind_rainfall_analysis(), eHos, tarLoc)
        if(len(lastOne) == 0):
            lastOne[i] = curMin
        elif (lastOne[list(tiles.keys())[0]] > curMin):
            lastOne.clear()
            lastOne[i] = curMin
    return lastOne

def checkConnection(loc):
    retVal = []
    if (loc.getDistance(aHos) <= 52.7):
        retVal.append('a')
    if (loc.getDistance(bHos) <= 52.7):
        retVal.append('b')
    if (loc.getDistance(cHos) <= 52.7):
        retVal.append('c')
    if (loc.getDistance(dHos) <= 52.7):
        retVal.append('d')
    if (loc.getDistance(lastP) <= 52.7):
        retVal.append('3')
    return retVal

def connxionPts(retVal):
    if('3' in retVal):
        return (3 + len(retVal))
    else:
        return len(retVal)

def removeBonuses(map, loc):
    for i in range (25):
        for j in range (72):
            temp = t1.Location(i, j)
            if (temp.getDistance(loc) <= 52.7):
                map[i][j] = 0
    return map

def arrToDict(array_1, array_2):
    dict = {}
    for i in range (len(array_2)):
        a = array_1[i][0]*100 + array_1[i][1]
        dict[a] = array_2[i]
    return dict

def roadPts(l1, l2):
    pts = 0
    roadMap = removeBonuses(dt.get_highway_map(), lastP)
    # print (roadMap)
    for i in range (25):
        for j in range (72):
            temp = t1.Location(i, j)
            if (temp.getDistance(l1) <= 52.7):
                if(roadMap[i][j] != 0):
                    pts += 1
    roadMap = removeBonuses(roadMap, l1)
    for i in range (25):
        for j in range (72):
            temp = t1.Location(i, j)
            if (temp.getDistance(l2) <= 52.7):
                if(roadMap[i][j] != 0):
                    pts += 1
    return pts

def populationPts(l1, l2):
    pts = 0
    popMap = removeBonuses(dt.get_populaton_map(), lastP)
    for i in range (25):
        for j in range (72):
            temp = t1.Location(i, j)
            if (temp.getDistance(l1) <= 52.7):
                pts += popMap[i][j]
    popMap = removeBonuses(popMap, l1)
    for i in range (25):
        for j in range (72):
            temp = t1.Location(i, j)
            if (temp.getDistance(l2) <= 52.7):
                pts += popMap[i][j]
    return pts

def lastTwo(tiles, s1, s2, s3):
    final = np.zeros((2,2))
    points = 0
    '''
    t_1 = []
    t_2 = []
    t_3 = []
    '''
    for i in tiles:
        temp_1 = t1.Location(int(i/100), i % 100)
        for j in tiles:
            if (i != j):

                temp_2 = t1.Location(int(j/100), j % 100)
                cPts = connxionPts(checkConnection(temp_1)) + connxionPts(checkConnection(temp_2))
                rPts = roadPts(temp_1, temp_2)
                pPts = populationPts(temp_1, temp_2)
                total = (cPts - 11.44)/3.8995 * s1 + (rPts - 64.18)/2.295 * s2 + (pPts - 641.6)/8.2486 * s3

                if (total > points):
                    final = [[temp_1.x, temp_1.y], [temp_2.x, temp_2.y]]
                    points = total
    return final

def finalOutput(l1, l2, l3):
    output = np.zeros((25, 72))
    output[l1.x][l1.y] = 1
    output[l2.x][l2.y] = 1
    output[l3.x][l3.y] = 1

    plt.imshow(output,cmap = "Greens")
    plt.show()

# f = t1.Location(3,55)
# g = t1.Location(6,46)

# roadPts(f, g)

# print (arrToDict(dt.fif, dt.fif_val))

# print (lastTwo(arrToDict(dt.fif, dt.fif_val), 1, 1, 1))

finalOutput(locBest1,locBest2, locBest3)

'''
tiles = pickTiles(dt.destruction_analysis(), 30)
fifty = downToFifty(tiles)
ax = []
bx = []
for i in fifty:
    ax.append([int(i/100), i % 100])
    bx.append(fifty[i])
print (ax)
print (bx)
printTiles(fifty)
'''


#lastTwo(fifty)

'''
tiles = pickTiles(dt.destruction_analysis(), 30)
fifty = downToFifty(tiles)
printTiles(fifty)
'''

'''
tilesE = pickEhosTiles(dt.destruction_analysis(), 5)
# printTiles(tilesE)
Etile = pickOneEtile(tilesE)
eKey = list(Etile.keys())
third = t1.Location(int(eKey[0]/100), eKey[0] % 100)
print(third.x, third.y)
# printTiles(Etile)
'''
'''
loc = t1.Location(1,1)
loc2 = t1.Location(1,2)
print(loc.getDistance(loc2))



tiles = pickTiles(dt.get_result_resistance_map(), 30)
print (len(tiles))
'''
'''
l1 = t1.Location(9, 6)
l2 = t1.Location(18, 10)
# print (l1.x, l2.x)
print (getMinWeight(dt.get_result_resistance_map(), l1, l2))
'''








