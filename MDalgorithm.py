
import math
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy as sp
import scipy.stats as st
import pickle as pkl
import csv as csv
import networkx as nx
main_map = np.zeros((72, 26))

# 1 unit = 1.8 mile = 2.88 km

main_map[16][1] = 1
main_map[17][1] = 1
main_map[18][1] = 1
main_map[18][2] = 1
main_map[19][2] = 1

main_map[20][2] = 1
main_map[21][3] = 1
main_map[22][3] = 1
main_map[23][3] = 1
main_map[24][4] = 1

main_map[25][4] = 1
main_map[26][4] = 1
main_map[26][5] = 1
main_map[27][5] = 1
main_map[28][5] = 1

main_map[29][6] = 1
main_map[30][6] = 1
main_map[31][6] = 1
main_map[32][7] = 1
main_map[33][7] = 1
main_map[34][7] = 1
main_map[35][8] = 1
main_map[36][8] = 1
main_map[37][8] = 1
main_map[37][9] = 1

main_map[38][9] = 1
main_map[39][9] = 1
main_map[40][10] = 1
main_map[41][10] = 1
main_map[42][10] = 1

main_map[43][11] = 1
main_map[44][11] = 1
main_map[45][11] = 1
main_map[46][12] = 1
main_map[47][12] = 1

main_map[48][13] = 1
main_map[49][13] = 1
main_map[50][13] = 1
main_map[51][14] = 1
main_map[52][14] = 1

main_map[53][15] = 1
main_map[54][15] = 1
main_map[55][16] = 1
main_map[56][16] = 1
main_map[57][17] = 1

main_map[58][17] = 1
main_map[58][18] = 1
main_map[58][19] = 1
main_map[59][20] = 1
main_map[60][20] = 1

main_map[61][21] = 1
main_map[62][21] = 1
main_map[63][22] = 1
main_map[64][23] = 1
main_map[65][23] = 1

main_map[66][23] = 1
main_map[66][24] = 1
main_map[67][24] = 1
main_map[68][25] = 1
main_map[69][25] = 1

shape_map = main_map.shape
routine = []
for i in range(0, 72):
    for j in range(0, 26):
        if main_map[i][j] == 1:
            routine.append([i, j])

def getMinimumDistance(x,y):
    distance = []
    for i in routine:
        distance.append(math.sqrt((x-i[0])**2+(y-i[1])**2))
    return min(distance)
def getWindSpeed(distance):
    return -0.5437227850097295*distance + 117.87950658948925

for i in range(0, 72):
    for j in range(0, 26):
        if main_map[i][j] != 1:
            main_map[i][j] = getWindSpeed(getMinimumDistance(i,j))


print (main_map)