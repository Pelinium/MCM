import math
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy as sp
import scipy.stats as st
import pickle as pkl
import csv as csv
import networkx as nx

main_map = np.zeros((76, 26))
#print(main_map)
array = [[math.sqrt(10),112], [math.sqrt(19*19+36), 109], [math.sqrt(121+25), 109], [math.sqrt(169+36), 118], [math.sqrt(16+25), 120], [0, 118]]

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return(b_0, b_1)

def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
                marker = "o", s = 30)

    # predicted response vector
    y_pred = b[0] + b[1]*x

    # plotting the regression line
    plt.plot(x, y_pred, color = "g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()

def main():
    # observations
    x = np.array([math.sqrt(10), math.sqrt(19*19+36),math.sqrt(121+25), math.sqrt(169+36), math.sqrt(16+25),0,math.sqrt(64+14*14)])
    y = np.array([112, 109, 109, 118, 120, 118, 100])

    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
    \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)

main()
"""
Estimated coefficients:
b_0 = 117.87950658948925      
b_1 = -0.5437227850097295
"""