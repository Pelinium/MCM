import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

array = [[math.sqrt(10),112], [math.sqrt(19*19+36), 109], [math.sqrt(121+25), 109], [math.sqrt(169+36), 118], [math.sqrt(16+25), 120], [0, 118]]
X = [math.sqrt(10),math.sqrt(19*19+36),math.sqrt(121+25),math.sqrt(169+36),math.sqrt(16+25),0]
Y = [112,109,109,118,120,118]

poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(array)

poly.fit(X_poly, Y)
lin2 = LinearRegression()
lin2.fit(X_poly, Y)
plt.scatter(X, Y, color = 'blue')

plt.plot(X, lin2.predict(poly.fit_transform(array)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')

plt.show()