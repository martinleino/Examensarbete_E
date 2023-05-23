import numpy as np
import sdeint
import matplotlib.pyplot as plt
from math import sqrt as sqrt

plt.close('all')

"""
Set the parameters
"""

beta = 0.8
c_delta = 1

"""
Define drift and diffusion matrix
"""

def f(u, t):
    return np.array([4*u[0]*(beta-u[1]), -8*u[1]*(u[1]-c_delta)])

def g(u, t):
    return np.array([[2*sqrt((2*c_delta*u[1])), 0], [0, 0]])

"""
Set initial conditions and timespan, integrate SDE
"""

initial_conditions = np.array([np.random.randn(), 1.5])
t_span = np.linspace(0, 10, 15001)

result = sdeint.itoint(f, g, initial_conditions, t_span)

"""
Plot results
"""
plt.figure(1)
plt.plot(t_span, result[:,0])
plt.xlabel('Time')
plt.ylabel('m')
plt.figure(2)
plt.plot(t_span, result[:,1])
plt.xlabel('Time')
plt.ylabel('r^2')
plt.show()
