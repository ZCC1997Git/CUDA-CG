#a python script to plot the contour of field

import numpy as np
import matplotlib.pyplot as plt
def plot():
    xCrd  = np.loadtxt('./output/xr.dat')
    yCrd  = np.loadtxt('./output/yr.dat')
    field = np.loadtxt('./output/solution.dat')
    plt.contour(xCrd,yCrd, field,50)
    plt.colorbar()
    plt.savefig("./output/SimulationResult.png")

plot()