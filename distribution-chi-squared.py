from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math
import random

X_RANGE = 1000

DF_1 = 2
DF_2 = 3
DF_3 = 8

YLIM1 = 3**2 * DF_1 # z1**2 + ..(degree 1).. + zd**2
YLIM2 = 3**2 * DF_2 # z1**2 + ..(degree 2).. + zd**2
YLIM3 = 3**2 * DF_3 # z1**2 + ..(degree 3).. + zd**2

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax00 = plt.subplot2grid((9, 6), (0, 0), rowspan=3, colspan=2)
ax01 = plt.subplot2grid((9, 6), (0, 2), rowspan=3, colspan=2)
ax02 = plt.subplot2grid((9, 6), (0, 4), rowspan=3, colspan=2)
ax10 = plt.subplot2grid((9, 6), (3, 0), rowspan=3, colspan=2)
ax11 = plt.subplot2grid((9, 6), (3, 2), rowspan=3, colspan=2)
ax12 = plt.subplot2grid((9, 6), (3, 4), rowspan=3, colspan=2)

ax2 = plt.subplot2grid((9, 6), (6, 0), rowspan=3, colspan=6)

ax00.grid(axis='both', linestyle='--', color='0.95')
ax01.grid(axis='both', linestyle='--', color='0.95')
ax02.grid(axis='both', linestyle='--', color='0.95')

ax10.grid(axis='both', linestyle='--', color='0.95')
ax10.set_xlim(0, X_RANGE)
ax10.set_ylim(0, YLIM1)
ax11.grid(axis='both', linestyle='--', color='0.95')
ax11.set_xlim(0, X_RANGE)
ax11.set_ylim(0, YLIM2)
ax12.grid(axis='both', linestyle='--', color='0.95')
ax12.set_xlim(0, X_RANGE)
ax12.set_ylim(0, YLIM3)

Z_X = np.linspace(-3, 3, 300)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
Z_PDF = stats.norm.pdf(Z_X, loc=0, scale=1)

ax00.plot(Z_X, Z_PDF, alpha=1.0, color='black', linewidth=3.0)
ax01.plot(Z_X, Z_PDF, alpha=1.0, color='black', linewidth=3.0)
ax02.plot(Z_X, Z_PDF, alpha=1.0, color='black', linewidth=3.0)

dots00, = ax00.plot([], [], 'ro', alpha=1.0)
lines00 = ax00.vlines([], [], [], color='r', alpha=1.0)
dots01, = ax01.plot([], [], 'go', alpha=1.0)
lines01 = ax01.vlines([], [], [], color='g', alpha=1.0)
dots02, = ax02.plot([], [], 'bo', alpha=1.0)
lines02 = ax02.vlines([], [], [], color='b', alpha=1.0)

line10, = ax10.plot([], color='r', label=f'p = {DF_1}')
line11, = ax11.plot([], color='g', label=f'p = {DF_2}')
line12, = ax12.plot([], color='b', label=f'p = {DF_3}')

chis_0 = []
chis_1 = []
chis_2 = []

X_1 = np.linspace(0, YLIM1, 100)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
PDF_1 = stats.chi2.pdf(X_1, df=DF_1)

X_2 = np.linspace(0, YLIM2, 100)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
PDF_2 = stats.chi2.pdf(X_2, df=DF_2)

X_3 = np.linspace(0, YLIM3, 100)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
PDF_3 = stats.chi2.pdf(X_3, df=DF_3)

# Generate normally distributed random arrays:
# 0: [0 .. 999]
# 1: [0 .. 999]
# ..
# 7: [0 .. 999]
#
# TRANSPOSE for each iteration of for-loop below:
# 0: [0 .. 7]
# 1: [0 .. 7]
# ..
# 999: [0 .. 7]
normal = [stats.norm.rvs(size=X_RANGE) for i in range(max(DF_1, DF_2, DF_3))]
normal = np.array(normal).transpose().tolist()

for i in range(X_RANGE):
    ########### Degrees of freedom 1
    x00 = normal[i][0: DF_1]
    y00 = stats.norm.pdf(x00, loc=0, scale=1)
    chi_squared_0 = np.sum(np.square(x00))

    dots00.set_data(x00, y00)
    lines00.remove()
    lines00 = ax00.vlines(x00, [0] * DF_1, y00, color='r', alpha=1.0)

    ########### Degrees of freedom 2
    x01 = normal[i][0: DF_2]
    y01 = stats.norm.pdf(x01, loc=0, scale=1)
    chi_squared_1 = np.sum(np.square(x01))

    dots01.set_data(x01, y01)
    lines01.remove()
    lines01 = ax01.vlines(x01, [0] * DF_2, y01, color='g', alpha=1.0)

    ########### Degrees of freedom 3
    x02 = normal[i][0: DF_3]
    y02 = stats.norm.pdf(x02, loc=0, scale=1)
    chi_squared_2 = np.sum(np.square(x02))

    dots02.set_data(x02, y02)
    lines02.remove()
    lines02 = ax02.vlines(x02, [0] * DF_3, y02, color='b', alpha=1.0)

    ###########
    
    chis_0.append(chi_squared_0)
    chis_1.append(chi_squared_1)
    chis_2.append(chi_squared_2)

    line10.set_data([i for i in range(len(chis_0))], chis_0)
    line11.set_data([i for i in range(len(chis_1))], chis_1)
    line12.set_data([i for i in range(len(chis_2))], chis_2)

    ###########

    if (i < 100) or (i == X_RANGE - 1):

        bins0 = int(max(chis_0) - min(chis_0)) + 1
        bins1 = int(max(chis_1) - min(chis_1)) + 1
        bins2 = int(max(chis_2) - min(chis_2)) + 1

        ax2.cla()
        ax2.hist(chis_0, bins = bins0, density=True, rwidth=0.8, alpha=0.4, color='r', label=f'degrees of freedom {DF_1}')
        ax2.hist(chis_1, bins = bins1, density=True, rwidth=0.6, alpha=0.6, color='g', label=f'degrees of freedom {DF_2}')
        ax2.hist(chis_2, bins = bins2, density=True, rwidth=0.4, alpha=0.8, color='b', label=f'degrees of freedom {DF_3}')
        ax2.plot(X_1, PDF_1, alpha=1.0, color='r', linewidth=2.0)
        ax2.plot(X_2, PDF_2, alpha=1.0, color='g', linewidth=2.0)
        ax2.plot(X_3, PDF_3, alpha=1.0, color='b', linewidth=2.0)

        ax2.grid(axis='both', linestyle='--', color='0.95')
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.set_xlim(0, max(max(chis_0), max(chis_1), max(chis_2)))

        plt.tight_layout()
        plt.pause(0.05)

plt.show()