from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math
import random

X_RANGE = 1000

K_1 = 1 # 1st Pair
K_2 = 9 # 1st Pair
K_3 = 2 # 2nd Pair
K_4 = 4 # 2nd Pair
K_5 = 0.5 # 3rd Pair
K_6 = 0.5 # 3rd Pair

YLIM1 = 1 # approximation (see the graphs)
YLIM2 = 1 # approximation (see the graphs)
YLIM3 = 1 # approximation (see the graphs)

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax00 = plt.subplot2grid((9, 6), (0, 0), rowspan=3, colspan=2)
ax01 = plt.subplot2grid((9, 6), (0, 2), rowspan=3, colspan=2)
ax02 = plt.subplot2grid((9, 6), (0, 4), rowspan=3, colspan=2)
ax10 = plt.subplot2grid((9, 6), (3, 0), rowspan=3, colspan=2)
ax11 = plt.subplot2grid((9, 6), (3, 2), rowspan=3, colspan=2)
ax12 = plt.subplot2grid((9, 6), (3, 4), rowspan=3, colspan=2)

ax20 = plt.subplot2grid((9, 6), (6, 0), rowspan=3, colspan=2)
ax21 = plt.subplot2grid((9, 6), (6, 2), rowspan=3, colspan=2)
ax22 = plt.subplot2grid((9, 6), (6, 4), rowspan=3, colspan=2)

ax00.grid(axis='both', linestyle='--', color='0.95')
ax00.set_title(f'Gamma Randoms Γ(k={K_1}, θ=1), Γ(k={K_2}, θ=1)')
ax00.set_xlim(0, 17)
ax00.set_ylim(0, 1)
ax01.grid(axis='both', linestyle='--', color='0.95')
ax01.set_title(f'Gamma Randoms Γ(k={K_3}, θ=1), Γ(k={K_4}, θ=1)')
ax01.set_xlim(0, 12)
ax01.set_ylim(0, 0.5)
ax02.grid(axis='both', linestyle='--', color='0.95')
ax02.set_title(f'Gamma Randoms Γ(k={K_5}, θ=1), Γ(k={K_6}, θ=1)')
ax02.set_xlim(0, 3)
ax02.set_ylim(0, 4)

ax10.grid(axis='both', linestyle='--', color='0.95')
ax10.set_xlim(0, X_RANGE)
ax10.set_ylim(0, YLIM1)
ax11.grid(axis='both', linestyle='--', color='0.95')
ax11.set_xlim(0, X_RANGE)
ax11.set_ylim(0, YLIM2)
ax12.grid(axis='both', linestyle='--', color='0.95')
ax12.set_xlim(0, X_RANGE)
ax12.set_ylim(0, YLIM3)

X = np.linspace(0, 20, 1000)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
GAMMA_PDF_1 = stats.gamma.pdf(X, a=K_1)
GAMMA_PDF_2 = stats.gamma.pdf(X, a=K_2)
GAMMA_PDF_3 = stats.gamma.pdf(X, a=K_3)
GAMMA_PDF_4 = stats.gamma.pdf(X, a=K_4)
GAMMA_PDF_5 = stats.gamma.pdf(X, a=K_5)
GAMMA_PDF_6 = stats.gamma.pdf(X, a=K_6)

ax00.plot(X, GAMMA_PDF_1, alpha=1.0, color='black', linewidth=2.0)
ax00.plot(X, GAMMA_PDF_2, alpha=1.0, color='black', linewidth=2.0)
ax01.plot(X, GAMMA_PDF_3, alpha=1.0, color='black', linewidth=2.0)
ax01.plot(X, GAMMA_PDF_4, alpha=1.0, color='black', linewidth=2.0)
ax02.plot(X, GAMMA_PDF_5, alpha=1.0, color='black', linewidth=2.0)
ax02.plot(X, GAMMA_PDF_6, alpha=1.0, color='black', linewidth=2.0)

dots00, = ax00.plot([], [], 'ro', alpha=1.0)
lines00 = ax00.vlines([], [], [], color='r', alpha=1.0)
dots01, = ax01.plot([], [], 'go', alpha=1.0)
lines01 = ax01.vlines([], [], [], color='g', alpha=1.0)
dots02, = ax02.plot([], [], 'bo', alpha=1.0)
lines02 = ax02.vlines([], [], [], color='b', alpha=1.0)

line10, = ax10.plot([], color='r', label=f'p = {K_1}')
text10 = ax10.text(20, 4 * YLIM1 / 5, f'')
line11, = ax11.plot([], color='g', label=f'p = {K_2}')
text11 = ax11.text(20, 4 * YLIM2 / 5, f'')
line12, = ax12.plot([], color='b', label=f'p = {K_3}')
text12 = ax12.text(20, 4 * YLIM3 / 5, f'')

betas_0 = []
betas_1 = []
betas_2 = []

X_1 = np.linspace(0, 1, 100)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
PDF_1 = stats.beta.pdf(X_1, a=K_1, b=K_2)

X_2 = np.linspace(0, 1, 100)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
PDF_2 = stats.beta.pdf(X_2, a=K_3, b=K_4)

X_3 = np.linspace(0, 1, 100)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
PDF_3 = stats.beta.pdf(X_3, a=K_5, b=K_6)

# Generate gamma distributed random arrays:
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

gamma = []
gamma.append(stats.gamma.rvs(a=K_1, size=X_RANGE))
gamma.append(stats.gamma.rvs(a=K_2, size=X_RANGE))
gamma.append(stats.gamma.rvs(a=K_3, size=X_RANGE))
gamma.append(stats.gamma.rvs(a=K_4, size=X_RANGE))
gamma.append(stats.gamma.rvs(a=K_5, size=X_RANGE))
gamma.append(stats.gamma.rvs(a=K_6, size=X_RANGE))
gamma = np.array(gamma).transpose().tolist()

for i in range(X_RANGE):
    ########### 
    x00 = [gamma[i][0], gamma[i][1]]
    y00 = [
        stats.gamma.pdf(x00[0], a=K_1, loc=0, scale=1),
        stats.gamma.pdf(x00[1], a=K_2, loc=0, scale=1)
    ]
    beta_0 = (x00[0]) / (x00[0] + x00[1])

    dots00.set_data(x00, y00)
    lines00.remove()
    lines00 = ax00.vlines(x00, [0] * 2, y00, color='r', alpha=1.0)

    ########### 
    x01 = [gamma[i][2], gamma[i][3]]
    y01 = [
        stats.gamma.pdf(x01[0], a=K_3, loc=0, scale=1),
        stats.gamma.pdf(x01[1], a=K_4, loc=0, scale=1)
    ]
    beta_1 = (x01[0]) / (x01[0] + x01[1])

    dots01.set_data(x01, y01)
    lines01.remove()
    lines01 = ax01.vlines(x01, [0] * 2, y01, color='g', alpha=1.0)

    ########### 
    x02 = [gamma[i][4], gamma[i][5]]
    y02 = [
        stats.gamma.pdf(x02[0], a=K_5, loc=0, scale=1),
        stats.gamma.pdf(x02[1], a=K_6, loc=0, scale=1)
    ]
    beta_2 = (x02[0]) / (x02[0] + x02[1])

    dots02.set_data(x02, y02)
    lines02.remove()
    lines02 = ax02.vlines(x02, [0] * 2, y02, color='b', alpha=1.0)

    ###########
    
    betas_0.append(beta_0)
    betas_1.append(beta_1)
    betas_2.append(beta_2)

    line10.set_data(list(range(len(betas_0))), betas_0)
    line11.set_data(list(range(len(betas_1))), betas_1)
    line12.set_data(list(range(len(betas_2))), betas_2)

    text10.set_text(f'X / (X + Y) = {beta_0}')
    text11.set_text(f'X / (X + Y) = {beta_1}')
    text12.set_text(f'X / (X + Y) = {beta_2}')

    ###########

    if (i < 100) or (i == X_RANGE - 1):

        bins0 = 20 # 6 * int(max(betas_0) - min(betas_0)) + 1
        bins1 = 20 # 8 * int(max(betas_1) - min(betas_1)) + 1
        bins2 = 20 # 10 * int(max(betas_2) - min(betas_2)) + 1

        ax20.cla()
        ax21.cla()
        ax22.cla()
        ax20.hist(betas_0, bins = bins0, density=True, rwidth=0.8, alpha=0.4, color='r', label=f'α={K_1},β={K_2}')
        ax21.hist(betas_1, bins = bins1, density=True, rwidth=0.8, alpha=0.4, color='g', label=f'α={K_3},β={K_4}')
        ax22.hist(betas_2, bins = bins2, density=True, rwidth=0.8, alpha=0.4, color='b', label=f'α={K_5},β={K_6}')
        ax20.plot(X_1, PDF_1, alpha=1.0, color='r', linewidth=2.0)
        ax21.plot(X_2, PDF_2, alpha=1.0, color='g', linewidth=2.0)
        ax22.plot(X_3, PDF_3, alpha=1.0, color='b', linewidth=2.0)

        ax20.grid(axis='both', linestyle='--', color='0.95')
        # ax20.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        # ax20.set_xlim(0, 2)

        ax21.grid(axis='both', linestyle='--', color='0.95')
        # ax21.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax21.set_xlim(0, 5)

        ax22.grid(axis='both', linestyle='--', color='0.95')
        # ax22.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        # ax22.set_xlim(0, 4)

        ax20.text(0.1, 5, f'Beta(α={K_1},β={K_2})')
        ax21.text(0.6, 1, f'Beta(α={K_3},β={K_4})')
        ax22.text(0.5, 1, f'Beta(α={K_5},β={K_6})')

        ax20.legend(loc="upper right")
        ax21.legend(loc="upper right")
        ax22.legend(loc="upper right")

        plt.tight_layout()
        plt.pause(0.05)

plt.show()