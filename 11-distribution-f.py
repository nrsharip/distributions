from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math
import random

X_RANGE = 1000

DF_1 = 1 # 1st Pair
DF_2 = 9 # 1st Pair
DF_3 = 2 # 2nd Pair
DF_4 = 4 # 2nd Pair
DF_5 = 3 # 3rd Pair
DF_6 = 5 # 3rd Pair

YLIM1 = 30 # approximation (see the graphs)
YLIM2 = 60 # approximation (see the graphs)
YLIM3 = 30 # approximation (see the graphs)

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
ax00.set_title(f'Chi-squared Randoms (dfn = {DF_1}, dfd = {DF_2})')
ax00.set_xlim(0, 20)
ax00.set_ylim(0, 0.5)
ax01.grid(axis='both', linestyle='--', color='0.95')
ax01.set_title(f'Chi-squared Randoms (dfn = {DF_3}, dfd = {DF_4})')
ax01.set_xlim(0, 15)
ax01.set_ylim(0, 0.5)
ax02.grid(axis='both', linestyle='--', color='0.95')
ax02.set_title(f'Chi-squared Randoms (dfn = {DF_5}, dfd = {DF_6})')
ax02.set_xlim(0, 15)
ax02.set_ylim(0, 0.25)

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
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
CHI_PDF_1 = stats.chi2.pdf(X, df=DF_1)
CHI_PDF_2 = stats.chi2.pdf(X, df=DF_2)
CHI_PDF_3 = stats.chi2.pdf(X, df=DF_3)
CHI_PDF_4 = stats.chi2.pdf(X, df=DF_4)
CHI_PDF_5 = stats.chi2.pdf(X, df=DF_5)
CHI_PDF_6 = stats.chi2.pdf(X, df=DF_6)

ax00.plot(X, CHI_PDF_1, alpha=1.0, color='black', linewidth=2.0)
ax00.plot(X, CHI_PDF_2, alpha=1.0, color='black', linewidth=2.0)
ax01.plot(X, CHI_PDF_3, alpha=1.0, color='black', linewidth=2.0)
ax01.plot(X, CHI_PDF_4, alpha=1.0, color='black', linewidth=2.0)
ax02.plot(X, CHI_PDF_5, alpha=1.0, color='black', linewidth=2.0)
ax02.plot(X, CHI_PDF_6, alpha=1.0, color='black', linewidth=2.0)

dots00, = ax00.plot([], [], 'ro', alpha=1.0)
lines00 = ax00.vlines([], [], [], color='r', alpha=1.0)
dots01, = ax01.plot([], [], 'go', alpha=1.0)
lines01 = ax01.vlines([], [], [], color='g', alpha=1.0)
dots02, = ax02.plot([], [], 'bo', alpha=1.0)
lines02 = ax02.vlines([], [], [], color='b', alpha=1.0)

line10, = ax10.plot([], color='r', label=f'p = {DF_1}')
text10 = ax10.text(20, 4 * YLIM1 / 5, f'')
line11, = ax11.plot([], color='g', label=f'p = {DF_2}')
text11 = ax11.text(20, 4 * YLIM2 / 5, f'')
line12, = ax12.plot([], color='b', label=f'p = {DF_3}')
text12 = ax12.text(20, 4 * YLIM3 / 5, f'')

fs_0 = []
fs_1 = []
fs_2 = []

X_1 = np.linspace(0, 5, 100)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html
PDF_1 = stats.f.pdf(X_1, dfn=DF_1, dfd=DF_2)

X_2 = np.linspace(0, 10, 100)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html
PDF_2 = stats.f.pdf(X_2, dfn=DF_3, dfd=DF_4)

X_3 = np.linspace(0, 5, 100)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html
PDF_3 = stats.f.pdf(X_3, dfn=DF_5, dfd=DF_6)

# Generate chi-squared distributed random arrays:
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

chi2 = []
chi2.append(stats.chi2.rvs(df=DF_1, size=X_RANGE))
chi2.append(stats.chi2.rvs(df=DF_2, size=X_RANGE))
chi2.append(stats.chi2.rvs(df=DF_3, size=X_RANGE))
chi2.append(stats.chi2.rvs(df=DF_4, size=X_RANGE))
chi2.append(stats.chi2.rvs(df=DF_5, size=X_RANGE))
chi2.append(stats.chi2.rvs(df=DF_6, size=X_RANGE))
chi2 = np.array(chi2).transpose().tolist()

for i in range(X_RANGE):
    ########### 
    x00 = [chi2[i][0], chi2[i][1]]
    y00 = [
        stats.chi2.pdf(x00[0], df=DF_1, loc=0, scale=1),
        stats.chi2.pdf(x00[1], df=DF_2, loc=0, scale=1)
    ]
    f_0 = (x00[0] / DF_1) / (x00[1] / DF_2)

    dots00.set_data(x00, y00)
    lines00.remove()
    lines00 = ax00.vlines(x00, [0] * 2, y00, color='r', alpha=1.0)

    ########### 
    x01 = [chi2[i][2], chi2[i][3]]
    y01 = [
        stats.chi2.pdf(x01[0], df=DF_3, loc=0, scale=1),
        stats.chi2.pdf(x01[1], df=DF_4, loc=0, scale=1)
    ]
    f_1 = (x01[0] / DF_3) / (x01[1] / DF_4)

    dots01.set_data(x01, y01)
    lines01.remove()
    lines01 = ax01.vlines(x01, [0] * 2, y01, color='g', alpha=1.0)

    ########### 
    x02 = [chi2[i][4], chi2[i][5]]
    y02 = [
        stats.chi2.pdf(x02[0], df=DF_5, loc=0, scale=1),
        stats.chi2.pdf(x02[1], df=DF_6, loc=0, scale=1)
    ]
    f_2 = (x02[0] / DF_5) / (x02[1] / DF_6)

    dots02.set_data(x02, y02)
    lines02.remove()
    lines02 = ax02.vlines(x02, [0] * 2, y02, color='b', alpha=1.0)

    ###########
    
    fs_0.append(f_0)
    fs_1.append(f_1)
    fs_2.append(f_2)

    line10.set_data([i for i in range(len(fs_0))], fs_0)
    line11.set_data([i for i in range(len(fs_1))], fs_1)
    line12.set_data([i for i in range(len(fs_2))], fs_2)

    text10.set_text(f'(S1/d1)/(S2/d2) = {f_0}')
    text11.set_text(f'(S1/d1)/(S2/d2) = {f_1}')
    text12.set_text(f'(S1/d1)/(S2/d2) = {f_2}')

    ###########

    if (i < 100) or (i == X_RANGE - 1):

        bins0 = 16 * int(max(fs_0) - min(fs_0)) + 1
        bins1 = 8 * int(max(fs_1) - min(fs_1)) + 1
        bins2 = 10 * int(max(fs_2) - min(fs_2)) + 1

        ax20.cla()
        ax21.cla()
        ax22.cla()
        ax20.hist(fs_0, bins = bins0, density=True, rwidth=0.8, alpha=0.4, color='r', label=f'dfn = {DF_1}, dfd = {DF_2}')
        ax21.hist(fs_1, bins = bins1, density=True, rwidth=0.8, alpha=0.4, color='g', label=f'dfn = {DF_3}, dfd = {DF_4}')
        ax22.hist(fs_2, bins = bins2, density=True, rwidth=0.8, alpha=0.4, color='b', label=f'dfn = {DF_5}, dfd = {DF_6}')
        ax20.plot(X_1, PDF_1, alpha=1.0, color='r', linewidth=2.0)
        ax21.plot(X_2, PDF_2, alpha=1.0, color='g', linewidth=2.0)
        ax22.plot(X_3, PDF_3, alpha=1.0, color='b', linewidth=2.0)

        ax20.grid(axis='both', linestyle='--', color='0.95')
        ax20.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax20.set_xlim(0, 2)

        ax21.grid(axis='both', linestyle='--', color='0.95')
        ax21.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax21.set_xlim(0, 5)

        ax22.grid(axis='both', linestyle='--', color='0.95')
        ax22.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax22.set_xlim(0, 4)

        ax20.text(0.2, 1, f'F(d1={DF_1},d2={DF_2})')
        ax21.text(1, 0.35, f'F(d1={DF_3},d2={DF_4})')
        ax22.text(1, 0.4, f'F(d1={DF_5},d2={DF_6})')

        ax20.legend(loc="upper right")
        ax21.legend(loc="upper right")
        ax22.legend(loc="upper right")

        plt.tight_layout()
        plt.pause(0.05)

plt.show()