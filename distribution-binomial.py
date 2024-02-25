from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math
import random

X_RANGE = 1000
Y_RANGE = 20

distr025 = pd.DataFrame(columns = ['count'])
distr05 = pd.DataFrame(columns = ['count'])
distr075 = pd.DataFrame(columns = ['count'])

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax0 = plt.subplot2grid((10, 6), (0, 0), rowspan=5, colspan=2)
ax1 = plt.subplot2grid((10, 6), (0, 2), rowspan=5, colspan=2)
ax2 = plt.subplot2grid((10, 6), (0, 4), rowspan=5, colspan=2)

ax0.grid(axis='both', linestyle='--', color='0.95')
ax0.set_xlim(0, X_RANGE) 
ax0.set_ylim(0, Y_RANGE) 

ax1.grid(axis='both', linestyle='--', color='0.95')
ax1.set_xlim(0, X_RANGE) 
ax1.set_ylim(0, Y_RANGE) 

ax2.grid(axis='both', linestyle='--', color='0.95')
ax2.set_xlim(0, X_RANGE) 
ax2.set_ylim(0, Y_RANGE)

# https://stackoverflow.com/questions/42435446/how-to-put-text-outside-of-plots
text025 = ax0.text(50, 19, '') # , transform=plt.gcf().transFigure
text025_1 = ax0.text(50, 18, '', color='r', fontweight='bold') # , transform=plt.gcf().transFigure
text05 = ax1.text(50, 19, '') # , transform=plt.gcf().transFigure
text05_1 = ax1.text(50, 18, '', color='g', fontweight='bold') # , transform=plt.gcf().transFigure
text075 = ax2.text(50, 2, '') # , transform=plt.gcf().transFigure
text075_1 = ax2.text(50, 1, '', color='b', fontweight='bold') # , transform=plt.gcf().transFigure

line025, = ax0.plot([3, 4, 5], color='r', label='p=0.25')
line05, = ax1.plot([3, 4, 5], color='g', label='p=0.5')
line075, = ax2.plot([3, 4, 5], color='b', label='p=0.75')

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax4 = plt.subplot2grid((10, 6), (5, 0), rowspan=5, colspan=6)

X_025 = np.linspace(0, Y_RANGE, Y_RANGE + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
PPF_025 = stats.binom.pmf(X_025, Y_RANGE, 0.1)

X_05 = np.linspace(0, Y_RANGE, Y_RANGE + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
PPF_05 = stats.binom.pmf(X_05, Y_RANGE, 0.5)

X_075 = np.linspace(0, Y_RANGE, Y_RANGE + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
PPF_075 = stats.binom.pmf(X_075, Y_RANGE, 0.8)

for i in range(X_RANGE): 
    p = 0.1
    sample025 = [1 if r < p else 0 for r in [random.random() for i in range(Y_RANGE)]]
    # print(i, sample025.count(1), sample025)
    p = 0.5
    sample05 = [1 if r < p else 0 for r in [random.random() for i in range(Y_RANGE)]]
    # print(i, sample05.count(1), sample05)
    p = 0.8
    sample075 = [1 if r < p else 0 for r in [random.random() for i in range(Y_RANGE)]]
    # print(i, sample075.count(1), sample075)

    distr025.loc[i] = sample025.count(1)
    distr05.loc[i] = sample05.count(1)
    distr075.loc[i] = sample075.count(1)

    if (i < 100) or (i == X_RANGE - 1):
        text025.set_text(f'{i}: {sample025}')
        text025_1.set_text(f'Number of "1": {sample025.count(1)}')
        text05.set_text(f'{i}: {sample05}')
        text05_1.set_text(f'Number of "1": {sample05.count(1)}')
        text075.set_text(f'{i}: {sample075}')
        text075_1.set_text(f'Number of "1": {sample075.count(1)}')

        line025.set_data(distr025.index.values, distr025['count'].values)
        line05.set_data(distr05.index.values, distr05['count'].values)
        line075.set_data(distr075.index.values, distr075['count'].values)

        bins025 = distr025["count"].max() - distr025["count"].min()
        bins05 = distr05["count"].max() - distr05["count"].min()
        bins075 = distr075["count"].max() - distr075["count"].min()

        ax4.cla()
        ax4.hist(distr025.values, bins = bins025 if bins025 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='r', label='sample 10 0.25')
        ax4.hist(distr05.values, bins = bins05 if bins05 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='g', label='sample 10 0.5')
        ax4.hist(distr075.values, bins = bins075 if bins075 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='b', label='sample 10 0.75')
        ax4.plot(X_025, PPF_025, alpha=1.0, color='r', linewidth=2.0)
        ax4.plot(X_05, PPF_05, alpha=1.0, color='g', linewidth=2.0)
        ax4.plot(X_075, PPF_075, alpha=1.0, color='b', linewidth=2.0)
    
        ax4.grid(axis='both', linestyle='--', color='0.95')
        ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))

    (i < 100) and (i % 20 == 0) and plt.tight_layout()

    # pause the plot for 0.01s before next point is shown 
    # plt.pause(0.5 if i < 100 else 0.0001) 
    (i < 100) and plt.pause(0.05)

count025 = pd.cut(distr025["count"], distr025["count"].max() - distr025["count"].min()).value_counts()
count05 = pd.cut(distr05["count"], distr05["count"].max() - distr05["count"].min()).value_counts()
count075 = pd.cut(distr075["count"], distr075["count"].max() - distr075["count"].min()).value_counts()

print(count025)
print(count05)
print(count075)

plt.show()