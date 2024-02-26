from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math
import random

X_RANGE = 1000 # up to 1000 days
Y_RANGE = 1000 # up to 1000 events a day

P1 = 0.001
P2 = 0.01
P3 = 0.03

YLIM1 = 10 * Y_RANGE * P1
YLIM2 = 4 * Y_RANGE * P2
YLIM3 = 2 * Y_RANGE * P3

distr_1 = pd.DataFrame(columns = ['count'])
distr_2 = pd.DataFrame(columns = ['count'])
distr_3 = pd.DataFrame(columns = ['count'])

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax1 = plt.subplot2grid((10, 6), (0, 0), rowspan=5, colspan=2)
ax2 = plt.subplot2grid((10, 6), (0, 2), rowspan=5, colspan=2)
ax3 = plt.subplot2grid((10, 6), (0, 4), rowspan=5, colspan=2)

ax1.grid(axis='both', linestyle='--', color='0.95')
ax1.set_xlim(0, X_RANGE) 
ax1.set_ylim(0, YLIM1) 
ax1.set_xlabel('sample\'s number')
ax1.set_ylabel('count of 1')
ax1.set_title(f'Number of successes (p = {P1})')

ax2.grid(axis='both', linestyle='--', color='0.95')
ax2.set_xlim(0, X_RANGE) 
ax2.set_ylim(0, YLIM2) 
ax2.set_xlabel('sample\'s number')
ax2.set_ylabel('count of 1')
ax2.set_title(f'Number of successes (p = {P2})')

ax3.grid(axis='both', linestyle='--', color='0.95')
ax3.set_xlim(0, X_RANGE) 
ax3.set_ylim(0, YLIM3)
ax3.set_xlabel('sample\'s number')
ax3.set_ylabel('count of 1')
ax3.set_title(f'Number of successes (p = {P3})')

# https://stackoverflow.com/questions/42435446/how-to-put-text-outside-of-plots
text_1 = ax1.text(50, YLIM1 * 0.9, '', color='r', fontweight='bold') # , transform=plt.gcf().transFigure
text_2 = ax2.text(50, YLIM2 * 0.9, '', color='g', fontweight='bold') # , transform=plt.gcf().transFigure
text_3 = ax3.text(50, YLIM3 * 0.9, '', color='b', fontweight='bold') # , transform=plt.gcf().transFigure

line_1, = ax1.plot([], color='r', label='p=0.25')
line_2, = ax2.plot([], color='g', label='p=0.5')
line_3, = ax3.plot([], color='b', label='p=0.75')

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax4 = plt.subplot2grid((10, 6), (5, 0), rowspan=5, colspan=6)

ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")

X = np.linspace(0, Y_RANGE, Y_RANGE + 1)

for i in range(X_RANGE): 
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
    # The Poisson distribution is the limit of the binomial distribution for large N.
    sample_1 = [1 if r < P1 else 0 for r in [random.random() for i in range(Y_RANGE)]]
    sample_2 = [1 if r < P2 else 0 for r in [random.random() for i in range(Y_RANGE)]]
    sample_3 = [1 if r < P3 else 0 for r in [random.random() for i in range(Y_RANGE)]]

    distr_1.loc[i] = sample_1.count(1)
    distr_2.loc[i] = sample_2.count(1)
    distr_3.loc[i] = sample_3.count(1)

    if (i < 100) or (i == X_RANGE - 1):
        text_1.set_text(f'{i}: Number of "1": {sample_1.count(1)} out of 1000')
        text_2.set_text(f'{i}: Number of "1": {sample_2.count(1)} out of 1000')
        text_3.set_text(f'{i}: Number of "1": {sample_3.count(1)} out of 1000')

        line_1.set_data(distr_1.index.values, distr_1['count'].values)
        line_2.set_data(distr_2.index.values, distr_2['count'].values)
        line_3.set_data(distr_3.index.values, distr_3['count'].values)

        mean_1 = distr_1["count"].mean()
        mean_2 = distr_2["count"].mean()
        mean_3 = distr_3["count"].mean()

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html
        PMF_1 = stats.poisson.pmf(X, mean_1)
        PMF_2 = stats.poisson.pmf(X, mean_2)
        PMF_3 = stats.poisson.pmf(X, mean_3)

        bins_1 = distr_1["count"].max() - distr_1["count"].min()
        bins_2 = distr_2["count"].max() - distr_2["count"].min()
        bins_3 = distr_3["count"].max() - distr_3["count"].min()

        ax4.cla()
        ax4.hist(distr_1.values, bins = bins_1 if bins_1 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='r', label='sample 20 0.1')
        ax4.hist(distr_2.values, bins = bins_2 if bins_2 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='g', label='sample 20 0.5')
        ax4.hist(distr_3.values, bins = bins_3 if bins_3 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='b', label='sample 20 0.8')
        ax4.plot(X, PMF_1, marker='o', linestyle='dashed', alpha=1.0, color='r', linewidth=2.0)
        ax4.plot(X, PMF_2, marker='o', linestyle='dashed', alpha=1.0, color='g', linewidth=2.0)
        ax4.plot(X, PMF_3, marker='o', linestyle='dashed', alpha=1.0, color='b', linewidth=2.0)
    
        ax4.grid(axis='both', linestyle='--', color='0.95')
        ax4.xaxis.set_major_locator(ticker.MultipleLocator(Y_RANGE / 200))
        ax4.set_xlim(0, Y_RANGE / 20) 
        # ax3.set_ylim(0, 1)
        # ax3.set_xlabel('')
        # ax3.set_ylabel('')
        # ax3.set_title('')
        ax4.legend(loc="upper right")

        ax4.text(2, 0.2, f'Pois({mean_1:.4f})')
        ax4.text(10, 0.13, f'Pois({mean_2:.4f})')
        ax4.text(30, 0.08, f'Pois({mean_3:.4f})')

    (i < 100) and (i % 20 == 0) and plt.tight_layout()

    # pause the plot for 0.01s before next point is shown 
    # plt.pause(0.5 if i < 100 else 0.0001) 
    (i < 100) and plt.pause(0.05)

count_1 = pd.cut(distr_1["count"], distr_1["count"].max() - distr_1["count"].min()).value_counts()
count_2 = pd.cut(distr_2["count"], distr_2["count"].max() - distr_2["count"].min()).value_counts()
count_3 = pd.cut(distr_3["count"], distr_3["count"].max() - distr_3["count"].min()).value_counts()

print(count_1)
print(count_2)
print(count_3)

plt.show()