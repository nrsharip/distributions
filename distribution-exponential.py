from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math
import random

X_RANGE = 100 # up to 100 days
Y_RANGE = 100 # up to 100 events a day

P1 = 0.1
P2 = 0.3
P3 = 0.5

YLIM1 = 100 / 1
YLIM2 = 100 / 3
YLIM3 = 100 / 5

distr_1 = pd.DataFrame(columns = ['time'])
distr_2 = pd.DataFrame(columns = ['time'])
distr_3 = pd.DataFrame(columns = ['time'])

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax1 = plt.subplot2grid((10, 6), (0, 0), rowspan=5, colspan=2)
ax2 = plt.subplot2grid((10, 6), (0, 2), rowspan=5, colspan=2)
ax3 = plt.subplot2grid((10, 6), (0, 4), rowspan=5, colspan=2)

ax1.grid(axis='both', linestyle='--', color='0.95')
ax1.set_xlim(0, X_RANGE**2 * P1) 
ax1.set_ylim(0, YLIM1) 
ax1.set_xlabel('number of measures')
ax1.set_ylabel('time between successes')
ax1.set_title(f'Time between the successes (p = {P1})')

ax2.grid(axis='both', linestyle='--', color='0.95')
ax2.set_xlim(0, X_RANGE**2 * P2) 
ax2.set_ylim(0, YLIM2) 
ax2.set_xlabel('number of measures')
ax2.set_ylabel('time between successes')
ax2.set_title(f'Time between the successes (p = {P2})')

ax3.grid(axis='both', linestyle='--', color='0.95')
ax3.set_xlim(0, X_RANGE**2 * P3) 
ax3.set_ylim(0, YLIM3)
ax3.set_xlabel('number of measures')
ax3.set_ylabel('time between successes')
ax3.set_title(f'Time between the successes (p = {P3})')

# https://stackoverflow.com/questions/42435446/how-to-put-text-outside-of-plots
text_1 = ax1.text(50, YLIM1 * 0.9, '', color='r', fontweight='bold') # , transform=plt.gcf().transFigure
text_2 = ax2.text(50, YLIM2 * 0.9, '', color='g', fontweight='bold') # , transform=plt.gcf().transFigure
text_3 = ax3.text(50, YLIM3 * 0.9, '', color='b', fontweight='bold') # , transform=plt.gcf().transFigure

line_1, = ax1.plot([], color='r', label=f'p = {P1}')
line_2, = ax2.plot([], color='g', label=f'p = {P2}')
line_3, = ax3.plot([], color='b', label=f'p = {P3}')

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax4 = plt.subplot2grid((10, 6), (5, 0), rowspan=5, colspan=6)

ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")

X = np.linspace(0, Y_RANGE, 1000)

# 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1
# Number of successes: 4
# Times between successes: [3, 8, 4] 
def calc_times(sample: list, df: pd.DataFrame):
    time = 0
    for event in sample:
        if event == 1:
            df = pd.concat([df, pd.DataFrame({ 'time': [time] })], ignore_index=True)
            time = 0
        elif event == 0:
            time += 1
    return df

for i in range(X_RANGE): 
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
    # The Poisson distribution is the limit of the binomial distribution for large N.
    sample_1 = [1 if r < P1 else 0 for r in [random.random() for i in range(Y_RANGE)]]
    sample_2 = [1 if r < P2 else 0 for r in [random.random() for i in range(Y_RANGE)]]
    sample_3 = [1 if r < P3 else 0 for r in [random.random() for i in range(Y_RANGE)]]

    distr_1 = calc_times(sample_1, distr_1)
    distr_2 = calc_times(sample_2, distr_2)
    distr_3 = calc_times(sample_3, distr_3)

    if (i < 100) or (i == X_RANGE - 1):
        text_1.set_text(f'{i}')
        text_2.set_text(f'{i}')
        text_3.set_text(f'{i}')

        line_1.set_data(distr_1.index.values, distr_1['time'].values)
        line_2.set_data(distr_2.index.values, distr_2['time'].values)
        line_3.set_data(distr_3.index.values, distr_3['time'].values)

        mean_1 = distr_1["time"].mean()
        mean_2 = distr_2["time"].mean()
        mean_3 = distr_3["time"].mean()

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
        PDF_1 = stats.expon.pdf(X, scale = mean_1)
        PDF_2 = stats.expon.pdf(X, scale = mean_2)
        PDF_3 = stats.expon.pdf(X, scale = mean_3)

        bins_1 = distr_1["time"].max() - distr_1["time"].min()
        bins_2 = distr_2["time"].max() - distr_2["time"].min()
        bins_3 = distr_3["time"].max() - distr_3["time"].min()

        ax4.cla()
        ax4.hist(distr_1.values, bins = bins_1 if bins_1 > 0 else 1, density=True, rwidth=0.4, alpha=0.8, color='r', label=f'sample {Y_RANGE} {P1}')
        ax4.hist(distr_2.values, bins = bins_2 if bins_2 > 0 else 1, density=True, rwidth=0.6, alpha=0.6, color='g', label=f'sample {Y_RANGE} {P2}')
        ax4.hist(distr_3.values, bins = bins_3 if bins_3 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='b', label=f'sample {Y_RANGE} {P3}')
        ax4.plot(X, PDF_1, alpha=1.0, color='r', linewidth=2.0)
        ax4.plot(X, PDF_2, alpha=1.0, color='g', linewidth=2.0)
        ax4.plot(X, PDF_3, alpha=1.0, color='b', linewidth=2.0)
    
        ax4.grid(axis='both', linestyle='--', color='0.95')
        ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax4.set_xlim(0, Y_RANGE / 10)
        # ax3.set_ylim(0, 1)
        # ax3.set_xlabel('')
        # ax3.set_ylabel('')
        # ax3.set_title('')
        ax4.legend(loc="upper right")

        ax4.text(0.2, 0.18, f'Exp(1/λ = {mean_1:.4f})')
        ax4.text(0.2, 0.45, f'Exp(1/λ = {mean_2:.4f})')
        ax4.text(0.2, 0.9, f'Exp(1/λ = {mean_3:.4f})')

    (i < 100) and (i % 20 == 0) and plt.tight_layout()

    # pause the plot for 0.01s before next point is shown 
    # plt.pause(0.5 if i < 100 else 0.0001) 
    (i < 100) and plt.pause(0.05)

count_1 = pd.cut(distr_1["time"], distr_1["time"].max() - distr_1["time"].min()).value_counts()
count_2 = pd.cut(distr_2["time"], distr_2["time"].max() - distr_2["time"].min()).value_counts()
count_3 = pd.cut(distr_3["time"], distr_3["time"].max() - distr_3["time"].min()).value_counts()

print(count_1)
print(count_2)
print(count_3)

plt.show()