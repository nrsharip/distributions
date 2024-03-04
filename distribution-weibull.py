from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math
import random

X_RANGE = 1000 # up to 100 days
Y_RANGE = 1000 # up to 100 events a day

BETA_1 = 3.5
BETA_2 = 1.5 
BETA_3 = 0.9 

K_1 = 1/BETA_1
K_2 = 1/BETA_2
K_3 = 1/BETA_3

LAMBDA_1 = 1 # mean of successes, 1 success in average per the given time range
LAMBDA_2 = 2 # mean of successes, 2 successes in average per the given time range
LAMBDA_3 = 3 # mean of successes, 3 successes in average per the given time range

P1 = LAMBDA_1 / Y_RANGE # (ex. 1% out of 100, 0.1% out of 1000)
P2 = LAMBDA_2 / Y_RANGE # (ex. 2% out of 100, 0.2% out of 1000)
P3 = LAMBDA_3 / Y_RANGE # (ex. 3% out of 100, 0.3% out of 1000)

THETA_1 = (Y_RANGE - LAMBDA_1)/LAMBDA_1 # mean of time interval between the successes 
                                        # (ex. 1 successes in 1000 means ~ 999 time interval in average)
THETA_2 = (Y_RANGE - LAMBDA_2)/LAMBDA_2 # mean of time interval between the successes 
                                        # (ex. 2 successes in 1000 means ~ 499 time interval in average)
THETA_3 = (Y_RANGE - LAMBDA_3)/LAMBDA_3 # mean of time interval between the successes 
                                        # (ex. 3 successes in 1000 means ~ 332 time interval in average)

YLIM1 = 5 * (Y_RANGE ** K_1) # in case we got up to 5 empty samples in a row (all zeroes)
YLIM2 = 4 * (Y_RANGE ** K_2) # in case we got up to 4 empty samples in a row (all zeroes)
YLIM3 = 3 * (Y_RANGE ** K_3) # in case we got up to 3 empty samples in a row (all zeroes)

distr_1 = pd.DataFrame(columns = ['time'])
distr_2 = pd.DataFrame(columns = ['time'])
distr_3 = pd.DataFrame(columns = ['time'])

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax1 = plt.subplot2grid((10, 6), (0, 0), rowspan=5, colspan=2)
ax2 = plt.subplot2grid((10, 6), (0, 2), rowspan=5, colspan=2)
ax3 = plt.subplot2grid((10, 6), (0, 4), rowspan=5, colspan=2)

ax1.grid(axis='both', linestyle='--', color='0.95')
ax1.set_xlim(0, Y_RANGE * LAMBDA_1) 
ax1.set_ylim(0, YLIM1)
ax1.set_xlabel('number of measures')
ax1.set_ylabel(f'time between successes of power k={K_1:.2f}')
ax1.set_title(f'Time between the successes of power k={K_1:.2f} (p = {P1})')
ax1.title.set_size(10)

ax2.grid(axis='both', linestyle='--', color='0.95')
ax2.set_xlim(0, Y_RANGE * LAMBDA_2) 
ax2.set_ylim(0, YLIM2) 
ax2.set_xlabel('number of measures')
ax2.set_ylabel(f'time between successes of power k={K_2:.2f}')
ax2.set_title(f'Time between the successes of power k={K_2:.2f} (p = {P2})')
ax2.title.set_size(10)

ax3.grid(axis='both', linestyle='--', color='0.95')
ax3.set_xlim(0, Y_RANGE * LAMBDA_3) 
ax3.set_ylim(0, YLIM3)
ax3.set_xlabel('number of measures')
ax3.set_ylabel(f'time between successes of power k={K_3:.2f}')
ax3.set_title(f'Time between the successes of power k={K_3:.2f} (p = {P3})')
ax3.title.set_size(10)

# https://stackoverflow.com/questions/42435446/how-to-put-text-outside-of-plots
text_1 = ax1.text(50, YLIM1 * 0.9, '', color='r', fontweight='bold') # , transform=plt.gcf().transFigure
text_2 = ax2.text(50, YLIM2 * 0.9, '', color='g', fontweight='bold') # , transform=plt.gcf().transFigure
text_3 = ax3.text(50, YLIM3 * 0.9, '', color='b', fontweight='bold') # , transform=plt.gcf().transFigure

line_1, = ax1.plot([], color='r', label=f'p = {P1}')
line_2, = ax2.plot([], color='g', label=f'p = {P2}')
line_3, = ax3.plot([], color='b', label=f'p = {P3}')

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax4 = plt.subplot2grid((10, 6), (5, 0), rowspan=5, colspan=2)
ax5 = plt.subplot2grid((10, 6), (5, 2), rowspan=5, colspan=2)
ax6 = plt.subplot2grid((10, 6), (5, 4), rowspan=5, colspan=2)

ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")

X_1 = np.linspace(0, int(YLIM1 / 2), 10 * int(YLIM1))
X_2 = np.linspace(0, int(YLIM2 / 2), 10 * int(YLIM2))
X_3 = np.linspace(0.001, int(YLIM3 / 2), 10 * int(YLIM3))

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html
PDF_1 = stats.weibull_min.pdf(X_1, c=1/K_1, scale = THETA_1**K_1)
PDF_2 = stats.weibull_min.pdf(X_2, c=1/K_2, scale = THETA_2**K_2)
PDF_3 = stats.weibull_min.pdf(X_3, c=1/K_3, scale = THETA_3**K_3)

# 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1
# Number of successes: 4
# Times between successes: [3, 8, 4] 
def calc_times(sample: list, df: pd.DataFrame, remainder: int):
    time = remainder
    for event in sample:
        if event == 1:
            df.loc[len(df), 'time'] = time
            time = 0
        elif event == 0:
            time += 1
    return df, time

remainder1 = 0
remainder2 = 0
remainder3 = 0
for i in range(X_RANGE): 
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
    # The Poisson distribution is the limit of the binomial distribution for large N.
    sample_1 = [1 if r < P1 else 0 for r in [random.random() for i in range(Y_RANGE)]]
    sample_2 = [1 if r < P2 else 0 for r in [random.random() for i in range(Y_RANGE)]]
    sample_3 = [1 if r < P3 else 0 for r in [random.random() for i in range(Y_RANGE)]]

    distr_1, remainder1 = calc_times(sample_1, distr_1, remainder1)
    distr_2, remainder2 = calc_times(sample_2, distr_2, remainder2)
    distr_3, remainder3 = calc_times(sample_3, distr_3, remainder3)

    distr_1_pow_k = np.power(distr_1.values, K_1)
    distr_2_pow_k = np.power(distr_2.values, K_2)
    distr_3_pow_k = np.power(distr_3.values, K_3)

    if (i < 100) or (i == X_RANGE - 1):
        mean_1 = distr_1["time"].mean()
        mean_2 = distr_2["time"].mean()
        mean_3 = distr_3["time"].mean()

        text_1.set_text(f'{i}: {mean_1}')
        text_2.set_text(f'{i}: {mean_2}')
        text_3.set_text(f'{i}: {mean_3}')

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html
        # Y = X**k is weibull_min distributed with shape c = 1/k and scale s**k.
        line_1.set_data(distr_1.index.values, distr_1_pow_k)
        line_2.set_data(distr_2.index.values, distr_2_pow_k)
        line_3.set_data(distr_3.index.values, distr_3_pow_k)

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html
        # PDF_1 = stats.weibull_min.pdf(X_1, c=1/K_1, scale = mean_1**K_1)
        # PDF_2 = stats.weibull_min.pdf(X_2, c=1/K_2, scale = mean_2**K_2)
        # PDF_3 = stats.weibull_min.pdf(X_3, c=1/K_3, scale = mean_3**K_3)

        bins_1 = 20
        bins_2 = 40
        bins_3 = 60

        ax4.cla()
        ax5.cla()
        ax6.cla()
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html
        # Y = X**k is weibull_min distributed with shape c = 1/k and scale s**k.
        ax4.hist(distr_1_pow_k, bins = bins_1, density=True, rwidth=0.8, alpha=0.4, color='r', label=f'β={BETA_1:.2f} {Y_RANGE} {P1}')
        ax5.hist(distr_2_pow_k, bins = bins_2, density=True, rwidth=0.8, alpha=0.4, color='g', label=f'β={BETA_2:.2f} {Y_RANGE} {P2}')
        ax6.hist(distr_3_pow_k, bins = bins_3, density=True, rwidth=0.8, alpha=0.4, color='b', label=f'β={BETA_3:.2f} {Y_RANGE} {P3}')
        ax4.plot(X_1, PDF_1, alpha=1.0, color='r', linewidth=2.0)
        ax5.plot(X_2, PDF_2, alpha=1.0, color='g', linewidth=2.0)
        ax6.plot(X_3, PDF_3, alpha=1.0, color='b', linewidth=2.0)
    
        ax4.grid(axis='both', linestyle='--', color='0.95')
        # ax4.xaxis.set_major_locator(ticker.MultipleLocator(int(YLIM1 / 10)))
        # ax4.set_xlim(0, YLIM1 / 2)
        # ax3.set_ylim(0, 1)
        # ax3.set_xlabel('')
        # ax3.set_ylabel('')
        # ax3.set_title('')
        ax4.legend(loc="upper right")

        ax5.grid(axis='both', linestyle='--', color='0.95')
        # ax5.xaxis.set_major_locator(ticker.MultipleLocator(int(YLIM2 / 10)))
        # ax5.set_xlim(0, YLIM2 / 2)
        ax5.legend(loc="upper right")

        ax6.grid(axis='both', linestyle='--', color='0.95')
        # ax6.xaxis.set_major_locator(ticker.MultipleLocator(int(YLIM3 / 10)))
        # ax6.set_xlim(0, YLIM3 / 2)
        ax6.legend(loc="upper right")

        ax4.text(8.75, 0.125, f'W(λ={LAMBDA_1},β={BETA_1:.2f})')
        ax5.text(80, 0.008, f'W(λ={LAMBDA_2},β={BETA_2:.2f})')
        ax6.text(1000, 0.001, f'W(λ={LAMBDA_3},β={BETA_3:.2f})')

    (i < 100) and (i % 20 == 0) and plt.tight_layout()

    # pause the plot for 0.01s before next point is shown 
    # plt.pause(0.5 if i < 100 else 0.0001) 
    (i < 100) and plt.pause(0.05)

plt.show()