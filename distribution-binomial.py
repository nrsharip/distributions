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

P_1 = 0.1
P_2 = 0.5
P_3 = 0.8

distr1 = pd.DataFrame(columns = ['count'])
distr2 = pd.DataFrame(columns = ['count'])
distr3 = pd.DataFrame(columns = ['count'])

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax0 = plt.subplot2grid((10, 6), (0, 0), rowspan=5, colspan=2)
ax1 = plt.subplot2grid((10, 6), (0, 2), rowspan=5, colspan=2)
ax2 = plt.subplot2grid((10, 6), (0, 4), rowspan=5, colspan=2)

ax0.grid(axis='both', linestyle='--', color='0.95')
ax0.set_xlim(0, X_RANGE) 
ax0.set_ylim(0, Y_RANGE) 
ax0.set_xlabel('sample\'s number')
ax0.set_ylabel('count of 1')
ax0.set_title('Number of successes (p = 0.1)')

ax1.grid(axis='both', linestyle='--', color='0.95')
ax1.set_xlim(0, X_RANGE) 
ax1.set_ylim(0, Y_RANGE) 
ax1.set_xlabel('sample\'s number')
ax1.set_ylabel('count of 1')
ax1.set_title('Number of successes (p = 0.5)')

ax2.grid(axis='both', linestyle='--', color='0.95')
ax2.set_xlim(0, X_RANGE) 
ax2.set_ylim(0, Y_RANGE)
ax2.set_xlabel('sample\'s number')
ax2.set_ylabel('count of 1')
ax2.set_title('Number of successes (p = 0.8)')

# https://stackoverflow.com/questions/42435446/how-to-put-text-outside-of-plots
text1 = ax0.text(50, 19, '') # , transform=plt.gcf().transFigure
text1_1 = ax0.text(50, 18, '', color='r', fontweight='bold') # , transform=plt.gcf().transFigure
text2 = ax1.text(50, 19, '') # , transform=plt.gcf().transFigure
text2_1 = ax1.text(50, 18, '', color='g', fontweight='bold') # , transform=plt.gcf().transFigure
text3 = ax2.text(50, 2, '') # , transform=plt.gcf().transFigure
text3_1 = ax2.text(50, 1, '', color='b', fontweight='bold') # , transform=plt.gcf().transFigure

line1, = ax0.plot([3, 4, 5], color='r', label='p=0.25')
line2, = ax1.plot([3, 4, 5], color='g', label='p=0.5')
line3, = ax2.plot([3, 4, 5], color='b', label='p=0.75')

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax3 = plt.subplot2grid((10, 6), (5, 0), rowspan=5, colspan=6)

ax0.legend(loc="center right")
ax1.legend(loc="lower right")
ax2.legend(loc="center right")

X = np.linspace(0, Y_RANGE, Y_RANGE + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
PPF_1 = stats.binom.pmf(X, Y_RANGE, P_1)
PPF_2 = stats.binom.pmf(X, Y_RANGE, P_2)
PPF_3 = stats.binom.pmf(X, Y_RANGE, P_3)

for i in range(X_RANGE): 
    sample1 = [1 if r < P_1 else 0 for r in [random.random() for i in range(Y_RANGE)]]
    sample2 = [1 if r < P_2 else 0 for r in [random.random() for i in range(Y_RANGE)]]
    sample3 = [1 if r < P_3 else 0 for r in [random.random() for i in range(Y_RANGE)]]

    distr1.loc[i] = sample1.count(1)
    distr2.loc[i] = sample2.count(1)
    distr3.loc[i] = sample3.count(1)

    if (i < 100) or (i == X_RANGE - 1):
        text1.set_text(f'{i}: {sample1}')
        text1_1.set_text(f'Number of "1": {sample1.count(1)}')
        text2.set_text(f'{i}: {sample2}')
        text2_1.set_text(f'Number of "1": {sample2.count(1)}')
        text3.set_text(f'{i}: {sample3}')
        text3_1.set_text(f'Number of "1": {sample3.count(1)}')

        line1.set_data(distr1.index.values, distr1['count'].values)
        line2.set_data(distr2.index.values, distr2['count'].values)
        line3.set_data(distr3.index.values, distr3['count'].values)

        bins1 = distr1["count"].max() - distr1["count"].min()
        bins2 = distr2["count"].max() - distr2["count"].min()
        bins3 = distr3["count"].max() - distr3["count"].min()

        ax3.cla()
        ax3.hist(distr1.values, bins = bins1 if bins1 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='r', label='sample 20 0.1')
        ax3.hist(distr2.values, bins = bins2 if bins2 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='g', label='sample 20 0.5')
        ax3.hist(distr3.values, bins = bins3 if bins3 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='b', label='sample 20 0.8')
        ax3.plot(X, PPF_1, marker='o', linestyle='dashed', alpha=1.0, color='r', linewidth=2.0)
        ax3.plot(X, PPF_2, marker='o', linestyle='dashed', alpha=1.0, color='g', linewidth=2.0)
        ax3.plot(X, PPF_3, marker='o', linestyle='dashed', alpha=1.0, color='b', linewidth=2.0)
    
        ax3.grid(axis='both', linestyle='--', color='0.95')
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax3.set_xlabel('count of 1')
        # ax3.set_ylabel('density')
        # ax3.set_title('Density Plots for number of successes (p = 0.1 0.5 0.8)')
        ax3.legend(loc="upper right")

        ax3.text(3, 0.2, "B(20, 0.1)")
        ax3.text(10, 0.19, "B(20, 0.5)")
        ax3.text(15, 0.22, "B(20, 0.8)")

    (i < 100) and (i % 20 == 0) and plt.tight_layout()

    # pause the plot for 0.01s before next point is shown 
    # plt.pause(0.5 if i < 100 else 0.0001) 
    (i < 100) and plt.pause(0.05)

count1 = pd.cut(distr1["count"], distr1["count"].max() - distr1["count"].min()).value_counts()
count2 = pd.cut(distr2["count"], distr2["count"].max() - distr2["count"].min()).value_counts()
count3 = pd.cut(distr3["count"], distr3["count"].max() - distr3["count"].min()).value_counts()

print(count1)
print(count2)
print(count3)

plt.show()