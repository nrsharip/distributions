from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math
import random

cards = pd.DataFrame([
    ['C', '2'], ['D', '2'], ['H', '2'], ['S', '2'],
    ['C', '3'], ['D', '3'], ['H', '3'], ['S', '3'],
    ['C', '4'], ['D', '4'], ['H', '4'], ['S', '4'],
    ['C', '5'], ['D', '5'], ['H', '5'], ['S', '5'],
    ['C', '6'], ['D', '6'], ['H', '6'], ['S', '6'],
    ['C', '7'], ['D', '7'], ['H', '7'], ['S', '7'],
    ['C', '8'], ['D', '8'], ['H', '8'], ['S', '8'],
    ['C', '9'], ['D', '9'], ['H', '9'], ['S', '9'],
    ['C', '10'], ['D', '10'], ['H', '10'], ['S', '10'],
    ['C', 'J'], ['D', 'J'], ['H', 'J'], ['S', 'J'],
    ['C', 'Q'], ['D', 'Q'], ['H', 'Q'], ['S', 'Q'],
    ['C', 'K'], ['D', 'K'], ['H', 'K'], ['S', 'K'],
    ['C', 'A'], ['D', 'A'], ['H', 'A'], ['S', 'A'],
], columns=['Suit', 'Rank'])

CARDS_1 = 5
CARDS_2 = 15
CARDS_3 = 25

N = len(cards)
S_COUNT = cards['Suit'].values.tolist().count('S')
H_COUNT = cards['Suit'].values.tolist().count('H')

X_RANGE = 1000
Y_RANGE = 5

distr_1 = pd.DataFrame(columns = ['count'])
distr_2 = pd.DataFrame(columns = ['count'])
distr_3 = pd.DataFrame(columns = ['count'])

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax1 = plt.subplot2grid((10, 6), (0, 0), rowspan=5, colspan=2)
ax2 = plt.subplot2grid((10, 6), (0, 2), rowspan=5, colspan=2)
ax3 = plt.subplot2grid((10, 6), (0, 4), rowspan=5, colspan=2)

ax1.grid(axis='both', linestyle='--', color='0.95')
ax1.set_xlim(0, X_RANGE) 
ax1.set_ylim(0, CARDS_1) 
ax1.set_xlabel('sample\'s number')
ax1.set_ylabel('count of spades')
ax1.set_title(f'Number of Spades and Hearts (n = {CARDS_1})')

ax2.grid(axis='both', linestyle='--', color='0.95')
ax2.set_xlim(0, X_RANGE) 
ax2.set_ylim(0, CARDS_2) 
ax2.set_xlabel('sample\'s number')
ax2.set_ylabel('count of spades and hearts')
ax2.set_title(f'Number of Spades and Hearts (n = {CARDS_2})')

ax3.grid(axis='both', linestyle='--', color='0.95')
ax3.set_xlim(0, X_RANGE) 
ax3.set_ylim(0, CARDS_3)
ax3.set_xlabel('sample\'s number')
ax3.set_ylabel('count of spades and hearts')
ax3.set_title(f'Number of Spades and Hearts (n = {CARDS_3})')

# https://stackoverflow.com/questions/42435446/how-to-put-text-outside-of-plots
text_1 = ax1.text(50, CARDS_1 * 0.9, '', color='black', fontweight='bold') # , transform=plt.gcf().transFigure
text_2 = ax2.text(50, CARDS_2 * 0.9, '', color='black', fontweight='bold') # , transform=plt.gcf().transFigure
text_3 = ax3.text(50, CARDS_3 * 0.9, '', color='black', fontweight='bold') # , transform=plt.gcf().transFigure

line_1, = ax1.plot([], color='r', label=f'n={CARDS_1}')
line_2, = ax2.plot([], color='g', label=f'n={CARDS_2}')
line_3, = ax3.plot([], color='b', label=f'n={CARDS_3}')

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax4 = plt.subplot2grid((10, 6), (5, 0), rowspan=5, colspan=6)

ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")

X_1 = np.linspace(0, CARDS_1, CARDS_1 + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
PMF_1 = stats.hypergeom.pmf(X_1, N, S_COUNT + H_COUNT, CARDS_1)

X_2 = np.linspace(0, CARDS_2, CARDS_2 + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
PMF_2 = stats.hypergeom.pmf(X_2, N, S_COUNT + H_COUNT, CARDS_2)

X_3 = np.linspace(0, CARDS_3, CARDS_3 + 1)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
PMF_3 = stats.hypergeom.pmf(X_3, N, S_COUNT + H_COUNT, CARDS_3)

for i in range(X_RANGE):
    sample_1 = cards.sample(CARDS_1)
    sample_2 = cards.sample(CARDS_2)
    sample_3 = cards.sample(CARDS_3)

    distr_1.loc[i] = sample_1['Suit'].values.tolist().count('S')
    distr_1.loc[i] += sample_1['Suit'].values.tolist().count('H')
    distr_2.loc[i] = sample_2['Suit'].values.tolist().count('S')
    distr_2.loc[i] += sample_2['Suit'].values.tolist().count('H')
    distr_3.loc[i] = sample_3['Suit'].values.tolist().count('S')
    distr_3.loc[i] += sample_3['Suit'].values.tolist().count('H')

    if (i < 100) or (i == X_RANGE - 1):
        # text025.set_text(f'{i}: {sample_1}')
        text_1.set_text(f'Number of Spades and Hearts: {distr_1.loc[i, "count"]}')
        # text05.set_text(f'{i}: {sample_2}')
        text_2.set_text(f'Number of Spades and Hearts: {distr_2.loc[i, "count"]}')
        # text075.set_text(f'{i}: {sample_3}')
        text_3.set_text(f'Number of Spades and Hearts: {distr_3.loc[i, "count"]}')

        line_1.set_data(distr_1.index.values, distr_1['count'].values)
        line_2.set_data(distr_2.index.values, distr_2['count'].values)
        line_3.set_data(distr_3.index.values, distr_3['count'].values)

        bins_1 = distr_1["count"].max() - distr_1["count"].min()
        bins_2 = distr_2["count"].max() - distr_2["count"].min()
        bins_3 = distr_3["count"].max() - distr_3["count"].min()

        ax4.cla()
        ax4.hist(distr_1.values, bins = bins_1 if bins_1 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='r', label=f'sample {CARDS_1}')
        ax4.hist(distr_2.values, bins = bins_2 if bins_2 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='g', label=f'sample {CARDS_2}')
        ax4.hist(distr_3.values, bins = bins_3 if bins_3 > 0 else 1, density=True, rwidth=0.8, alpha=0.4, color='b', label=f'sample {CARDS_3}')
        ax4.plot(X_1, PMF_1, marker='o', linestyle='dashed', alpha=1.0, color='r', linewidth=2.0)
        ax4.plot(X_2, PMF_2, marker='o', linestyle='dashed', alpha=1.0, color='g', linewidth=2.0)
        ax4.plot(X_3, PMF_3, marker='o', linestyle='dashed', alpha=1.0, color='b', linewidth=2.0)
    
        ax4.grid(axis='both', linestyle='--', color='0.95')
        ax4.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax3.set_xlabel('count of 1')
        # ax3.set_ylabel('density')
        # ax3.set_title('Density Plots for number of successes (p = 0.1 0.5 0.8)')
        ax4.legend(loc="upper right")

        ax4.text(3.2, 0.3, f'Hypergeometric(N={N}, K={S_COUNT + H_COUNT}, n={CARDS_1})')
        ax4.text(8, 0.24, f'Hypergeometric(N={N}, K={S_COUNT + H_COUNT}, n={CARDS_2})')
        ax4.text(14, 0.17, f'Hypergeometric(N={N}, K={S_COUNT + H_COUNT}, n={CARDS_3})')

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