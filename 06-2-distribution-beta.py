from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math
import random

P_NUM = 100

data = pd.DataFrame(data={ 'p':  np.linspace(0, 1, P_NUM + 1) })
ax = plt.subplot2grid((1, 1), (0, 0))
X = np.linspace(0, 1, P_NUM + 1)

for i in range(26):
    a = (i / 5)
    b = 5
    data['distr'] = data['p']**a * (1 - data['p'])**b
    data['distr'] /= data['distr'].sum()

    # print(data)
    # print(data['distr'].sum())
    # print(np.sum(stats.beta.pdf(X, a + 1, b + 1) / 100))

    ax.cla()
    ax.bar(data['p'], data['distr'], width=0.008, alpha=0.4)
    # https://stackoverflow.com/questions/65391178/scipy-normal-pdf-summing-to-values-larger-than-1
    ax.plot(X, stats.beta.pdf(X, a + 1, b + 1) / P_NUM) # / P_NUM due to discrete X values
    ax.text(i / 50, data['distr'].max(), f'Beta(α={a:.2f},β={b:.2f})')
    ax.grid(axis='both', linestyle='--', color='0.95')

    plt.tight_layout()
    plt.pause(0.001)

for i in range(26):
    a = 5
    b = 5 - (i / 5)
    data['distr'] = data['p']**a * (1 - data['p'])**b
    data['distr'] /= data['distr'].sum()

    ax.cla()
    ax.bar(data['p'], data['distr'], width=0.008, alpha=0.4)
    # https://stackoverflow.com/questions/65391178/scipy-normal-pdf-summing-to-values-larger-than-1
    ax.plot(X, stats.beta.pdf(X, a + 1, b + 1) / P_NUM) # / P_NUM due to discrete X values
    ax.text(0.5 + i / 50, data['distr'].max(), f'Beta(α={a:.2f},β={b:.2f})')
    ax.grid(axis='both', linestyle='--', color='0.95')

    plt.tight_layout()
    plt.pause(0.001)

plt.tight_layout()
plt.show()