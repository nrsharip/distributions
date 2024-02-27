from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import math

DATA_PATH = Path().resolve()

# Define paths to data sets. If you don't keep your data in the same directory as the code, adapt the path names.

DATA_CSV = DATA_PATH / 'data.csv'
BINS = 50

raw_data = pd.read_csv(DATA_CSV).squeeze('columns')
intervals = pd.interval_range(0, raw_data.max(), BINS)
print(intervals)

# Confidence Intervals
ci90 = list(raw_data.quantile([0.05, 0.95]))

mean = raw_data.mean()
print("mean (μ): ", mean)

SAMPLE_SIZE_1 = 3
SAMPLE_SIZE_2 = 50

# max - min
# --------- - standard deviation limit
#     2
# 
#  max - min
# ----------- - Standard Error limit
# 2 * sqrt(3)
STD_ERR_LIMIT_1 = (raw_data.max() - raw_data.min()) / (2*math.sqrt(SAMPLE_SIZE_1))
STD_ERR_LIMIT_2 = (raw_data.max() - raw_data.min()) / (2*math.sqrt(SAMPLE_SIZE_2))

# https://stackoverflow.com/questions/53978121/how-can-i-plot-four-subplots-with-different-colspans
ax00 = plt.subplot2grid((12, 6), (0, 0), rowspan=4, colspan=2)
ax01 = plt.subplot2grid((12, 6), (0, 2), rowspan=4, colspan=2)
ax02 = plt.subplot2grid((12, 6), (0, 4), rowspan=4, colspan=2)
ax10 = plt.subplot2grid((12, 6), (4, 0), rowspan=4, colspan=2)
ax11 = plt.subplot2grid((12, 6), (4, 2), rowspan=4, colspan=2)
ax12 = plt.subplot2grid((12, 6), (4, 4), rowspan=4, colspan=2)

ax20 = plt.subplot2grid((12, 6), (8, 0), rowspan=4, colspan=3)
ax21 = plt.subplot2grid((12, 6), (8, 3), rowspan=4, colspan=3)

ax00.grid(axis='both', linestyle='--', color='0.95')
# ax00.set_xlabel('raw data from CSV')
# ax00.set_ylabel('density')
# ax00.set_title('Density Plot for CSV data')

ax10.grid(axis='both', linestyle='--', color='0.95')
# ax10.set_xlabel('raw data from CSV')
# ax10.set_ylabel('density')
# ax10.set_title('Density Plot for CSV data')

ax01.grid(axis='both', linestyle='--', color='0.95')
# ax01.set_xlabel('sample number')
# ax01.set_ylabel('sample mean (x̄)')
# ax01.set_title('Sample Mean (x̄) for sample size of 3 (n=3)')

ax02.grid(axis='both', linestyle='--', color='0.95')
# ax02.set_xlabel('sample number')
# ax02.set_ylabel('Standard Error Estimator (SEE03)')
# ax02.set_title('Standard Error Estimator (n=3)')

ax11.grid(axis='both', linestyle='--', color='0.95')
# ax11.set_xlabel('sample number')
# ax11.set_ylabel('sample mean (x̄)')
# ax11.set_title('Sample Mean (x̄) for sample size of 50 (n=50)')

ax12.grid(axis='both', linestyle='--', color='0.95')
# ax12.set_xlabel('sample number')
# ax12.set_ylabel('Standard Error Estimator (SEE50)')
# ax12.set_title('Standard Error Estimator (n=50)')

####### [0,0] #######
N00, bins00, patches00 = ax00.hist(
    raw_data,
    bins = BINS,
    density=True,
    rwidth=0.8,
    label='Data Density'
)
ax00.text(90_000, 0.000015, f'Population Mean (μ) = {mean:.0f}')
ax00.xaxis.set_major_locator(ticker.MultipleLocator(30_000))

####### [1,0] #######
N10, bins10, patches10 = ax10.hist(
    raw_data,
    bins = BINS,
    density=True,
    rwidth=0.8, 
    label='Data Density'
)
ax10.text(90_000, 0.000015, f'Population Mean (μ) = {mean:.0f}')
ax10.xaxis.set_major_locator(ticker.MultipleLocator(30_000))

####### [0,1] #######
line_1, = ax01.plot([], color='r', label='Sample Mean (n=3)')
# https://stackoverflow.com/questions/57093572/set-y-axis-to-scientific-notation
ax01.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax01.set_xlim(0, 1000) 
ax01.set_ylim(0, raw_data.max()) 

text_1 = ax01.text(50, 4 * raw_data.max() / 5, f'')

####### [0,2] #######
line_see_1, = ax02.plot([], color='r', label='SEE (n=3)')
# https://stackoverflow.com/questions/57093572/set-y-axis-to-scientific-notation
ax02.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax02.set_xlim(0, 1000) 
ax02.set_ylim(0, STD_ERR_LIMIT_1) 

text_1_SEE = ax02.text(25, STD_ERR_LIMIT_1 * 0.9, f'')

####### [1,1] #######
line_2, = ax11.plot([], color='g', label='Sample Mean (n=50)')
# https://stackoverflow.com/questions/57093572/set-y-axis-to-scientific-notation
ax11.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax11.set_xlim(0, 1000) 
ax11.set_ylim(0, raw_data.max()) 

text_2 = ax11.text(50, 4 * raw_data.max() / 5, f'')

####### [1,2] #######
line_see_2, = ax12.plot([], color='g', label='SEE (n=50)')
# https://stackoverflow.com/questions/57093572/set-y-axis-to-scientific-notation
ax12.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax12.set_xlim(0, 1000) 
ax12.set_ylim(0, STD_ERR_LIMIT_2)

text_2_SEE = ax12.text(25, STD_ERR_LIMIT_2 * 0.9, f'')

####### Legends #######

# ax00.legend(loc="upper right")
# ax10.legend(loc="upper right")
# ax01.legend(loc="lower right")
# ax11.legend(loc="lower right")

sample_mean_1 = pd.DataFrame(columns = ['mean'])
sample_mean_2 = pd.DataFrame(columns = ['mean'])
sample_see_1 = pd.DataFrame(columns = ['see'])
sample_see_2 = pd.DataFrame(columns = ['see'])
t_sample_mean_1 = pd.DataFrame(columns = ['mean'])
t_sample_mean_2 = pd.DataFrame(columns = ['mean'])

# https://www.geeksforgeeks.org/dynamic-visualization-using-python/
for i in range(1000): 
    sample_1 = raw_data.sample(SAMPLE_SIZE_1)
    sample_2 = raw_data.sample(SAMPLE_SIZE_2)

    # https://en.wikipedia.org/wiki/Standard_error#Estimate
    see_1 = sample_1.std() / math.sqrt(SAMPLE_SIZE_1)
    see_2 = sample_2.std() / math.sqrt(SAMPLE_SIZE_2)

    if see_1 == 0 or see_2 == 0:
        continue

    bins = [intervals.get_loc(value) for value in sample_1.values]
    for j in range(len(patches00)):
        patches00[j].set_facecolor('r' if j in bins else 'b')

    bins = [intervals.get_loc(value) for value in sample_2.values]
    for j in range(len(patches10)):
        patches10[j].set_facecolor('g' if j in bins else 'b')

    ########## Means ###########
    # https://www.geeksforgeeks.org/how-to-add-one-row-in-an-existing-pandas-dataframe/
    sample_mean_1.loc[i] = [sample_1.mean()]
    line_1.set_data(sample_mean_1.index.values, sample_mean_1.values)
    sample_mean_2.loc[i] = [sample_2.mean()]
    line_2.set_data(sample_mean_2.index.values, sample_mean_2.values)

    ########## Standard Errors ###########
    sample_see_1.loc[i] = [see_1]
    line_see_1.set_data(sample_see_1.index.values, sample_see_1.values)
    sample_see_2.loc[i] = [see_2]
    line_see_2.set_data(sample_see_2.index.values, sample_see_2.values)

    # https://stackoverflow.com/questions/39223286/how-to-refresh-text-in-matplotlib
    text_1.set_text(f'' 
                    + f'Sample {i}: {str(sample_1.values)}\n' 
                    + f'Sample mean (x̄): {sample_1.mean()}'
    )
    text_2.set_text(f'' 
                    + f'Sample {i}: \n' 
                    + f'Sample mean (x̄): {sample_2.mean()}'
    )
    text_1_SEE.set_text(f'' 
                        # Standard Deviation of the entire set of Sample Means
                        # + f'Standard Deviation (s) = {sample_mean_1["mean"].std():.2f}\n'
                        + f'Standard Error Estimator (SEE03) = {see_1:.2f}'
    )
    text_2_SEE.set_text(f'' 
                        # Standard Deviation of the entire set of Sample Means
                        # + f'Standard Deviation (s) = {sample_mean_2["mean"].std():.2f}\n' 
                        + f'Standard Error Estimator (SEE50) = {see_2:.2f}'
    )

    # https://en.wikipedia.org/wiki/Student%27s_t-test#One-sample_t-test
    t_sample_mean_1.loc[i] = ((sample_1.mean() - mean) / see_1) - 1 # -1 to separate two distributions (n=3 and n=50)
    t_sample_mean_2.loc[i] = ((sample_2.mean() - mean) / see_2) + 1 # +1 to separate two distributions (n=3 and n=50)

    # t_1_min = math.floor(t_sample_mean_1.min())
    # t_1_max = math.ceil(t_sample_mean_1.max())
    # t_2_min = math.floor(t_sample_mean_2.min())
    # t_2_max = math.ceil(t_sample_mean_2.max())

    t_1_min = -10
    t_1_max = 10
    t_2_min = -10
    t_2_max = 10

    # X_1 = np.linspace(math.floor(sample_mean_1.min()), math.ceil(sample_mean_1.max()), 400)
    X_1 = np.linspace(ci90[0], ci90[1], 400)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    PDF_1 = stats.t.pdf(x=X_1, df=2, loc=mean, scale=see_1)

    # X_2 = np.linspace(math.floor(sample_mean_2.min()), math.ceil(sample_mean_2.max()), 400)
    X_2 = np.linspace(ci90[0], ci90[1], 400)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    PDF_2 = stats.t.pdf(x=X_2, df=49, loc=mean, scale=see_2)

    T_X_1 = np.linspace(t_1_min, t_1_max, 400)
    T_X_2 = np.linspace(t_2_min, t_2_max, 400)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    T_PDF_1 = stats.t.pdf(T_X_1, SAMPLE_SIZE_1 - 1,loc=-1,scale=1)
    T_PDF_2 = stats.t.pdf(T_X_2, SAMPLE_SIZE_2 - 1,loc=1,scale=1)

    if (i < 100) or (i == 999): # 
        ####### [2,0] #######
        ax20.cla()
        ax20.hist(sample_mean_1.values, bins = 20, density=True, rwidth=0.9, alpha=0.8, color='r', label='Sample Mean Density (n=3)')
        ax20.hist(sample_mean_2.values, bins = 20, density=True, rwidth=0.9, alpha=0.8, color='g', label='Sample Mean Density (n=50)')
        ax20.plot(X_1, PDF_1, alpha=1.0, color='black', linewidth=2.0)
        ax20.plot(X_2, PDF_2, alpha=1.0, color='purple', linewidth=2.0)

        ax20.text(75000, 0.00004, f'T(μ, SEE03, df={SAMPLE_SIZE_1 - 1})')
        ax20.text(85000, 0.00002, f'T(μ, SEE50, df={SAMPLE_SIZE_2 - 1})')

        ax20.set_xlim(ci90[0], ci90[1])
        # https://stackoverflow.com/questions/57093572/set-y-axis-to-scientific-notation
        ax20.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ####### [2,1] #######
        ax21.cla()
        ax21.hist(t_sample_mean_1.values, bins = 80, density=True, rwidth=0.9, alpha=0.8, color='r', label='Sample Mean T-score Density (n=3)')
        ax21.hist(t_sample_mean_2.values, bins = 20, density=True, rwidth=0.9, alpha=0.8, color='g', label='Sample Mean T-score Density (n=50)')
        ax21.plot(T_X_1, T_PDF_1, alpha=1.0, color='black', linewidth=2.0)
        ax21.plot(T_X_2, T_PDF_2, alpha=1.0, color='purple', linewidth=2.0)

        ax21.text(-5, 0.3, f'T(-1, 1, df={SAMPLE_SIZE_1 - 1})')
        ax21.text(2.5, 0.3, f'T(1, 1, df={SAMPLE_SIZE_2 - 1})')

        ax21.set_xlim(max(-10, min(t_1_min, t_2_min)), min(10, max(t_1_max, t_2_max)))
        ####### Legends, Titles, Labels #######

        ax20.grid(axis='both', linestyle='--', color='0.95')
        # ax20.set_xlabel('sample mean')
        # ax20.set_ylabel('density')
        # ax20.set_title('Sample Means (sample sizes n=3 and n=50)')

        ax21.grid(axis='both', linestyle='--', color='0.95')
        # ax21.set_xlabel('sample mean')
        # ax21.set_ylabel('density')
        # ax21.set_title('Sample Mean t-scores (sample sizes n=3 and n=50)')
        
        # ax20.legend(loc="upper right")
        # ax21.legend(loc="upper left")

        ((i % 20 == 0) or (i == 999)) and plt.tight_layout()

    # pause the plot for 0.01s before next point is shown 
    # plt.pause(0.5 if i < 100 else 0.0001) 
    (i < 100) and plt.pause(0.05)

print("end")

plt.tight_layout()
plt.show()

