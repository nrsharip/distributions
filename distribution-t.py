from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import math

DATA_PATH = Path().resolve()

# Define paths to data sets. If you don't keep your data in the same directory as the code, adapt the path names.

DATA_CSV = DATA_PATH / 'data.csv'
BINS = 100

## Sampling Distribution of a Statistic

raw_data = pd.read_csv(DATA_CSV).squeeze('columns')

intervals = pd.interval_range(0, raw_data.max(), BINS)

print(intervals)

mean = raw_data.mean()

print("mean (μ): ", mean)

# print("std 5: ", sample_mean_05["income"].std())
# print("std 20: ", sample_mean_20["income"].std())

fig, axes = plt.subplots(figsize=(14, 5), ncols=2, nrows=3)

axes[0,0].grid(axis='both', linestyle='--', color='0.95')
axes[0,0].set_xlabel('raw data from CSV')
axes[0,0].set_ylabel('density')
axes[0,0].set_title('Density Plot for CSV data')

axes[1,0].grid(axis='both', linestyle='--', color='0.95')
axes[1,0].set_xlabel('raw data from CSV')
axes[1,0].set_ylabel('density')
axes[1,0].set_title('Density Plot for CSV data')

axes[0,1].grid(axis='both', linestyle='--', color='0.95')
axes[0,1].set_xlabel('sample number')
axes[0,1].set_ylabel('sample mean (x̄)')
axes[0,1].set_title('Sample Mean (x̄) for sample size of 5 (n=5)')

axes[1,1].grid(axis='both', linestyle='--', color='0.95')
axes[1,1].set_xlabel('sample number')
axes[1,1].set_ylabel('sample mean (x̄)')
axes[1,1].set_title('Sample Mean (x̄) for sample size of 20 (n=20)')


####### [0,0] #######
N00, bins00, patches00 = axes[0,0].hist(
    raw_data,
    bins = BINS,
    density=True,
    rwidth=0.8,
    label='Data Density'
)
axes[0,0].text(125_000, 0.00001, f'Population Mean (μ) = {mean:.0f}')

####### [1,0] #######
N10, bins10, patches10 = axes[1,0].hist(
    raw_data,
    bins = BINS,
    density=True,
    rwidth=0.8, 
    label='Data Density'
)
axes[1,0].text(125_000, 0.00001, f'Population Mean (μ) = {mean:.0f}')

####### [0,1] #######
line03, = axes[0,1].plot([], color='r', label='Sample Mean (n=3)')
# Set the x-axis and y-axis limits to 100 
axes[0,1].set_xlim(0, 1000) 
axes[0,1].set_ylim(0, raw_data.max()) 

text03 = axes[0,1].text(5, 4*raw_data.max()/5, f'')
text05_SEE = axes[0,1].text(25, 5_000, f'')

####### [1,1] #######
line50, = axes[1,1].plot([], color='g', label='Sample Mean (n=50)') 
# Set the x-axis and y-axis limits to 100 
axes[1,1].set_xlim(0, 1000) 
axes[1,1].set_ylim(0, raw_data.max()) 

text50 = axes[1,1].text(5, 3.5*raw_data.max()/5, f'')
text50_SEE = axes[1,1].text(25, 5_000, f'')

####### Legends #######

axes[0,0].legend(loc="upper right")
axes[1,0].legend(loc="upper right")
axes[0,1].legend(loc="lower right")
axes[1,1].legend(loc="lower right")

sample_mean_03 = pd.DataFrame(columns = ['mean_03'])
sample_mean_50 = pd.DataFrame(columns = ['mean_50'])
t_sample_mean_03 = pd.DataFrame(columns = ['mean_03'])
t_sample_mean_50 = pd.DataFrame(columns = ['mean_50'])

# https://www.geeksforgeeks.org/dynamic-visualization-using-python/
for i in range(1000): 
    sample_03 = raw_data.sample(3)
    sample_50 = raw_data.sample(50)

    see_03 = sample_03.std() / math.sqrt(3)
    see_50 = sample_50.std() / math.sqrt(50)

    if see_03 == 0 or see_50 == 0:
        continue
    # print(see_03, see_50)

    bins = [intervals.get_loc(value) for value in sample_03.values]
    for j in range(len(patches00)):
        patches00[j].set_facecolor('r' if j in bins else 'b')

    bins = [intervals.get_loc(value) for value in sample_50.values]
    for j in range(len(patches10)):
        patches10[j].set_facecolor('g' if j in bins else 'b')

    # https://www.geeksforgeeks.org/how-to-add-one-row-in-an-existing-pandas-dataframe/
    sample_mean_03.loc[i] = [sample_03.mean()]
    line03.set_data(sample_mean_03.index.values, sample_mean_03.values)

    sample_mean_50.loc[i] = [sample_50.mean()]
    line50.set_data(sample_mean_50.index.values, sample_mean_50.values)

    # https://stackoverflow.com/questions/39223286/how-to-refresh-text-in-matplotlib
    text03.set_text(f'Sample {i}: {str(sample_03.values)}\nSample mean (x̄): {sample_03.mean()}')
    text50.set_text(f'Sample {i}: Sample mean (x̄): {sample_50.mean()}')
    text05_SEE.set_text(f'Standard Deviation (s) = {sample_mean_03["mean_03"].std():.2f}\n'
                       + f'Standard Error Estimator (SEE03) = {see_03:.2f}')
    text50_SEE.set_text(f'Standard Deviation (s) = {sample_mean_50["mean_50"].std():.2f}\n'
                       + f'Standard Error Estimator (SEE50) = {see_50:.2f}')

    t_sample_mean_03.loc[i] = [(sample_03.mean() - mean) / see_03 - 1] # to separate two distributions (n=3 and n=50)
    t_sample_mean_50.loc[i] = [(sample_50.mean() - mean) / see_50 + 1] # to separate two distributions (n=3 and n=50)

    X_03 = np.linspace(math.floor(sample_mean_03.min()), math.ceil(sample_mean_03.max()), 400)
    # https://proclusacademy.com/blog/practical/normal-distribution-python-scipy/
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    PDF_03 = stats.t.pdf(x=X_03, df=2, loc=mean, scale=see_03)

    X_50 = np.linspace(math.floor(sample_mean_50.min()), math.ceil(sample_mean_50.max()), 400)
    # https://proclusacademy.com/blog/practical/normal-distribution-python-scipy/
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    PDF_50 = stats.t.pdf(x=X_50, df=49, loc=mean, scale=see_50)

    T_X_03 = np.linspace(math.floor(t_sample_mean_03.min()), math.ceil(t_sample_mean_03.max()), 400)
    T_X_50 = np.linspace(math.floor(t_sample_mean_50.min()), math.ceil(t_sample_mean_50.max()), 400)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    T_PDF_3 = stats.t.pdf(T_X_03, 2,loc=-1,scale=1)
    T_PDF_49 = stats.t.pdf(T_X_50, 49,loc=1,scale=1)

    if (i < 100) or (i == 999): # 
        ####### [2,0] #######
        axes[2,0].cla()
        axes[2,0].hist(sample_mean_03.values, bins = BINS, density=True, rwidth=0.9, alpha=0.8, color='r', label='Sample Mean Density (n=3)')
        axes[2,0].hist(sample_mean_50.values, bins = BINS, density=True, rwidth=0.9, alpha=0.8, color='g', label='Sample Mean Density (n=50)')
        axes[2,0].plot(X_03, PDF_03, alpha=1.0, color='black', linewidth=2.0)
        axes[2,0].plot(X_50, PDF_50, alpha=1.0, color='purple', linewidth=2.0)

        axes[2,0].text(75000, 0.00004, f'T(μ, SEE03, df=2)')
        axes[2,0].text(85000, 0.00002, f'T(μ, SEE50, df=49)')

        # axes[2,0].set_xlim(0, raw_data.max()) 
        ####### [2,1] #######
        axes[2,1].cla()
        axes[2,1].hist(t_sample_mean_03.values, bins = BINS, density=True, rwidth=0.9, alpha=0.8, color='r', label='Sample Mean T-score Density (n=3)')
        axes[2,1].hist(t_sample_mean_50.values, bins = BINS, density=True, rwidth=0.9, alpha=0.8, color='g', label='Sample Mean T-score Density (n=50)')
        axes[2,1].plot(T_X_03, T_PDF_3, alpha=1.0, color='black', linewidth=2.0)
        axes[2,1].plot(T_X_50, T_PDF_49, alpha=1.0, color='purple', linewidth=2.0)

        axes[2,1].text(-7, 0.35, f'T(-1, 1, df=2)')
        axes[2,1].text(2, 0.40, f'T(1, 1, df=49)')

        # axes[2,1].set_xlim(-6, 6)
        ####### Legends, Titles, Labels #######

        axes[2,0].grid(axis='both', linestyle='--', color='0.95')
        axes[2,0].set_xlabel('sample mean')
        axes[2,0].set_ylabel('density')
        axes[2,0].set_title('Density Plots for sample means (sample sizes n=3 and n=50)')

        axes[2,1].grid(axis='both', linestyle='--', color='0.95')
        axes[2,1].set_xlabel('sample mean')
        axes[2,1].set_ylabel('density')
        axes[2,1].set_title('Density Plots for sample mean t-scores (sample sizes n=3 and n=50)')
        
        axes[2,0].legend(loc="upper right")
        axes[2,1].legend(loc="upper left")

        ((i % 20 == 0) or (i == 999)) and plt.tight_layout()

    # pause the plot for 0.01s before next point is shown 
    # plt.pause(0.5 if i < 100 else 0.0001) 
    (i < 100) and plt.pause(0.05)

print("end")

plt.tight_layout()
plt.show()

