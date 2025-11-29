import numpy as np
from scipy import stats

sample_data = [10, 12, 13, 15, 18, 20, 22, 25]
population_mean = 16

t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)
print(f"One-sample t-test: t-statistic = {t_stat}, p-value = {p_value}")

sample1 = [14, 15, 16, 17, 18, 19, 20]
sample2 = [22, 23, 24, 25, 26, 27, 28]

t_stat, p_value = stats.ttest_ind(sample1, sample2)
print(f"Two-sample t-test: t-statistic = {t_stat}, p-value = {p_value}")

before = [10, 12, 14, 16, 18]
after = [12, 14, 16, 18, 20]

t_stat, p_value = stats.ttest_rel(before, after)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value}")
