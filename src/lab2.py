import numpy as np
import scipy.stats as stats
import math
from tabulate import tabulate


distributions = {
     'Normal': lambda num: np.random.normal(0, 1, num),
     'Cauchy': lambda num: np.random.standard_cauchy(num),
     'Laplace': lambda num: np.random.laplace(0, math.sqrt(2) / 2, num),
     'Poisson': lambda num: np.random.poisson(10, num),
     'Uniform': lambda num: np.random.uniform(-math.sqrt(3), math.sqrt(3), num)
}


def get_distribution(distr_name, num):
    return distributions.get(distr_name)(num)


def get_quartil(sample_sorted, p):
    return np.percentile(sample_sorted, p*100)


len_list = [10, 100, 1000]
distrs = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']


for distr_name in distrs:
    print(f"Распределение {distr_name}")

    field_names = ["Characteristic", "Mean", "Median", "z_R", "z_Q", "z_tr"]
    rows = []
    for d_num in len_list:
        mean = []
        med = []
        z_R = []
        z_Q = []
        z_tr = []
        for _ in range(1000):
            sample_d = get_distribution(distr_name, d_num)
            sample_d_sorted = np.sort(sample_d)
            mean.append(np.mean(sample_d))
            med.append(np.median(sample_d))
            z_R.append((sample_d_sorted[0] + sample_d_sorted[-1]) / 2)
            z_Q.append((get_quartil(sample_d, 0.25) + get_quartil(sample_d, 0.75)) / 2)
            z_tr.append(stats.trim_mean(sample_d, 0.25))
        rows.append([distr_name + " E(z) " + str(d_num),
                          np.around(np.mean(mean), decimals=6),
                          np.around(np.mean(med), decimals=6),
                          np.around(np.mean(z_R), decimals=6),
                          np.around(np.mean(z_Q), decimals=6),
                          np.around(np.mean(z_tr), decimals=6)])
        rows.append([distr_name + " D(z) " + str(d_num),
                          np.around(np.std(mean) * np.std(mean), decimals=6),
                          np.around(np.std(med) * np.std(med), decimals=6),
                          np.around(np.std(z_R) * np.std(z_R), decimals=6),
                          np.around(np.std(z_Q) * np.std(z_Q), decimals=6),
                          np.around(np.std(z_tr) * np.std(z_tr), decimals=6)])
    table = rows
    print(tabulate(table, field_names, tablefmt="latex"))