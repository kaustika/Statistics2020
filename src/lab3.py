import numpy as np
import math
import matplotlib.pyplot as plt
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


def theoretical_edjes(sample):
    min = np.quantile(sample, 0.25) - 1.5 * (np.quantile(sample, 0.75) - np.quantile(sample, 0.25))
    max = np.quantile(sample, 0.75) + 1.5 * (np.quantile(sample, 0.75) - np.quantile(sample, 0.25))
    return min, max


def outlier_num(sample, min, max):
    outlier = 0
    for elem in sample:
        if elem < min or elem > max:
            outlier += 1
    return outlier


if __name__ == "__main__":

    """
        Сгенерировать выборки размером 20 и 100 элементов.
        Построить для них боксплот Тьюки.
    """
    for distr_name in distributions.keys():
        sample_20 = get_distribution(distr_name, 20)
        sample_100 = get_distribution(distr_name, 100)
        plt.boxplot((sample_20, sample_100), labels=["n = 20", "n = 100"])
        plt.ylabel("X")
        plt.title(distr_name)
        # plt.show()

    """
        Для каждого распределения определить долю выбросов экспериментально
        (сгенерировав выборку, соответствующую распределению 1000 раз, и вычислив среднюю долю выбросов)
        и сравнить с результатами, полученными теоретически.
    """
    header = ["Выборка", "Доля выбросов"]
    rows = []

    for distr_name in distributions.keys():
        outlier_20, outlier_100 = 0, 0
        for _ in range(1000):
            sample_20 = get_distribution(distr_name, 20)
            sample_100 = get_distribution(distr_name, 100)

            # теоретические нижняя и верхняя границы уса
            min_20, max_20 = theoretical_edjes(sample_20)
            min_100, max_100 = theoretical_edjes(sample_100)

            # подсчет выбросов
            outlier_20 += outlier_num(sample_20, min_20, max_20)
            outlier_100 += outlier_num(sample_100, min_100, max_100)

        outlier_20 /= 1000
        outlier_100 /= 1000

        rows.append([distr_name + ", n = 20", np.around(outlier_20 / 20, decimals=2)])
        rows.append([distr_name + ", n = 100", np.around(outlier_100 / 100, decimals=2)])

    print(tabulate(rows, header, tablefmt="latex"))
    print("\n")




