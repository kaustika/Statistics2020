import math
from scipy.special import factorial
import numpy as np  # стандартный алиас для numpy
import matplotlib.pyplot as plt  # стандартный алиас для pyplot


distributions = {
     'Normal': lambda num: np.random.normal(0, 1, num),
     'Cauchy': lambda num: np.random.standard_cauchy(num),
     'Laplace': lambda num: np.random.laplace(0, math.sqrt(2) / 2, num),
     'Poisson': lambda num: np.random.poisson(10, num),
     'Uniform': lambda num: np.random.uniform(-math.sqrt(3), math.sqrt(3), num)
}


pdfs = {
     'Normal': lambda x: (1 / (np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / (2 ** 2))),
     'Cauchy': lambda x: 1 / (np.pi * (x * x + 1)),
     'Laplace': lambda x: np.exp(-abs(x) / (1. / np.sqrt(2))) / (2. * (1. / np.sqrt(2))),
     'Poisson': lambda x: np.exp(-10)*np.power(10, x)/factorial(x),
     'Uniform': lambda x: [(1 / (2 * np.sqrt(3)) if np.fabs(x_k) <= np.sqrt(3) else 0) for x_k in x]
}


def get_distribution(distr_name, num):
    return distributions.get(distr_name)(num)


def get_pdf(distr_name, x):
    return pdfs.get(distr_name)(x)


for disrt_name in distributions.keys():
    # Здесь все как обычно
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(disrt_name)

    nrows = 1
    ncols = 3

    sizes = [10, 50, 1000]

    for a in range(1, nrows + 1):
        for b in range(1, ncols + 1):
            # Выбираем ячейку
            ax = fig.add_subplot(nrows, ncols, (a - 1) * ncols + b)
            # Строим гистограмму
            s = get_distribution(disrt_name, sizes[b-1])
            ax.hist(s, normed=True, facecolor='grey', edgecolor='black')
            # Строим график плотности
            x = np.arange(-8., 8., .01)
            pdf = get_pdf(disrt_name, x)
            ax.plot(x, pdf)

            # Делаем подписи
            ax.set_title(disrt_name + 'Distribution, n = ' + str(sizes[b-1]))

    plt.show()