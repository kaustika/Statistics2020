import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


distributions = {
     'Normal': lambda num: np.random.normal(0, 1, num),
     'Cauchy': lambda num: np.random.standard_cauchy(num),
     'Laplace': lambda num: np.random.laplace(0, math.sqrt(2) / 2, num),
     'Poisson': lambda num: np.random.poisson(10, num),
     'Uniform': lambda num: np.random.uniform(-math.sqrt(3), 2 * math.sqrt(3), num)
}


def get_distribution(distr_name, num):
    return distributions.get(distr_name)(num)


cdfs = {
     'Normal': lambda x: stats.norm.cdf(x),
     'Cauchy': lambda x: stats.cauchy.cdf(x),
     'Laplace': lambda x: stats.laplace.cdf(x),
     'Poisson': lambda x: stats.poisson.cdf(x, 10),
     'Uniform': lambda x: stats.uniform.cdf(x)
}


def get_cdf(distr_name, x):
    return cdfs.get(distr_name)(x)


pdfs = {
     'Normal': lambda x: stats.norm.pdf(x, 0, 1),
     'Cauchy': lambda x: stats.cauchy.pdf(x),
     'Laplace': lambda x: stats.laplace.pdf(x, 0, 1 / 2 ** 0.5),
     'Poisson': lambda k: stats.poisson.pmf(10, k),
     'Uniform': lambda x: stats.uniform.pdf(x, -math.sqrt(3), 2 * math.sqrt(3))
}


def get_pdf(distr_name, x):
    return pdfs.get(distr_name)(x)


if __name__ == '__main__':
    nums = [20, 60, 100]
    for distr_name in distributions.keys():
        a, b, step = (6, 14, 1) if distr_name == 'Poisson' else (-4, 4, 0.01)
        x_range = np.arange(a, b, step)
        """ Сгенерировать выборки размером 20, 60 и 100 элементов. """
        samples = []
        for num in nums:
            samples.append([elem for elem in get_distribution(distr_name, num) if elem >= a or elem <= b])

        """ Построить на них эмпирические функции распределения. """
        index = 1
        for sample in samples:
            plt.subplot(1, 3, index)
            plt.title(distr_name + ', n = ' + str(nums[index - 1]))
            if distr_name == 'Poisson' or distr_name == 'Uniform':
                plt.step(x_range, get_cdf(distr_name, x_range), color='orange', label='cdf')
            else:
                plt.plot(x_range, get_cdf(distr_name, x_range), color='orange', label='cdf')
            array_ex = np.linspace(a, b)
            ecdf = ECDF(sample)
            y = ecdf(array_ex)
            plt.step(array_ex, y, color='black', label='ecdf')
            plt.xlabel('x')
            plt.ylabel('(e)cdf')
            plt.legend(loc='lower right')
            plt.subplots_adjust(wspace=0.5)
            index += 1
        plt.savefig(distr_name + 'CDF.png', format='png')
        #plt.show()

        """ Построить ядерные оценки плотности распределения. """
        index = 1
        for sample in samples:
            kern_names = [r'$h = h_n/2$', r'$h = h_n$', r'$h = 2 * h_n$']
            fig, ax = plt.subplots(1, 3)
            plt.subplots_adjust(wspace=0.5)
            i = 0
            for factor in [0.5, 1, 2]:
                kde = stats.gaussian_kde(sample, bw_method='silverman')
                h_n = kde.factor
                fig.suptitle(distr_name + ', n = ' + str(nums[index - 1]))
                ax[i].plot(x_range, get_pdf(distr_name, x_range), color='black', alpha=0.5, label='pdf')
                ax[i].set_title(kern_names[i])
                sns.kdeplot(sample, ax=ax[i], bw=h_n * factor, label='kde', color='orange')
                ax[i].set_xlabel('x')
                ax[i].set_ylabel('(kde)pdf')
                ax[i].set_ylim([0, 1])
                ax[i].set_xlim([a, b])
                ax[i].legend()
                i = i + 1
            plt.savefig(distr_name + 'KDE n = ' + str(nums[index - 1]) + '.png', format='png')
            #plt.show()
            index += 1