import numpy as np
from tabulate import tabulate
import scipy.stats as stats

start_border, end_border = -1.5, 1.5
sample_size = 100
alpha = 0.05
p = 1 - alpha
k = 7


def MLE(sample):
    """
        Calculates maximum likelihood estimations
        of μ, σ for normally distributed variates.
    """
    mu_ml = np.mean(sample)
    sigma_ml = np.std(sample)
    print("mu_ml = ", np.around(mu_ml, decimals=2),
          " sigma_ml=", np.around(sigma_ml, decimals=2))
    return mu_ml, sigma_ml


def quantileChi2(sample, mu, sigma):
    """
        Decides whether to accept the hypothesis that N(mu, sigma)
        describes the given sample using chi2 criteria.
        Calculates sample value of chi2 statistic for a given sample.
    """
    hypothesis = lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)

    borders = np.linspace(start_border, end_border, num=k - 1)  # 𝑎0 = −∞, 𝑎𝑘 = +∞

    probabilities = np.array(hypothesis(start_border))
    quantities = np.array(len(sample[sample < start_border]))

    for i in range(k - 2):
        p_i = hypothesis(borders[i + 1]) - hypothesis(borders[i])
        probabilities = np.append(probabilities, p_i)
        n_i = len(sample[(sample < borders[i + 1]) & (sample >= borders[i])])
        quantities = np.append(quantities, n_i)

    probabilities = np.append(probabilities, 1 - hypothesis(end_border))
    quantities = np.append(quantities, len(sample[sample >= end_border]))

    chi2 = np.divide(
        np.multiply(
            (quantities - sample_size * probabilities),
            (quantities - sample_size * probabilities)
        ),
        probabilities * sample_size
    )

    quantile = stats.chi2.ppf(p, k - 1)
    isAccepted = True if quantile > np.sum(chi2) else False
    return chi2, isAccepted, borders, probabilities, quantities


def buildTable(chi2, borders, probabilities, quantities):
    """
        Builds chi2 calculation table.
    """
    headers = ["$i$", "$\\Delta_i = [a_{i-1}, a_i)$", "$n_i$", "$p_i$",
               "$np_i$", "$n_i - np_i$", "$(n_i - np_i)^2/np_i$"]
    rows = []
    for i in range(0, len(quantities)):
        if i == 0:
            limits = ["$-\infty$", np.around(borders[0], decimals=2)]
        elif i == len(quantities) - 1:
            limits = [np.around(borders[-1], decimals=2), "$\infty$"]
        else:
            limits = [np.around(borders[i - 1], decimals=2), np.around(borders[i], decimals=2)]
        rows.append(
            [i + 1,
             limits,
             quantities[i],
             np.around(probabilities[i], decimals=4),
             np.around(probabilities[i] * sample_size, decimals=2),
             np.around(quantities[i] - sample_size * probabilities[i], decimals=2),
             np.around(chi2[i], decimals=2)]
        )
    rows.append(["\\sum", "--", np.sum(quantities), np.around(np.sum(probabilities), decimals=4),
                 np.around(np.sum(probabilities * sample_size), decimals=2),
                 -np.around(np.sum(quantities - sample_size * probabilities), decimals=2),
                 np.around(np.sum(chi2), decimals=2)]
    )
    return tabulate(rows, headers, tablefmt="latex_raw")


if __name__ == '__main__':
    normal_sample = np.random.normal(0, 1, size=sample_size)
    mu_ml, sigma_ml = MLE(normal_sample)
    chi2, isAccepted, borders, probabilities, quantities = quantileChi2(normal_sample, mu_ml, sigma_ml)
    print(buildTable(chi2, borders, probabilities, quantities))
    print("Hypothesis Accepted!") if isAccepted else print("Hypothesis Not Accepted!")
