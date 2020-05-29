import numpy as np
import scipy.stats as stats


confidence_level = 0.95
alpha = 1 - confidence_level
nums = [20, 100]


def o(x):
    return np.around(x, decimals=2)


def muConfidenceInterval(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)
    n = len(sample)
    ci = sigma * stats.t.ppf(1 - alpha/2, n - 1) / np.sqrt(n - 1)
    return o(mu - ci), o(mu + ci)


def sigmaConfidenceInterval(sample):
    sigma = np.std(sample)
    n = len(sample)
    ci_start = sigma * np.sqrt(n / stats.chi2.ppf(1 - alpha/2, n - 1))
    ci_end = sigma * np.sqrt(n/stats.chi2.ppf(alpha/2, n - 1))
    return o(ci_start), o(ci_end)


def muConfidenceIntervalAsymptotic(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)
    n = len(sample)
    ci = sigma * stats.norm.ppf(1 - alpha/2) / np.sqrt(n)
    return o(mu - ci), o(mu + ci)


def sigmaConfidenceIntervalAsymptotic(sample):
    sigma = np.std(sample)
    n = len(sample)
    m4 = stats.moment(sample, 4)
    e = m4 / pow(float(sigma), 4) - 3
    U = stats.norm.ppf(1 - alpha/2) * np.sqrt((e + 2)/n)
    ci_start = sigma / np.sqrt(1 + U)
    ci_end = sigma / np.sqrt(1 - U)
    return o(ci_start), o(ci_end)


if __name__ == "__main__":
    for num in nums:
        sample = np.random.normal(0, 1, size=num)
        print('Size = ', num)
        print('Mean', muConfidenceInterval(sample))
        print('Variance', sigmaConfidenceInterval(sample))
        print('Asymptotic_mean', muConfidenceIntervalAsymptotic(sample))
        print('Asymptotic_variance', sigmaConfidenceIntervalAsymptotic(sample))
