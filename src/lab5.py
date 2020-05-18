import statistics
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy.stats as stats
import tabulate
from matplotlib.patches import Ellipse


nums = [20, 60, 100]
rhos = [0, 0.5, 0.9]
mean = [0, 0]
n_times = 1000


def get_rvs(num, cov, mixed):
    """ Generates a 2D array of size num of random variates:
        if not mixed -> with [0, 0] mean and cov matrix;
        if mixed -> following the lab task.
    """
    return stats.multivariate_normal.rvs(mean, cov, num) if not mixed \
        else 0.9 * stats.multivariate_normal.rvs(mean, [[1, 0.9], [0.9, 1]], num) \
             + 0.1 * stats.multivariate_normal.rvs(mean, [[10, -0.9], [-0.9, 10]], num)


def quadrant_coef(x, y):
    """ Calculates quadrant correlation coefficient of x and y. """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    n = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    n1, n2, n3, n4 = (0, 0, 0, 0)
    for i in range(n):
        if x[i] >= med_x and y[i] >= med_y:
            n1 += 1
        if x[i] < med_x and y[i] > med_y:
            n2 += 1
        if x[i] <= med_x and y[i] <= med_y:
            n3 += 1
        if x[i] > med_x and y[i] < med_y:
            n4 += 1
    return ((n1 + n3) - (n2 + n4)) / n


def accumulate_coefs(num, cov, mixed):
    """ Accumulates coefs calculating them n_times for num-sized sample. """
    pearson_coefs = []
    spearman_coefs = []
    quadrant_coefs = []
    for _ in range(n_times):
        sample = get_rvs(num, cov, mixed)
        x, y = sample[:, 0], sample[:, 1]
        pearson_coefs.append(stats.pearsonr(x, y)[0])
        spearman_coefs.append(stats.spearmanr(x, y)[0])
        quadrant_coefs.append(quadrant_coef(x, y))
    return pearson_coefs, spearman_coefs, quadrant_coefs


def build_table(num, cov, mixed):
    """ Builds table with characteristics of sets of coefs treated as random variates. """
    pearson, spearman, quadrant = accumulate_coefs(num, cov, mixed)
    p = np.median(pearson)
    s = np.median(spearman)
    q = np.median(quadrant)
    rows = []
    rows.append(['$E(z)$', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    p = np.median([pow(pearson[k], 2) for k in range(n_times)])
    s = np.median([pow(spearman[k], 2) for k in range(n_times)])
    q = np.median([pow(quadrant[k], 2) for k in range(n_times)])
    rows.append(['$E(z^2)$', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    p = statistics.variance(pearson)
    s = statistics.variance(spearman)
    q = statistics.variance(quadrant)
    rows.append(['$D(z)$', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    return rows


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """ Creates a plot of the covariance confidence ellipse of x and y. """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_ellipses(samples):
    """ For each sample in samples of normally distributed data plots its confidence ellipse and the data itself. """
    num = len(samples[0])
    fig, ax = plt.subplots(1, len(samples))
    fig.suptitle("n = " + str(num))
    titles = ['$ \\rho = 0$', '$\\rho = 0.5 $', '$ \\rho = 0.9$']
    i = 0
    for sample in samples:
        x = sample[:, 0]
        y = sample[:, 1]
        ax[i].scatter(x, y, c='black', s=3)
        confidence_ellipse(x, y, ax[i], edgecolor='orange')
        ax[i].scatter(np.mean(x), np.mean(y), c='black', s=3)
        ax[i].set_title(titles[i])
        i += 1
    plt.savefig(
        "Ellipse n = " + str(num) + ".png",
        format='png'
    )


def main():
    for num in nums:
        print("n = ", num)
        table = []
        for rho in rhos:
            cov = [[1.0, rho], [rho, 1.0]]
            extension_table = build_table(num, cov, False)
            title_row = ["$\\rho$ = " + str(rho), '$r$', '$r_S$', '$r_Q$']
            table.append(title_row)
            table.extend(extension_table)
        print(tabulate.tabulate(
            table,
            headers=[],
            tablefmt="latex_raw")
        )
    table_mixed = []
    for num in nums:
        extension_table_mixed = build_table(num, None, True)
        title_row = ["$n = " + str(num) + "$", '$r$', '$r_S$', '$r_Q$']
        table_mixed.append(title_row)
        table_mixed.extend(extension_table_mixed)
    print(tabulate.tabulate(
        table_mixed,
        headers=[],
        tablefmt="latex_raw")
    )
    samples = []
    for num in nums:
        for rho in rhos:
            samples.append(get_rvs(num, [[1.0, rho], [rho, 1.0]], False))
        plot_ellipses(samples)
        samples = []


if __name__ == '__main__':
    main()
