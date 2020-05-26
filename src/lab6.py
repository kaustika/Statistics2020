import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
from matplotlib import pyplot as plt

num_points = 20
bounds = -1.8, 2
step = 0.2
mu, sigma_squared = 0, 1
coefs = 2, 2
perturbations = [10, -10]


def leastSquares(x, y):
    b_ls = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    a_ls = np.mean(y) - b_ls * np.mean(x)
    return a_ls, b_ls


def leastModulo(x, y, initial_guess):
    functionToMinimize = lambda beta: np.sum(np.abs(y - beta[0] - beta[1] * x))
    result = opt.minimize(functionToMinimize, initial_guess)
    a_lm = result['x'][0]
    b_lm = result['x'][1]
    return a_lm, b_lm


def coefficientEstimates(x, y):
    a_ls, b_ls = leastSquares(x, y)
    a_lm, b_lm = leastModulo(x, y, np.array([a_ls, b_ls]))
    return a_ls, b_ls, a_lm, b_lm


def printEstimates(type, estimates):
    a_ls, b_ls, a_lm, b_lm = estimates
    print(type)
    print("Least Squares - МНК")
    print('a_ls = ' + str(np.around(a_ls, decimals=2)))
    print('b_ls = ' + str(np.around(b_ls, decimals=2)))
    print("Least Modulo - МНМ")
    print('a_lm = ' + str(np.around(a_lm, decimals=2)))
    print('b_lm = ' + str(np.around(b_lm, decimals=2)))


def plotRegression(x, y, type, estimates):
    a_ls, b_ls, a_lm, b_lm = estimates
    plt.scatter(x, y, label="Sample", edgecolor='navy')
    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Model', color='steelblue')
    plt.plot(x, x * (b_ls * np.ones(len(x))) + a_ls * np.ones(len(x)), label='Least Squares', color='turquoise')
    plt.plot(x, x * (b_lm * np.ones(len(x))) + a_lm * np.ones(len(x)), label='Least Modulo', color='blue')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.title(type)
    plt.savefig(type + '.png', format='png')
    plt.close()


def criteriaComparison(x, estimates):
    a_ls, b_ls, a_lm, b_lm = estimates
    model = lambda x: coefs[0] + coefs[1] * x
    # least squares criteria
    lsc = lambda x: a_ls + b_ls * x
    # least modulo criteria
    lmc = lambda x: a_lm + b_lm * x
    sum_ls, sum_lm = 0, 0
    for point in x:
        y_ls = lsc(point)
        y_lm = lmc(point)
        y_model = model(point)
        sum_ls += pow(y_model - y_ls, 2)
        sum_lm += pow(y_model - y_lm, 2)
    print("Least squares approximate better! - ", sum_ls, " < ", sum_lm) if sum_ls < sum_lm\
        else print("Least modulo approximates better! - ", sum_lm, " < ", sum_ls)


if __name__ == "__main__":
    x = np.linspace(bounds[0], bounds[1], num_points)
    y = coefs[0] + coefs[1] * x + stats.norm(0, 1).rvs(num_points)
    for type in ['Without perturbations', 'With perturbations']:
        estimates = coefficientEstimates(x, y)
        printEstimates(type, estimates)
        plotRegression(x, y, type, estimates)
        criteriaComparison(x, estimates)
        y[0] += perturbations[0]
        y[-1] += perturbations[1]
