import numpy as np  # стандартный алиас для numpy
import matplotlib.pyplot as plt  # стандартный алиас для pyplot
from scipy.special import factorial


nrows = 1
ncols = 3
sizes = [10, 50, 1000]

# Здесь все как обычно
fig = plt.figure(figsize=(16, 20))
fig.suptitle('Poisson Distribution')

for a in range(1, nrows + 1):
    for b in range(1, ncols + 1):
        # Выбираем ячейку
        ax = fig.add_subplot(nrows, ncols, (a - 1) * ncols + b)
        # Строим гистограмму
        s = np.random.poisson(10, (sizes[b-1], 1))
        ax.hist(s, normed=True, facecolor='grey', edgecolor='black')
        # Строим график плотности
        x = np.arange(0., 25., 0.1)
        pdf = np.exp(-10)*np.power(10, x)/factorial(x)
        ax.plot(x, pdf)

        # Делаем подписи
        ax.set_title('Poisson Distribution, n = ' + str(sizes[b-1]))

plt.show()