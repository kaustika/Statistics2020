import numpy as np  # стандартный алиас для numpy
import matplotlib.pyplot as plt  # стандартный алиас для pyplot

nrows = 1
ncols = 3
sizes = [10, 50, 1000]

# Здесь все как обычно
fig = plt.figure(figsize=(16, 20))
fig.suptitle('Cauchy Distribution')

for a in range(1, nrows + 1):
    for b in range(1, ncols + 1):
        # Выбираем ячейку
        ax = fig.add_subplot(nrows, ncols, (a - 1) * ncols + b)
        # Строим гистограмму
        s = np.random.standard_cauchy(sizes[b-1])
        ax.hist(s, normed=True, facecolor='grey', edgecolor='black')
        # Строим график плотности
        x = np.arange(-8., 8., .01)
        pdf = 1 / (np.pi * (x * x + 1))
        ax.plot(x, pdf)

        # Делаем подписи
        ax.set_title('Cauchy Distribution, n = ' + str(sizes[b-1]))

plt.show()