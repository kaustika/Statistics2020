import numpy as np  # стандартный алиас для numpy
import matplotlib.pyplot as plt  # стандартный алиас для pyplot


# Здесь все как обычно
fig = plt.figure(figsize=(16, 20))
fig.suptitle('Normal(Gauss) Distribution')

nrows = 1
ncols = 3

sizes = [10, 50, 1000]

for a in range(1, nrows + 1):
    for b in range(1, ncols + 1):
        # Выбираем ячейку
        ax = fig.add_subplot(nrows, ncols, (a - 1) * ncols + b)
        # Строим гистограмму
        loc, scale = 0., 1.
        s = np.random.normal(loc, scale, sizes[b-1])
        ax.hist(s, normed=True, facecolor='grey', edgecolor='black')
        # Строим график плотности
        x = np.arange(-8., 8., .01)
        pdf = (1 / (scale * np.sqrt(2 * np.pi)) * np.exp(-(x - loc) ** 2 / (2 * scale ** 2)))
        ax.plot(x, pdf)

        # Делаем подписи
        ax.set_title('Normal(Gauss) Distribution, n = ' + str(sizes[b-1]))

plt.show()

