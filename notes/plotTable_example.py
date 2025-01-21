import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import random
from threading import Timer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


data = np.random.rand(4, 1)
data = data.reshape(2, 2)
fig, ax = plt.subplots()
table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=["0", "1"], rowLabels=["0", "1"])
ax.axis('off')
colors = [(1, 0, 0), (1, 1, 1), (0, 1, 0)]
cmap = LinearSegmentedColormap.from_list("RedWhiteGreen", colors, N=256)

def update_data(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = data[i, j] + random.randint(0, 100)/1000 - 0.05
    return data

def update_figure(t):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            cell = table[(i+1, j)]
            cell.get_text().set_text(str(data[i, j]))
            color = cmap(NormalizeData(data)[i, j])
            cell.set_facecolor(color)

    fig.canvas.draw_idle()

def periodic_update():
    update_data(data)
    Timer(0.1, periodic_update).start()

if __name__ == '__main__':
    periodic_update()

    ani = FuncAnimation(fig, update_figure, frames=100, interval=100)
    plt.show()