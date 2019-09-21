import matplotlib.pyplot as plt
import numpy as np

class Line():
    def __init__(self, title="", line_type="default", x=[], y=[], down_border=0.97, up_border=1.03, c="red"):
        self.title = title
        self.line_type = line_type
        if len(x) == 0 and len(y) != 0:
            self._x = np.arange(len(y))
        self._y = y
        self.up_border = up_border
        self.down_border = down_border
        self.c = c

    @property
    def x(self):
        return np.array(self._x)

    @property
    def y(self):
        return np.array(self._y)
    
    @property
    def data(self):
        return np.array(self.x), np.array(self.y)

    def add_point(self, y, x):
        if x is None:
            x = len(self._x)
        self._x.append(x)
        self._y.append(y)

class DashBoard():
    _data = {}

    def init_graph(self, name, title, line_type="default", x=[], y=[], c="g"):
        self._data[name] = Line(title=title, line_type=line_type, x=x, y=y, c=c)

    def add_point(self, name, y, x=None):
        self._data[name].add_point(y, x)

    def make_plot(self):
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        for line in self._data.values():
            ax.plot(*line.data, label=line.title, c=line.c)
            if line.line_type == "baseline":
                ax.plot(line.x, line.y * line.up_border, c="green", linewidth=1)
                ax.plot(line.x, line.y * line.down_border, c="green", linewidth=1)

        ax.legend()
        plt.grid()
        plt.show()
        fig.savefig('plot.png')
