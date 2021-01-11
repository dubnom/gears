import math
from typing import Callable, Optional

from x7.geom.utils import t_range

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.axes import Axes
# from matplotlib import style; style.use('ggplot')

import tkinter as tk
from tkinter import ttk


class PlotViewer:
    def __init__(self, update_func: Callable[[Axes], None], fps=60.0):
        self.update_func = update_func or self.update_example
        self.delay_ms = round(1000/fps)
        self.root = tk.Tk()

        fig = Figure(figsize=(5, 5), dpi=100)
        self.ax: Axes = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # toolbar = NavigationToolbar2Tk(self.canvas, self)
        # toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.do_update()

    def mainloop(self):
        self.root.mainloop()

    def do_update(self):
        self.ax.clear()
        self.update_func(self.ax)
        self.canvas.draw()
        self.root.after(self.delay_ms, self.do_update)

    @staticmethod
    def update_example():
        offset = [0]

        def moving_sine(ax: Axes):
            off = offset[0]
            offset[0] += 0.1
            vals = [(t, math.sin(t+off)) for t in t_range(40, 0, math.tau)]
            ax.plot(*zip(*vals))
        return moving_sine


if __name__ == '__main__':
    app = PlotViewer(PlotViewer.update_example())
    app.mainloop()
