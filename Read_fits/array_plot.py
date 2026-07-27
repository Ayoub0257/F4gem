"""Simple plot for one-dimensional FITS image arrays."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class FitsArrayPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Array index")
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)

    def show_array(self, data, title: str = "1D FITS array") -> None:
        values = np.asanyarray(data).reshape(-1)
        if np.iscomplexobj(values):
            values = np.abs(values)
        values = np.asarray(values, dtype=float)

        self.plot_widget.clear()
        self.plot_widget.plot(np.arange(values.size), values)
        self.plot_widget.setTitle(title)
