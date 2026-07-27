import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from pathlib import Path

class SpectrumPlotWidget(QWidget):
    """
    A widget to display a 1D spectrum using the powerful PyQtGraph library.
    """
    def __init__(self):
        super().__init__()
        # Configure the default look of pyqtgraph to match a standard UI
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # --- Create a toolbar ---
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 2, 5, 2)
        self.back_button = QPushButton("← Back to 2D Image")
        toolbar_layout.addWidget(self.back_button)
        toolbar_layout.addStretch()

        # --- Plot Widget ---
        self.plot_widget = pg.PlotWidget()
        self.setup_plot()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(self.plot_widget)

    def setup_plot(self):
        """Configure the plot's appearance, labels, and interactive features."""
        self.plot_widget.setLabel('bottom', "Wavelength", units='Å')
        self.plot_widget.setLabel('left', "Intensity", units='ADU')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setTitle("1D Extracted Spectrum")

    def plot_data(self, wavelengths, intensities, filename=""):
        """
        Clears any existing plot and draws the new spectral data.
        """
        self.plot_widget.clear()
        # Plot the data with a pen for styling
        pen = pg.mkPen(color=(0, 0, 200), width=1)
        self.plot_widget.plot(wavelengths, intensities, pen=pen)
        
        if filename:
            title = f"1D Spectrum - {Path(filename).name}"
            self.plot_widget.setTitle(title)