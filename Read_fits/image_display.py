import numpy as np
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from pyqtgraph import ImageView


class ImageDisplay(QWidget):
    """Display one two-dimensional numerical array."""

    def __init__(self):
        super().__init__()

        self.image_view = ImageView()

        # FITS/NumPy images use the standard (row, column) convention:
        #     array[y, x]
        #
        # PyQtGraph defaults to the older (column, row) convention, which makes
        # normal 2D FITS arrays appear transposed/sideways. Configure only this
        # ImageView to interpret the data as row-major. This changes the display
        # transform without rotating or copying the pixel array.
        self.image_view.getImageItem().setOpts(axisOrder="row-major")

        self.hist_visible = True
        self.toggle_button = QPushButton("Hide Histogram")
        self.toggle_button.clicked.connect(self.toggle_histogram)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_view)
        layout.addWidget(self.toggle_button)

    def toggle_histogram(self):
        """Show or hide the histogram bar."""
        if self.hist_visible:
            self.image_view.ui.histogram.hide()
            self.toggle_button.setText("Show Histogram")
        else:
            self.image_view.ui.histogram.show()
            self.toggle_button.setText("Hide Histogram")
        self.hist_visible = not self.hist_visible

    def show_image(self, data):
        """Display a 2D FITS plane with robust automatic contrast scaling."""
        if data is None:
            raise ValueError("The selected HDU contains no image data.")

        array = np.asanyarray(data)
        if array.ndim != 2:
            raise ValueError(
                f"ImageDisplay requires a 2D array; received shape {array.shape}."
            )

        if np.ma.isMaskedArray(array):
            array = array.filled(np.nan)

        if np.iscomplexobj(array):
            array = np.abs(array)

        try:
            array = np.asarray(array, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "The selected image does not contain numerical data."
            ) from exc

        finite = np.isfinite(array)
        if not finite.any():
            raise ValueError("The selected image contains no finite pixel values.")

        clean_array = array.copy()
        clean_array[~finite] = np.nan

        try:
            vmin, vmax = ZScaleInterval().get_limits(clean_array)
        except Exception:
            vmin, vmax = np.nanpercentile(clean_array, [1.0, 99.0])

        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin = float(np.nanmin(clean_array))
            vmax = float(np.nanmax(clean_array))

        if vmin == vmax:
            padding = abs(vmin) * 0.01 or 1.0
            vmin -= padding
            vmax += padding

        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
        normalized = np.ma.filled(norm(clean_array), 0.0)
        normalized = np.asarray(normalized, dtype=np.float32)

        self.image_view.setImage(
            normalized,
            autoRange=False,
            autoLevels=False,
        )
        self.image_view.setLevels(0, 1)