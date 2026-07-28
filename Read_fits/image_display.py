"""Two-dimensional FITS image display and contrast controls.

The scientific array is retained separately from the float32 display buffer.
Contrast, stretch, histogram adjustments, and overlays therefore never modify
or replace the original FITS values used by cursor readout and later analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from astropy.visualization import (
    AsinhStretch,
    LinearStretch,
    LogStretch,
    SqrtStretch,
    ZScaleInterval,
)
from pyqtgraph import ImageView, SignalProxy


class ImageDisplay(QWidget):
    """Display one 2D numerical array with bounded-memory contrast handling."""

    pixel_hovered = pyqtSignal(int, int, object)
    cursor_left_image = pyqtSignal()

    PRESET_ZSCALE = "zscale"
    PRESET_PERCENT_995 = "percent_995"
    PRESET_PERCENT_99 = "percent_99"
    PRESET_FAINT = "faint"
    PRESET_FULL = "full"

    STRETCH_LINEAR = "linear"
    STRETCH_ASINH = "asinh"
    STRETCH_LOG = "log"
    STRETCH_SQRT = "sqrt"

    def __init__(self):
        super().__init__()

        self.image_view = ImageView()
        self.image_item = self.image_view.getImageItem()
        self.image_item.setOpts(axisOrder="row-major")

        self._histogram_widget = self.image_view.getHistogramWidget()
        self._histogram_widget.hide()
        self._histogram_visible = False

        self._science_data: np.ndarray | np.ma.MaskedArray | None = None
        self._display_source: np.ndarray | None = None
        self._raw_levels: tuple[float, float] | None = None
        self._display_levels: tuple[float, float] = (0.0, 1.0)
        self._last_image_shape: tuple[int, int] | None = None
        self._setting_histogram_levels = False

        self._last_hovered_pixel: tuple[int, int] | None = None
        self._cursor_was_inside = False
        self._overlays: dict[str, list[Any]] = {}

        scene = self.image_view.getView().scene()
        self._mouse_proxy = SignalProxy(
            scene.sigMouseMoved,
            rateLimit=25,
            slot=self._on_mouse_moved,
        )

        histogram_item = getattr(self._histogram_widget, "item", self._histogram_widget)
        histogram_item.sigLevelChangeFinished.connect(
            self._on_histogram_levels_finished
        )

        self.auto_button = QPushButton("Auto")
        self.auto_button.setToolTip(
            "Recalculate black and white levels for the current image."
        )
        self.auto_button.clicked.connect(self.auto_contrast)

        self.lock_levels_checkbox = QCheckBox("Lock levels")
        self.lock_levels_checkbox.setToolTip(
            "Keep the same physical contrast limits while moving through cube slices."
        )

        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Automatic contrast interval")
        self.preset_combo.addItem("ZScale", self.PRESET_ZSCALE)
        self.preset_combo.addItem("99.5%", self.PRESET_PERCENT_995)
        self.preset_combo.addItem("99%", self.PRESET_PERCENT_99)
        self.preset_combo.addItem("Faint features", self.PRESET_FAINT)
        self.preset_combo.addItem("Full range", self.PRESET_FULL)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        self.stretch_combo = QComboBox()
        self.stretch_combo.setToolTip("Display stretch")
        self.stretch_combo.addItem("Linear", self.STRETCH_LINEAR)
        self.stretch_combo.addItem("Asinh", self.STRETCH_ASINH)
        self.stretch_combo.addItem("Logarithmic", self.STRETCH_LOG)
        self.stretch_combo.addItem("Square root", self.STRETCH_SQRT)
        self.stretch_combo.currentIndexChanged.connect(self._on_stretch_changed)

        self.histogram_button = QPushButton("Histogram…")
        self.histogram_button.setToolTip(
            "Show the histogram only when manual display-level adjustment is needed."
        )
        self.histogram_button.clicked.connect(self.toggle_histogram)

        self.levels_label = QLabel("Levels: —")
        self.levels_label.setToolTip(
            "Physical data values used as the automatic black and white limits."
        )

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(4, 3, 4, 3)
        toolbar_layout.setSpacing(6)
        toolbar_layout.addWidget(self.auto_button)
        toolbar_layout.addWidget(self.lock_levels_checkbox)
        toolbar_layout.addWidget(QLabel("Preset:"))
        toolbar_layout.addWidget(self.preset_combo)
        toolbar_layout.addWidget(QLabel("Stretch:"))
        toolbar_layout.addWidget(self.stretch_combo)
        toolbar_layout.addWidget(self.histogram_button)
        toolbar_layout.addWidget(self.levels_label)
        toolbar_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.image_view, 1)

    @property
    def levels_locked(self) -> bool:
        return bool(self.lock_levels_checkbox.isChecked())

    @property
    def image_shape(self) -> tuple[int, int] | None:
        if self._science_data is None:
            return None
        return tuple(int(size) for size in self._science_data.shape)

    def show_image(self, data: Any) -> None:
        """Display a 2D FITS plane with a clear automatic first contrast."""

        if data is None:
            raise ValueError("The selected HDU contains no image data.")

        science_array = np.asanyarray(data)
        if science_array.ndim != 2:
            raise ValueError(
                f"ImageDisplay requires a 2D array; received shape "
                f"{science_array.shape}."
            )

        self._science_data = science_array
        self._last_hovered_pixel = None
        self._cursor_was_inside = False

        display_source: Any = science_array
        if np.ma.isMaskedArray(display_source):
            display_source = display_source.filled(np.nan)
        if np.iscomplexobj(display_source):
            display_source = np.abs(display_source)

        try:
            source = np.asarray(display_source, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            self._science_data = None
            raise TypeError(
                "The selected image does not contain numerical data."
            ) from exc

        finite = np.isfinite(source)
        if not finite.any():
            self._science_data = None
            raise ValueError("The selected image contains no finite pixel values.")

        # Keep one float32 rendering source. The original data are still held in
        # _science_data, and no full-size additional copy is retained after the
        # stretched display buffer has been sent to PyQtGraph.
        self._display_source = source

        shape = tuple(int(size) for size in source.shape)
        shape_changed = shape != self._last_image_shape
        self._last_image_shape = shape

        if self._raw_levels is None or not self.levels_locked:
            self._raw_levels = self._calculate_raw_levels(source)
            self._display_levels = (0.0, 1.0)

        self._render_display_buffer(
            auto_range=shape_changed,
            reset_histogram_levels=not self.levels_locked,
        )

    def reset_contrast_state(self) -> None:
        """Forget physical limits when switching to a different HDU/file.

        The lock checkbox remains unchanged, but locking is intentionally scoped
        to slices of the same image HDU rather than unrelated extensions.
        """

        self._raw_levels = None
        self._display_levels = (0.0, 1.0)
        self._last_image_shape = None
        self.levels_label.setText("Levels: —")

    def auto_contrast(self) -> None:
        """Recalculate physical levels for the current plane."""

        if self._display_source is None:
            return
        self._raw_levels = self._calculate_raw_levels(self._display_source)
        self._display_levels = (0.0, 1.0)
        self._render_display_buffer(
            auto_range=False,
            reset_histogram_levels=True,
        )

    def toggle_histogram(self) -> None:
        """Show the manual histogram only when explicitly requested."""

        if self._histogram_visible:
            self._histogram_widget.hide()
            self.histogram_button.setText("Histogram…")
            self._histogram_visible = False
        else:
            self._histogram_widget.show()
            self.histogram_button.setText("Hide histogram")
            self._histogram_visible = True

    def science_value(self, x: int, y: int) -> Any:
        """Return the original science value at one integer image pixel."""

        if self._science_data is None:
            raise RuntimeError("No image is currently displayed.")

        height, width = self._science_data.shape
        if not 0 <= x < width or not 0 <= y < height:
            raise IndexError(
                f"Pixel (x={x}, y={y}) is outside image shape "
                f"{self._science_data.shape}."
            )
        return self._science_data[y, x]

    def set_point_overlay(
        self,
        marker_id: str,
        *,
        x: float,
        y: float,
        label: str,
    ) -> None:
        """Draw or replace one non-destructive labelled point marker."""

        self.remove_overlay(marker_id)
        pen = pg.mkPen((255, 215, 0), width=2)
        scatter = pg.ScatterPlotItem(
            [float(x)],
            [float(y)],
            symbol="+",
            size=18,
            pen=pen,
            brush=None,
            pxMode=True,
        )
        text = pg.TextItem(
            text=str(label),
            color=(255, 215, 0),
            anchor=(0.0, 1.0),
        )
        text.setPos(float(x) + 2.0, float(y) - 2.0)

        view = self.image_view.getView()
        view.addItem(scatter)
        view.addItem(text)
        self._overlays[marker_id] = [scatter, text]

    def set_line_overlay(
        self,
        marker_id: str,
        *,
        x_values: np.ndarray,
        y_values: np.ndarray,
        label: str,
    ) -> None:
        """Draw or replace one sampled constant-coordinate curve."""

        self.remove_overlay(marker_id)
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        pen = pg.mkPen((255, 215, 0), width=1.5)
        line = pg.PlotDataItem(x=x_array, y=y_array, pen=pen, connect="finite")

        finite = np.flatnonzero(np.isfinite(x_array) & np.isfinite(y_array))
        items: list[Any] = [line]
        view = self.image_view.getView()
        view.addItem(line)

        if finite.size:
            first = int(finite[0])
            text = pg.TextItem(
                text=str(label),
                color=(255, 215, 0),
                anchor=(0.0, 1.0),
            )
            text.setPos(float(x_array[first]) + 2.0, float(y_array[first]) - 2.0)
            view.addItem(text)
            items.append(text)

        self._overlays[marker_id] = items

    def remove_overlay(self, marker_id: str) -> None:
        items = self._overlays.pop(marker_id, [])
        view = self.image_view.getView()
        for item in items:
            try:
                view.removeItem(item)
            except Exception:
                pass

    def clear_overlays(self) -> None:
        for marker_id in list(self._overlays):
            self.remove_overlay(marker_id)

    def _on_preset_changed(self) -> None:
        # A user-selected preset is an explicit request to recalculate the
        # current plane, even when level locking is enabled.
        self.auto_contrast()

    def _on_stretch_changed(self) -> None:
        if self._display_source is None or self._raw_levels is None:
            return
        self._display_levels = (0.0, 1.0)
        self._render_display_buffer(
            auto_range=False,
            reset_histogram_levels=True,
        )

    def _calculate_raw_levels(self, source: np.ndarray) -> tuple[float, float]:
        sample = self._finite_sample(source)
        preset = str(self.preset_combo.currentData() or self.PRESET_ZSCALE)

        if preset == self.PRESET_ZSCALE:
            try:
                vmin, vmax = ZScaleInterval().get_limits(sample)
            except Exception:
                vmin, vmax = np.nanpercentile(sample, [0.5, 99.5])
        elif preset == self.PRESET_PERCENT_995:
            vmin, vmax = np.nanpercentile(sample, [0.25, 99.75])
        elif preset == self.PRESET_PERCENT_99:
            vmin, vmax = np.nanpercentile(sample, [0.5, 99.5])
        elif preset == self.PRESET_FAINT:
            # Deliberately clip more of the bright tail to reveal faint
            # structure. This is a display preset, not a data operation.
            vmin, vmax = np.nanpercentile(sample, [1.0, 98.0])
        else:
            vmin = float(np.nanmin(sample))
            vmax = float(np.nanmax(sample))

        vmin = float(vmin)
        vmax = float(vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin = float(np.nanmin(sample))
            vmax = float(np.nanmax(sample))

        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            padding = abs(vmin) * 0.01 or 1.0
            vmin -= padding
            vmax += padding
        return vmin, vmax

    @staticmethod
    def _finite_sample(source: np.ndarray, maximum: int = 1_000_000) -> np.ndarray:
        """Return a bounded deterministic finite sample for percentile work."""

        flat = np.ravel(source)
        if flat.size > maximum:
            stride = max(1, flat.size // maximum)
            flat = flat[::stride][:maximum]
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            raise ValueError("The selected image contains no finite pixel values.")
        return finite

    def _render_display_buffer(
        self,
        *,
        auto_range: bool,
        reset_histogram_levels: bool,
    ) -> None:
        if self._display_source is None or self._raw_levels is None:
            return

        vmin, vmax = self._raw_levels
        scale = vmax - vmin

        # One float32 working buffer is enough for normalization and stretch.
        # Using NumPy/Astropy out parameters avoids retaining several full-size
        # temporary images while a large plane is being displayed.
        display_buffer = np.array(
            self._display_source,
            dtype=np.float32,
            copy=True,
            order="C",
        )
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            np.subtract(display_buffer, vmin, out=display_buffer)
            np.divide(display_buffer, scale, out=display_buffer)
        np.clip(display_buffer, 0.0, 1.0, out=display_buffer)

        stretch_name = str(
            self.stretch_combo.currentData() or self.STRETCH_LINEAR
        )
        if stretch_name == self.STRETCH_ASINH:
            stretch = AsinhStretch()
        elif stretch_name == self.STRETCH_LOG:
            stretch = LogStretch()
        elif stretch_name == self.STRETCH_SQRT:
            stretch = SqrtStretch()
        else:
            stretch = LinearStretch()

        try:
            stretched = stretch(display_buffer, clip=True, out=display_buffer)
        except TypeError:
            # Compatibility with older Astropy versions that did not expose
            # the optional out parameter on every stretch implementation.
            stretched = stretch(display_buffer, clip=True)
        if stretched is not display_buffer:
            display_buffer = np.asarray(stretched, dtype=np.float32)
        display_buffer[~np.isfinite(display_buffer)] = 0.0

        self.image_view.setImage(
            display_buffer,
            autoRange=bool(auto_range),
            autoLevels=False,
            autoHistogramRange=True,
        )

        if reset_histogram_levels:
            self._display_levels = (0.0, 1.0)
        self._set_histogram_levels(self._display_levels)
        self._update_levels_label()

    def _set_histogram_levels(self, levels: tuple[float, float]) -> None:
        low, high = (float(levels[0]), float(levels[1]))
        self._setting_histogram_levels = True
        try:
            self.image_view.setLevels(low, high)
            histogram_item = getattr(
                self._histogram_widget,
                "item",
                self._histogram_widget,
            )
            try:
                histogram_item.setHistogramRange(0.0, 1.0, padding=0.05)
            except TypeError:
                histogram_item.setHistogramRange(0.0, 1.0)
        finally:
            self._setting_histogram_levels = False

    def _on_histogram_levels_finished(self, *_args: Any) -> None:
        if self._setting_histogram_levels:
            return
        histogram_item = getattr(
            self._histogram_widget,
            "item",
            self._histogram_widget,
        )
        try:
            low, high = histogram_item.getLevels()
            low = float(low)
            high = float(high)
        except Exception:
            return
        if np.isfinite(low) and np.isfinite(high) and low != high:
            self._display_levels = (min(low, high), max(low, high))

    def _update_levels_label(self) -> None:
        if self._raw_levels is None:
            self.levels_label.setText("Levels: —")
            return
        vmin, vmax = self._raw_levels
        self.levels_label.setText(f"Levels: {vmin:.6g} → {vmax:.6g}")

    def _on_mouse_moved(self, event: Any) -> None:
        """Map a scene position to ``(x, y)`` and emit the original value."""

        if self._science_data is None:
            return

        if isinstance(event, (tuple, list)):
            if not event:
                return
            scene_position = event[0]
        else:
            scene_position = event

        if not self.image_item.sceneBoundingRect().contains(scene_position):
            self._mark_cursor_outside()
            return

        image_position = self.image_item.mapFromScene(scene_position)
        x = int(np.floor(image_position.x()))
        y = int(np.floor(image_position.y()))

        height, width = self._science_data.shape
        if not (0 <= x < width and 0 <= y < height):
            self._mark_cursor_outside()
            return

        self._cursor_was_inside = True
        pixel = (x, y)
        if pixel == self._last_hovered_pixel:
            return

        self._last_hovered_pixel = pixel
        self.pixel_hovered.emit(x, y, self._science_data[y, x])

    def _mark_cursor_outside(self) -> None:
        self._last_hovered_pixel = None
        if self._cursor_was_inside:
            self._cursor_was_inside = False
            self.cursor_left_image.emit()
