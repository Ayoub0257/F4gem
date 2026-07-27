"""Dynamic controls for selecting planes from N-dimensional FITS images."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QWidget,
)


class AxisSelector(QWidget):
    indices_changed = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spin_boxes: list[QSpinBox] = []
        self._shape: tuple[int, ...] = ()

        self.summary_label = QLabel()
        self.form_layout = QFormLayout()
        self.form_layout.setContentsMargins(5, 3, 5, 3)

        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.summary_label)
        outer_layout.addLayout(self.form_layout, 1)

        self.hide()

    def configure(self, shape: tuple[int, ...], axis_names: list[str]) -> None:
        self.clear()
        self._shape = tuple(shape)
        leading_shape = self._shape[:-2]
        self.summary_label.setText(
            "Slice: " + " × ".join(str(size) for size in self._shape)
        )

        if not leading_shape:
            self.hide()
            return

        for axis, axis_size in enumerate(leading_shape):
            name = axis_names[axis] if axis < len(axis_names) else f"Axis {axis}"
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, max(axis_size - 1, 0))
            slider.setSingleStep(1)
            slider.setPageStep(max(1, axis_size // 10))

            spin_box = QSpinBox()
            spin_box.setRange(0, max(axis_size - 1, 0))
            spin_box.setSuffix(f" / {max(axis_size - 1, 0)}")

            slider.valueChanged.connect(spin_box.setValue)
            spin_box.valueChanged.connect(slider.setValue)
            spin_box.valueChanged.connect(self._emit_indices)

            row_layout.addWidget(slider, 1)
            row_layout.addWidget(spin_box)
            self.form_layout.addRow(f"{name}:", row_widget)
            self._spin_boxes.append(spin_box)

        self.show()

    def indices(self) -> tuple[int, ...]:
        return tuple(spin_box.value() for spin_box in self._spin_boxes)

    def clear(self) -> None:
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)
        self._spin_boxes.clear()
        self._shape = ()
        self.summary_label.clear()
        self.hide()

    def _emit_indices(self, _value: int) -> None:
        self.indices_changed.emit(self.indices())
