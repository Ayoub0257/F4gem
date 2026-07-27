"""Compact HDU selector shown above every FITS data view."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QWidget

from .fits_document import HDUDescriptor


class HDUSelector(QWidget):
    hdu_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._descriptors: dict[int, HDUDescriptor] = {}

        self.title_label = QLabel("HDU:")
        self.combo = QComboBox()
        self.combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.details_label = QLabel("No FITS file loaded")
        self.details_label.setMinimumWidth(220)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.addWidget(self.title_label)
        layout.addWidget(self.combo, 1)
        layout.addWidget(self.details_label)

        self.combo.currentIndexChanged.connect(self._on_combo_changed)

    def set_descriptors(
        self,
        descriptors: tuple[HDUDescriptor, ...],
        preferred_index: int,
    ) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        self._descriptors = {item.index: item for item in descriptors}

        preferred_combo_index = 0
        for combo_index, descriptor in enumerate(descriptors):
            self.combo.addItem(self._combo_text(descriptor), descriptor.index)
            if descriptor.index == preferred_index:
                preferred_combo_index = combo_index

        if descriptors:
            self.combo.setCurrentIndex(preferred_combo_index)
            self._update_details(preferred_index)
        else:
            self.details_label.setText("No HDUs")

        self.combo.blockSignals(False)

    def selected_hdu_index(self) -> int | None:
        value = self.combo.currentData()
        return int(value) if value is not None else None

    def select_hdu(self, hdu_index: int) -> None:
        combo_index = self.combo.findData(hdu_index)
        if combo_index >= 0:
            self.combo.setCurrentIndex(combo_index)

    def _on_combo_changed(self, combo_index: int) -> None:
        if combo_index < 0:
            return
        hdu_index = self.combo.itemData(combo_index)
        if hdu_index is None:
            return
        hdu_index = int(hdu_index)
        self._update_details(hdu_index)
        self.hdu_selected.emit(hdu_index)

    def _update_details(self, hdu_index: int) -> None:
        descriptor = self._descriptors.get(hdu_index)
        if descriptor is None:
            self.details_label.clear()
            return

        dtype = f" | {descriptor.dtype}" if descriptor.dtype else ""
        self.details_label.setText(
            f"{descriptor.hdu_type} | {descriptor.dimension_label}{dtype}"
        )

    @staticmethod
    def _combo_text(descriptor: HDUDescriptor) -> str:
        category_names = {
            "image_2d": "2D image",
            "image_nd": "N-D image",
            "table": "table",
            "array_1d": "1D array",
            "empty": "header only",
            "unsupported": "unsupported",
        }
        category = category_names.get(descriptor.category, descriptor.category)
        return (
            f"[{descriptor.index}] {descriptor.name} — "
            f"{category} ({descriptor.dimension_label})"
        )
