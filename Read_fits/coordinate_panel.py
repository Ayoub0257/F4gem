"""Visible coordinate inspector and manual celestial-location controls."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .coordinates import (
    COORDINATE_FORMAT_BOTH,
    COORDINATE_FORMAT_OPTIONS,
    parse_dec_input,
    parse_ra_input,
)


@dataclass(frozen=True)
class CoordinateMarker:
    """One user-requested celestial point or coordinate line."""

    marker_id: str
    label: str
    ra_deg: float | None
    dec_deg: float | None

    @property
    def kind(self) -> str:
        if self.ra_deg is not None and self.dec_deg is not None:
            return "point"
        if self.ra_deg is not None:
            return "ra_line"
        return "dec_line"


class CoordinatePanel(QWidget):
    """Always-visible cursor readout and manual RA/Dec locator."""

    format_changed = pyqtSignal(str)
    marker_requested = pyqtSignal(object)
    marker_removed = pyqtSignal(str)
    clear_markers_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setMinimumWidth(285)
        self.setMaximumWidth(430)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        title = QLabel("Coordinate Inspector")
        title.setStyleSheet("font-weight: 700; font-size: 14px;")
        root.addWidget(title)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("RA/Dec format:"))
        self.format_combo = QComboBox()
        for visible_name, value in COORDINATE_FORMAT_OPTIONS:
            self.format_combo.addItem(visible_name, value)
        self.format_combo.setCurrentIndex(0)
        self.format_combo.currentIndexChanged.connect(self._emit_format_changed)
        format_row.addWidget(self.format_combo, 1)
        root.addLayout(format_row)

        cursor_group = QGroupBox("Cursor")
        cursor_form = QFormLayout(cursor_group)
        cursor_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.pixel_value_label = QLabel("—")
        self.pixel_value_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.science_value_label = QLabel("—")
        self.science_value_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.world_value_label = QLabel(
            "Move the cursor over the image to inspect coordinates."
        )
        self.world_value_label.setWordWrap(True)
        self.world_value_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.backend_label = QLabel("—")
        self.backend_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        cursor_form.addRow("Pixel:", self.pixel_value_label)
        cursor_form.addRow("Value:", self.science_value_label)
        cursor_form.addRow("World:", self.world_value_label)
        cursor_form.addRow("Backend:", self.backend_label)
        root.addWidget(cursor_group)

        locator_group = QGroupBox("Locate celestial coordinate")
        locator_layout = QVBoxLayout(locator_group)

        self.frame_label = QLabel("Frame: —")
        self.frame_label.setWordWrap(True)
        locator_layout.addWidget(self.frame_label)

        locator_form = QFormLayout()
        self.ra_input = QLineEdit()
        self.ra_input.setPlaceholderText("12:34:56.7 or 188.73625")
        self.dec_input = QLineEdit()
        self.dec_input.setPlaceholderText("-24:18:42 or -24.31167")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Optional label")
        locator_form.addRow("RA:", self.ra_input)
        locator_form.addRow("Dec:", self.dec_input)
        locator_form.addRow("Name:", self.name_input)
        locator_layout.addLayout(locator_form)

        hint = QLabel(
            "Enter RA and Dec for a point. Enter only RA or only Dec to draw "
            "the corresponding coordinate line. Pure numbers are interpreted "
            "as decimal degrees."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        locator_layout.addWidget(hint)

        button_row = QHBoxLayout()
        self.locate_button = QPushButton("Locate")
        self.locate_button.clicked.connect(self._request_marker)
        self.clear_button = QPushButton("Clear all")
        self.clear_button.clicked.connect(self._clear_markers)
        button_row.addWidget(self.locate_button)
        button_row.addWidget(self.clear_button)
        locator_layout.addLayout(button_row)

        self.locator_status = QLabel("")
        self.locator_status.setWordWrap(True)
        self.locator_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        locator_layout.addWidget(self.locator_status)

        self.marker_list = QListWidget()
        self.marker_list.setMinimumHeight(100)
        locator_layout.addWidget(self.marker_list)

        self.remove_button = QPushButton("Remove selected")
        self.remove_button.clicked.connect(self._remove_selected)
        locator_layout.addWidget(self.remove_button)

        root.addWidget(locator_group)
        root.addStretch(1)

        self._locator_group = locator_group
        self.set_image_context(
            world_available=False,
            celestial_available=False,
            frame_name="Unknown",
        )

    @property
    def coordinate_format(self) -> str:
        value = self.format_combo.currentData()
        return str(value or COORDINATE_FORMAT_BOTH)

    def set_image_context(
        self,
        *,
        world_available: bool,
        celestial_available: bool,
        frame_name: str,
    ) -> None:
        self.backend_label.setText("Ready" if world_available else "Pixel only")
        self.frame_label.setText(f"Frame: {frame_name or 'Unknown'}")
        self._locator_group.setEnabled(bool(celestial_available))
        if not celestial_available:
            self.locator_status.setText(
                "Manual RA/Dec location requires a celestial WCS with both RA and Dec."
            )
        else:
            self.locator_status.setText("")

    def show_cursor_prompt(self) -> None:
        self.pixel_value_label.setText("—")
        self.science_value_label.setText("—")
        self.world_value_label.setText(
            "Move the cursor over the image to inspect coordinates."
        )

    def set_cursor_readout(
        self,
        *,
        x: int,
        y: int,
        value_text: str,
        world_lines: list[str],
        backend: str,
    ) -> None:
        self.pixel_value_label.setText(f"x={int(x)}, y={int(y)}")
        self.science_value_label.setText(value_text)
        self.world_value_label.setText("\n".join(world_lines) if world_lines else "—")
        self.backend_label.setText(str(backend or "Unknown"))

    def set_locator_status(self, message: str, *, is_error: bool = False) -> None:
        self.locator_status.setText(str(message or ""))
        if is_error:
            self.locator_status.setStyleSheet("color: #c62828;")
        else:
            self.locator_status.setStyleSheet("")

    def replace_marker_list(self, markers: list[CoordinateMarker]) -> None:
        self.marker_list.clear()
        for marker in markers:
            self._append_marker_item(marker)

    def add_marker_record(self, marker: CoordinateMarker) -> None:
        self._append_marker_item(marker)
        self.ra_input.clear()
        self.dec_input.clear()
        self.name_input.clear()

    def remove_marker_record(self, marker_id: str) -> None:
        for row in range(self.marker_list.count()):
            item = self.marker_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == marker_id:
                self.marker_list.takeItem(row)
                break

    def clear_marker_records(self) -> None:
        self.marker_list.clear()

    def _append_marker_item(self, marker: CoordinateMarker) -> None:
        if marker.kind == "point":
            coordinate_text = f"RA={marker.ra_deg:.8f}°, Dec={marker.dec_deg:+.8f}°"
        elif marker.kind == "ra_line":
            coordinate_text = f"RA={marker.ra_deg:.8f}°"
        else:
            coordinate_text = f"Dec={marker.dec_deg:+.8f}°"

        item = QListWidgetItem(f"{marker.label} — {coordinate_text}")
        item.setData(Qt.ItemDataRole.UserRole, marker.marker_id)
        self.marker_list.addItem(item)

    def _emit_format_changed(self) -> None:
        self.format_changed.emit(self.coordinate_format)

    def _request_marker(self) -> None:
        ra_text = self.ra_input.text().strip()
        dec_text = self.dec_input.text().strip()

        if not ra_text and not dec_text:
            self.set_locator_status(
                "Enter RA, Dec, or both before pressing Locate.",
                is_error=True,
            )
            return

        try:
            ra_deg = parse_ra_input(ra_text) if ra_text else None
            dec_deg = parse_dec_input(dec_text) if dec_text else None
        except ValueError as exc:
            self.set_locator_status(str(exc), is_error=True)
            return

        label = self.name_input.text().strip()
        if not label:
            number = self.marker_list.count() + 1
            if ra_deg is not None and dec_deg is not None:
                label = f"Target {number}"
            elif ra_deg is not None:
                label = f"RA line {number}"
            else:
                label = f"Dec line {number}"

        marker = CoordinateMarker(
            marker_id=uuid4().hex,
            label=label,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
        )
        self.marker_requested.emit(marker)

    def _remove_selected(self) -> None:
        item = self.marker_list.currentItem()
        if item is None:
            self.set_locator_status("Select a marker to remove.", is_error=True)
            return

        marker_id = str(item.data(Qt.ItemDataRole.UserRole))
        self.marker_list.takeItem(self.marker_list.row(item))
        self.marker_removed.emit(marker_id)
        self.set_locator_status("Marker removed.")

    def _clear_markers(self) -> None:
        self.marker_list.clear()
        self.clear_markers_requested.emit()
        self.set_locator_status("All markers for this HDU were cleared.")
