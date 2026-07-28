from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from Spectro.spectrum_plot import SpectrumPlotWidget
from .array_plot import FitsArrayPlot
from .axis_selector import AxisSelector
from .coordinate_panel import CoordinateMarker, CoordinatePanel
from .coordinates import (
    CoordinateResult,
    format_data_value,
    format_world_coordinate_lines,
)
from .drop_area import DropArea
from .file_loader import LoadFiles
from .fits_document import FitsDocument
from .GUI_utils import show_error
from .hdu_selector import HDUSelector
from .header_viewer import HeaderViewer
from .image_display import ImageDisplay
from .table_viewer import FitsTableViewer


class FitsLoaderWorker(QObject):
    """Open and inspect a FITS document outside the GUI thread."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            document = LoadFiles.open_fits_document(self.path)
            self.finished.emit(document)
        except Exception as exc:
            self.error.emit(str(exc))


class FitsViewer(QWidget):
    """Multi-HDU FITS viewer with WCS inspection and display controls."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Faraway//Spectroscopy")
        self.setAcceptDrops(True)

        self.path: str | None = None
        self.document: FitsDocument | None = None
        self.current_hdu_index: int | None = None
        self._header_visible = False
        self._pending_slice_indices: tuple[int, ...] = ()
        self.thread: QThread | None = None
        self.worker: FitsLoaderWorker | None = None

        # Markers are stored per HDU because two extensions may use different
        # coordinate systems or image geometries.
        self._coordinate_markers: dict[int, dict[str, CoordinateMarker]] = {}
        self._last_cursor_lookup: tuple[
            int,
            int,
            object,
            CoordinateResult,
            str | None,
        ] | None = None

        self._slice_timer = QTimer(self)
        self._slice_timer.setSingleShot(True)
        self._slice_timer.setInterval(60)
        self._slice_timer.timeout.connect(self._render_pending_slice)

        self._create_document_view()

        self.drop_area = DropArea()
        self.spectrum_plot = SpectrumPlotWidget()
        self.spectrum_plot.back_button.setText("← Back to FITS data")
        self.spectrum_plot.back_button.clicked.connect(self.show_current_fits_data)

        self.stack = QStackedLayout()
        self.stack.addWidget(self.drop_area)
        self.stack.addWidget(self.document_container)
        self.stack.addWidget(self.spectrum_plot)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(self.stack)

        self.drop_area.file_dropped.connect(self.display_from_drop)

    def _create_document_view(self) -> None:
        self.document_container = QWidget()
        container_layout = QVBoxLayout(self.document_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        self.file_label = QLabel("No FITS file loaded")
        self.file_label.setStyleSheet("font-weight: 600; padding: 4px 6px;")

        self.hdu_selector = HDUSelector()
        self.hdu_selector.hdu_selected.connect(self.on_hdu_selected)

        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(5, 2, 5, 2)

        self.header_button = QPushButton("Show Header")
        self.header_button.clicked.connect(self.toggle_header)
        self.header_button.setEnabled(False)

        self.show_1d_button = QPushButton("Show Extracted 1D Spectrum")
        self.show_1d_button.clicked.connect(self.show_1d_spectrum)
        self.show_1d_button.setEnabled(False)

        actions_layout.addWidget(self.header_button)
        actions_layout.addWidget(self.show_1d_button)
        actions_layout.addStretch()

        self.axis_selector = AxisSelector()
        self.axis_selector.indices_changed.connect(self.on_slice_indices_changed)

        self.image_display = ImageDisplay()
        self.image_display.pixel_hovered.connect(self.on_image_pixel_hovered)
        self.image_display.cursor_left_image.connect(self.on_image_cursor_left)

        self.coordinate_panel = CoordinatePanel()
        self.coordinate_panel.format_changed.connect(
            self.on_coordinate_format_changed
        )
        self.coordinate_panel.marker_requested.connect(
            self.on_coordinate_marker_requested
        )
        self.coordinate_panel.marker_removed.connect(
            self.on_coordinate_marker_removed
        )
        self.coordinate_panel.clear_markers_requested.connect(
            self.on_clear_coordinate_markers
        )

        self.image_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.image_splitter.setChildrenCollapsible(False)
        self.image_splitter.addWidget(self.image_display)
        self.image_splitter.addWidget(self.coordinate_panel)
        self.image_splitter.setStretchFactor(0, 1)
        self.image_splitter.setStretchFactor(1, 0)
        self.image_splitter.setSizes([900, 320])

        self.image_workspace = QWidget()
        image_workspace_layout = QVBoxLayout(self.image_workspace)
        image_workspace_layout.setContentsMargins(0, 0, 0, 0)
        image_workspace_layout.addWidget(self.image_splitter)

        self.table_viewer = FitsTableViewer()
        self.header_viewer = HeaderViewer()
        self.array_plot = FitsArrayPlot()

        self.message_view = QLabel()
        self.message_view.setWordWrap(True)
        self.message_view.setStyleSheet("padding: 20px;")

        self.data_stack = QStackedLayout()
        self.data_stack.addWidget(self.image_workspace)
        self.data_stack.addWidget(self.table_viewer)
        self.data_stack.addWidget(self.header_viewer)
        self.data_stack.addWidget(self.array_plot)
        self.data_stack.addWidget(self.message_view)

        data_widget = QWidget()
        data_widget.setLayout(self.data_stack)

        container_layout.addWidget(self.file_label)
        container_layout.addWidget(self.hdu_selector)
        container_layout.addWidget(actions_widget)
        container_layout.addWidget(self.axis_selector)
        container_layout.addWidget(data_widget, 1)

    def open_file_dialog(self):
        start_dir = str(Path(self.path).parent) if self.path else "."
        filename = LoadFiles.select_fits(parent=self, start_dir=start_dir)
        if filename:
            self._begin_loading(filename)

    def display_from_drop(self, path: str):
        self._begin_loading(path)

    def load_fits_from_path(self, path: str):
        self._begin_loading(path)

    def _begin_loading(self, path: str) -> None:
        if self.thread is not None and self.thread.isRunning():
            show_error(self, "A FITS file is already being loaded.")
            return

        if not path.lower().endswith((".fit", ".fits", ".fts")):
            show_error(self, "The selected file is not a FITS file.")
            return

        self.drop_area.set_text(f"Loading {Path(path).name}...")
        self.stack.setCurrentWidget(self.drop_area)

        self.thread = QThread(self)
        self.worker = FitsLoaderWorker(path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_fits_loaded)
        self.worker.error.connect(self.on_fits_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._on_loader_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _on_loader_thread_finished(self) -> None:
        self.thread = None
        self.worker = None

    def on_fits_loaded(self, document: FitsDocument) -> None:
        old_document = self.document
        self.document = document
        self.path = str(document.path)
        self.current_hdu_index = None
        self._header_visible = False
        self._pending_slice_indices = ()
        self._coordinate_markers.clear()
        self._last_cursor_lookup = None
        self.image_display.clear_overlays()
        self.coordinate_panel.clear_marker_records()

        if old_document is not None:
            old_document.close()

        preferred_index = document.preferred_hdu_index()
        self.file_label.setText(document.path.name)
        self.file_label.setToolTip(str(document.path))
        self.hdu_selector.set_descriptors(document.descriptors, preferred_index)
        self.header_button.setEnabled(True)
        self.show_1d_button.setEnabled(True)
        self.on_hdu_selected(preferred_index)

        self.stack.setCurrentWidget(self.document_container)
        self.drop_area.set_text("Drop a FITS file here...")

    def on_fits_error(self, message: str) -> None:
        show_error(parent=self, message=message)
        if self.document is not None:
            self.stack.setCurrentWidget(self.document_container)
        else:
            self.stack.setCurrentWidget(self.drop_area)
        self.drop_area.set_text("Drop a FITS file here...")

    def on_hdu_selected(self, hdu_index: int) -> None:
        if self.document is None:
            return

        self._slice_timer.stop()
        self.current_hdu_index = hdu_index
        self._header_visible = False
        self._last_cursor_lookup = None
        self.header_button.setText("Show Header")
        self.image_display.clear_overlays()
        self.image_display.reset_contrast_state()
        self.coordinate_panel.replace_marker_list(
            list(self._markers_for_current_hdu().values())
        )

        try:
            self._display_selected_hdu(reset_axes=True)
        except Exception as exc:
            self._show_data_error(hdu_index, exc)

    def _display_selected_hdu(self, reset_axes: bool) -> None:
        if self.document is None or self.current_hdu_index is None:
            return

        descriptor = self.document.descriptor(self.current_hdu_index)
        hdu = self.document.hdu(self.current_hdu_index)

        if descriptor.category == "image_2d":
            self.axis_selector.clear()
            self._pending_slice_indices = ()
            image = self.document.extract_2d_slice(self.current_hdu_index)
            self.image_display.show_image(image)
            self.data_stack.setCurrentWidget(self.image_workspace)
            self._configure_coordinate_panel()
            self._refresh_coordinate_overlays()

        elif descriptor.category == "image_nd":
            if reset_axes:
                axis_names = self.document.axis_names(self.current_hdu_index)
                assert descriptor.shape is not None
                self.axis_selector.configure(descriptor.shape, axis_names)
            self._pending_slice_indices = self.axis_selector.indices()
            self._render_pending_slice()

        elif descriptor.category == "table":
            self.axis_selector.clear()
            self._pending_slice_indices = ()
            self.table_viewer.set_hdu(hdu)
            self.data_stack.setCurrentWidget(self.table_viewer)

        elif descriptor.category == "array_1d":
            self.axis_selector.clear()
            self._pending_slice_indices = ()
            data = self.document.extract_1d_array(self.current_hdu_index)
            title = f"[{descriptor.index}] {descriptor.name}"
            self.array_plot.show_array(data, title=title)
            self.data_stack.setCurrentWidget(self.array_plot)

        elif descriptor.category == "empty":
            self.axis_selector.clear()
            self._pending_slice_indices = ()
            self._show_header()

        else:
            self.axis_selector.clear()
            self._pending_slice_indices = ()
            self.message_view.setText(
                "This HDU type is not currently displayable. "
                "Its header is still available through ‘Show Header’."
            )
            self.data_stack.setCurrentWidget(self.message_view)

    def on_slice_indices_changed(self, indices: tuple) -> None:
        self._pending_slice_indices = tuple(int(value) for value in indices)
        self._slice_timer.start()

    def _render_pending_slice(self) -> None:
        if self.document is None or self.current_hdu_index is None:
            return

        try:
            image = self.document.extract_2d_slice(
                self.current_hdu_index,
                self._pending_slice_indices,
            )
            self.image_display.show_image(image)
            self.data_stack.setCurrentWidget(self.image_workspace)
            self._configure_coordinate_panel()
            self._refresh_coordinate_overlays()
        except Exception as exc:
            self._show_data_error(self.current_hdu_index, exc)

    def _configure_coordinate_panel(self) -> None:
        if self.document is None or self.current_hdu_index is None:
            return

        provider = self.document.coordinate_provider(self.current_hdu_index)
        self.coordinate_panel.set_image_context(
            world_available=provider.backend != "pixel",
            celestial_available=provider.has_celestial_coordinates,
            frame_name=provider.frame_name,
        )
        self.coordinate_panel.show_cursor_prompt()
        self.coordinate_panel.replace_marker_list(
            list(self._markers_for_current_hdu().values())
        )
        self._last_cursor_lookup = None

    def on_image_pixel_hovered(self, x: int, y: int, value: object) -> None:
        """Update the visible coordinate inspector without modal errors."""

        if self.document is None or self.current_hdu_index is None:
            return

        try:
            result = self.document.world_at(
                self.current_hdu_index,
                x=x,
                y=y,
                leading_indices=self._current_leading_indices(),
            )
            data_unit = self.document.data_unit(self.current_hdu_index)
            self._last_cursor_lookup = (x, y, value, result, data_unit)
            self._display_cursor_lookup(
                x=x,
                y=y,
                value=value,
                result=result,
                data_unit=data_unit,
            )
        except Exception as exc:
            self.coordinate_panel.set_cursor_readout(
                x=x,
                y=y,
                value_text=str(value),
                world_lines=[f"Coordinate lookup failed: {exc}"],
                backend="Error",
            )

    def _display_cursor_lookup(
        self,
        *,
        x: int,
        y: int,
        value: object,
        result: CoordinateResult,
        data_unit: str | None,
    ) -> None:
        world_lines = format_world_coordinate_lines(
            result,
            coordinate_format=self.coordinate_panel.coordinate_format,
        )
        self.coordinate_panel.set_cursor_readout(
            x=x,
            y=y,
            value_text=format_data_value(value, data_unit),
            world_lines=world_lines,
            backend=result.backend,
        )

    def on_coordinate_format_changed(self, _format: str) -> None:
        if self._last_cursor_lookup is None:
            return
        x, y, value, result, data_unit = self._last_cursor_lookup
        self._display_cursor_lookup(
            x=x,
            y=y,
            value=value,
            result=result,
            data_unit=data_unit,
        )

    def on_image_cursor_left(self) -> None:
        # Keep the last successful coordinate visible for reading and copying.
        pass

    def on_coordinate_marker_requested(self, marker: CoordinateMarker) -> None:
        if self.document is None or self.current_hdu_index is None:
            return

        try:
            success, message = self._project_marker(marker)
        except Exception as exc:
            success = False
            message = f"Could not locate coordinate: {exc}"

        if not success:
            self.coordinate_panel.set_locator_status(message, is_error=True)
            return

        self._markers_for_current_hdu()[marker.marker_id] = marker
        self.coordinate_panel.add_marker_record(marker)
        self.coordinate_panel.set_locator_status(message)

    def on_coordinate_marker_removed(self, marker_id: str) -> None:
        markers = self._markers_for_current_hdu()
        markers.pop(marker_id, None)
        self.image_display.remove_overlay(marker_id)

    def on_clear_coordinate_markers(self) -> None:
        self._markers_for_current_hdu().clear()
        self.image_display.clear_overlays()

    def _project_marker(self, marker: CoordinateMarker) -> tuple[bool, str]:
        if self.document is None or self.current_hdu_index is None:
            return False, "No FITS image is selected."

        leading = self._current_leading_indices()
        if marker.kind == "point":
            assert marker.ra_deg is not None
            assert marker.dec_deg is not None
            result = self.document.celestial_to_pixel(
                self.current_hdu_index,
                ra_deg=marker.ra_deg,
                dec_deg=marker.dec_deg,
                leading_indices=leading,
            )
            if not result.success or result.x is None or result.y is None:
                return False, result.message or "The coordinate could not be projected."
            if not result.inside_image:
                return False, result.message or "The coordinate is outside the image."

            self.image_display.set_point_overlay(
                marker.marker_id,
                x=result.x,
                y=result.y,
                label=marker.label,
            )
            return (
                True,
                f"{marker.label} located at x={result.x:.2f}, y={result.y:.2f}.",
            )

        line_result = self.document.celestial_line(
            self.current_hdu_index,
            ra_deg=marker.ra_deg,
            dec_deg=marker.dec_deg,
            leading_indices=leading,
        )
        if (
            not line_result.success
            or line_result.x_values is None
            or line_result.y_values is None
        ):
            return False, line_result.message or "The coordinate line does not cross the image."

        self.image_display.set_line_overlay(
            marker.marker_id,
            x_values=line_result.x_values,
            y_values=line_result.y_values,
            label=marker.label,
        )
        return True, f"{marker.label} was drawn on the current image."

    def _refresh_coordinate_overlays(self) -> None:
        self.image_display.clear_overlays()
        for marker in self._markers_for_current_hdu().values():
            try:
                self._project_marker(marker)
            except Exception:
                # A marker may fall outside one cube slice. Keep its record so
                # it can reappear on another slice, but never interrupt viewing.
                continue

    def _markers_for_current_hdu(self) -> dict[str, CoordinateMarker]:
        if self.current_hdu_index is None:
            return {}
        return self._coordinate_markers.setdefault(self.current_hdu_index, {})

    def _current_leading_indices(self) -> tuple[int, ...]:
        if self.document is None or self.current_hdu_index is None:
            return ()
        descriptor = self.document.descriptor(self.current_hdu_index)
        if descriptor.category == "image_nd":
            return self._pending_slice_indices
        return ()

    def toggle_header(self) -> None:
        if self.document is None or self.current_hdu_index is None:
            return

        descriptor = self.document.descriptor(self.current_hdu_index)
        if self._header_visible and descriptor.category != "empty":
            self._header_visible = False
            self.header_button.setText("Show Header")
            try:
                self._display_selected_hdu(reset_axes=False)
            except Exception as exc:
                self._show_data_error(self.current_hdu_index, exc)
        else:
            self._show_header()

    def _show_header(self) -> None:
        if self.document is None or self.current_hdu_index is None:
            return

        self._slice_timer.stop()
        self.header_viewer.set_header_text(
            self.document.header_text(self.current_hdu_index)
        )
        self.data_stack.setCurrentWidget(self.header_viewer)
        self._header_visible = True

        descriptor = self.document.descriptor(self.current_hdu_index)
        if descriptor.category == "empty":
            self.header_button.setText("Header Only")
        else:
            self.header_button.setText("Back to Data")

    def _show_data_error(self, hdu_index: int, error: Exception) -> None:
        self.axis_selector.clear()
        self.message_view.setText(
            f"Could not display HDU {hdu_index}.\n\n{error}"
        )
        self.data_stack.setCurrentWidget(self.message_view)

    def show_1d_spectrum(self):
        """Load a companion ``*_1D.txt`` file created by ScienceProcessor."""

        if not self.path:
            return

        path = Path(self.path)
        candidate_stems = [path.stem]
        if path.stem.endswith("_calibrated"):
            candidate_stems.append(path.stem[: -len("_calibrated")])

        txt_path = None
        for stem in candidate_stems:
            candidate = path.with_name(f"{stem}_1D.txt")
            if candidate.exists():
                txt_path = candidate
                break

        if txt_path is None:
            searched = "\n".join(
                str(path.with_name(f"{stem}_1D.txt"))
                for stem in candidate_stems
            )
            show_error(
                self,
                "Could not find 1D spectrum file. Searched:\n" + searched,
            )
            return

        try:
            data = np.atleast_2d(np.loadtxt(txt_path))
            if data.shape[1] < 2:
                raise ValueError(
                    "1D spectrum file must contain at least two columns."
                )

            wavelengths = data[:, 0]
            intensities = data[:, 1]
            self.spectrum_plot.plot_data(
                wavelengths,
                intensities,
                filename=path.name,
            )
            self.stack.setCurrentWidget(self.spectrum_plot)
        except Exception as exc:
            show_error(self, f"Error loading or plotting 1D spectrum:\n{exc}")

    def show_current_fits_data(self):
        if self.document is not None:
            self.stack.setCurrentWidget(self.document_container)

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        if self.document is not None:
            self.document.close()

        super().closeEvent(event)
