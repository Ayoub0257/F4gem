from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from Spectro.spectrum_plot import SpectrumPlotWidget

from .array_plot import FitsArrayPlot
from .axis_selector import AxisSelector
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
    """Multi-HDU FITS viewer with image, N-D, table, and header support."""

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
        self.table_viewer = FitsTableViewer()
        self.header_viewer = HeaderViewer()
        self.array_plot = FitsArrayPlot()
        self.message_view = QLabel()
        self.message_view.setWordWrap(True)
        self.message_view.setStyleSheet("padding: 20px;")

        self.data_stack = QStackedLayout()
        self.data_stack.addWidget(self.image_display)
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
        self.header_button.setText("Show Header")

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
            image = self.document.extract_2d_slice(self.current_hdu_index)
            self.image_display.show_image(image)
            self.data_stack.setCurrentWidget(self.image_display)

        elif descriptor.category == "image_nd":
            if reset_axes:
                axis_names = self.document.axis_names(self.current_hdu_index)
                assert descriptor.shape is not None
                self.axis_selector.configure(descriptor.shape, axis_names)
            self._pending_slice_indices = self.axis_selector.indices()
            self._render_pending_slice()

        elif descriptor.category == "table":
            self.axis_selector.clear()
            self.table_viewer.set_hdu(hdu)
            self.data_stack.setCurrentWidget(self.table_viewer)

        elif descriptor.category == "array_1d":
            self.axis_selector.clear()
            data = self.document.extract_1d_array(self.current_hdu_index)
            title = f"[{descriptor.index}] {descriptor.name}"
            self.array_plot.show_array(data, title=title)
            self.data_stack.setCurrentWidget(self.array_plot)

        elif descriptor.category == "empty":
            self.axis_selector.clear()
            self._show_header()

        else:
            self.axis_selector.clear()
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
            self.data_stack.setCurrentWidget(self.image_display)
        except Exception as exc:
            self._show_data_error(self.current_hdu_index, exc)

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
