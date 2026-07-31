"""Virtualized Qt table viewer for FITS ASCII and binary tables."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class FitsTableModel(QAbstractTableModel):
    def __init__(self, table_data=None, columns=None, parent=None):
        super().__init__(parent)
        self._table = table_data
        self._columns = columns
        self._names = self._get_column_names(table_data)

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid() or self._table is None:
            return 0
        return len(self._table)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._names)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._table is None:
            return None

        name = self._names[index.column()]
        value = self._table[name][index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return self._format_value(value, compact=True)
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._format_value(value, compact=False)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Vertical:
            return str(section)
        if not 0 <= section < len(self._names):
            return None

        name = self._names[section]
        unit = None
        if self._columns is not None:
            try:
                unit = self._columns[section].unit
            except (IndexError, AttributeError):
                unit = None
        return f"{name} [{unit}]" if unit else name

    @staticmethod
    def _get_column_names(table_data) -> list[str]:
        if table_data is None:
            return []
        names = getattr(table_data, "names", None)
        if names is None:
            names = getattr(getattr(table_data, "dtype", None), "names", None)
        return list(names or [])

    @classmethod
    def _format_value(cls, value, compact: bool) -> str:
        if np.ma.is_masked(value):
            return "—"
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8", errors="replace")

        array = np.asanyarray(value)
        if array.ndim > 0:
            if compact and array.size > 8:
                return f"Array shape={array.shape}, dtype={array.dtype}"
            return np.array2string(
                array,
                threshold=8 if compact else 200,
                edgeitems=3,
                max_line_width=120,
            )
        if isinstance(value, np.generic):
            value = value.item()
        return str(value)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self._names)


class FitsTableViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._table_data = None
        self._suggested_filename = "fits_table.csv"

        self.summary_label = QLabel("No FITS table selected")
        self.save_button = QPushButton("Save Table as CSV")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_as_csv)

        summary_layout = QHBoxLayout()
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.addWidget(self.summary_label)
        summary_layout.addStretch()
        summary_layout.addWidget(self.save_button)

        self.table_view = QTableView()
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.table_view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table_view.setSortingEnabled(False)
        self.table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table_view.horizontalHeader().setResizeContentsPrecision(100)
        self.table_view.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed
        )
        self.table_view.verticalHeader().setDefaultSectionSize(24)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addLayout(summary_layout)
        layout.addWidget(self.table_view, 1)

        self._model = FitsTableModel()
        self.table_view.setModel(self._model)

    def set_hdu(
        self,
        hdu,
        suggested_filename: str = "fits_table.csv",
    ) -> None:
        table_data = hdu.data
        if table_data is None:
            raise ValueError("The selected FITS table contains no rows.")

        self._table_data = table_data
        self._suggested_filename = self._normalise_csv_filename(
            suggested_filename
        )

        old_model = self._model
        self._model = FitsTableModel(
            table_data=table_data,
            columns=getattr(hdu, "columns", None),
            parent=self,
        )
        self.table_view.setModel(self._model)
        old_model.deleteLater()

        self.summary_label.setText(
            f"{len(table_data):,} rows × {self._model.columnCount():,} columns"
        )
        self.save_button.setEnabled(True)
        self.table_view.resizeColumnsToContents()

    def save_as_csv(self) -> None:
        """Write the current FITS table row by row without copying it."""

        if self._table_data is None:
            QMessageBox.information(
                self,
                "No Table",
                "There is no FITS table to save.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FITS Table as CSV",
            self._suggested_filename,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        output_path = Path(path)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".csv")

        names = self._model.column_names
        try:
            with output_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(names)

                for row_index in range(len(self._table_data)):
                    writer.writerow(
                        self._csv_value(self._table_data[name][row_index])
                        for name in names
                    )
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(
                self,
                "Table Export Failed",
                f"Could not save the FITS table:\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Table Saved",
            f"The FITS table was saved to:\n{output_path}",
        )

    @staticmethod
    def _csv_value(value):
        """Convert scalar and vector FITS cells into safe CSV fields."""

        if np.ma.is_masked(value):
            return ""
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8", errors="replace")

        array = np.asanyarray(value)
        if array.ndim > 0:
            return np.array2string(
                array,
                separator=" ",
                threshold=array.size,
                max_line_width=1_000_000,
            )

        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _normalise_csv_filename(filename: str) -> str:
        path = Path(filename or "fits_table.csv")
        return path.name if path.suffix.lower() == ".csv" else f"{path.name}.csv"
