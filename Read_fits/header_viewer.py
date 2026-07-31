"""Read-only FITS header display with text export."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QFontDatabase, QTextCursor
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HeaderViewer(QWidget):
    """Display the selected HDU header and optionally save it as plain text."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._suggested_filename = "fits_header.txt"

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text_edit.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        self.text_edit.setPlaceholderText(
            "Select an HDU to inspect its FITS header."
        )

        self.save_button = QPushButton("Save Header as TXT")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_header)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addLayout(toolbar_layout)
        layout.addWidget(self.text_edit, 1)

    def set_header_text(
        self,
        text: str,
        suggested_filename: str = "fits_header.txt",
    ) -> None:
        self._suggested_filename = self._normalise_txt_filename(
            suggested_filename
        )
        self.text_edit.setPlainText(text)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.Start)
        self.save_button.setEnabled(bool(text))

    def save_header(self) -> None:
        """Save the currently displayed FITS header without modifying it."""

        text = self.text_edit.toPlainText()
        if not text:
            QMessageBox.information(
                self,
                "No Header",
                "There is no FITS header to save.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FITS Header",
            self._suggested_filename,
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return

        output_path = Path(path)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".txt")

        try:
            output_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(
                self,
                "Header Export Failed",
                f"Could not save the FITS header:\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Header Saved",
            f"The FITS header was saved to:\n{output_path}",
        )

    @staticmethod
    def _normalise_txt_filename(filename: str) -> str:
        path = Path(filename or "fits_header.txt")
        return path.name if path.suffix.lower() == ".txt" else f"{path.name}.txt"
