"""Read-only FITS header display."""

from PyQt6.QtGui import QFontDatabase, QTextCursor
from PyQt6.QtWidgets import QPlainTextEdit


class HeaderViewer(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.setPlaceholderText("Select an HDU to inspect its FITS header.")

    def set_header_text(self, text: str) -> None:
        self.setPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.Start)
