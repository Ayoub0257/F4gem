
from PyQt6.QtWidgets import QMessageBox

def show_error(parent, message):
    """Show an error pop-up with the given message."""
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Error")
    msg_box.setText(message)
    msg_box.exec()