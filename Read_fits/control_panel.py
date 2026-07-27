# class for buttons,sliders..etc...
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.open_button = QPushButton("Open file")
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter your path here..")
        self.OK_button = QPushButton("OK")
        self.Cancel_button = QPushButton("Cancel")

        # Layout
        V_layout = QVBoxLayout()
        H_layout = QHBoxLayout()
        H_layout.addWidget(self.OK_button)
        H_layout.addWidget(self.Cancel_button)
        V_layout.addWidget(self.open_button)
        V_layout.addWidget(self.path_input)
        V_layout.addLayout(H_layout)

        self.setLayout(V_layout)
        
     
