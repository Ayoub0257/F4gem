import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QSplitter, QHBoxLayout, QWidget, QVBoxLayout, QPushButton, QToolBar, QTextEdit
from PyQt6.QtCore import Qt
from .menu_bar import MenuBar
from Read_fits.viewer import FitsViewer  # your existing spectro module
from Folder_Exolprer.File_explorer import FileExplorer  # import your explorer widget
from Spectro.calibration_controller import CalibrationController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Faraway App")
        self.setGeometry(100, 100, 800, 400)
        self.init_ui()

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main vertical layout to stack toolbar and splitter
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create toolbar for the explorer toggle button
        toolbar = QToolBar()
        self.explorer_button = QPushButton("🗂 Explorer")
        self.explorer_button.setCheckable(True)
        self.explorer_button.toggled.connect(self.toggle_explorer)
        toolbar.addWidget(self.explorer_button)
        main_layout.addWidget(toolbar)

        # Create horizontal splitter for the main content
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Add a style to the splitter to make the handle visible and clean
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #d0d0d0; /* A light gray color */
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
        """)

        # File Explorer (left panel) - wrapped in container for show/hide
        self.file_explorer = FileExplorer(root_path=".")  # start at current folder
        self.file_explorer_container = QWidget()

        fe_layout = QVBoxLayout(self.file_explorer_container)
        fe_layout.setContentsMargins(0, 0, 0, 0)
        fe_layout.addWidget(self.file_explorer)
        self.file_explorer_container.hide()  # start hidden
        self.splitter.addWidget(self.file_explorer_container)
        '''
        # Text Preview pane (middle panel)
        self.preview_pane = QTextEdit()
        self.preview_pane.setReadOnly(True)
        self.preview_pane.setPlaceholderText("Double-click a text file in the explorer to preview it here.")
        self.preview_pane.hide() # Start hidden, tied to explorer
        self.splitter.addWidget(self.preview_pane)'''

        # Spectroscopy viewer (right panel)'''
        self.spectro_ui = FitsViewer()
        self.splitter.addWidget(self.spectro_ui)

        # Connect explorer signal to FITS viewer
        self.file_explorer.fileSelected.connect(self.spectro_ui.load_fits_from_path)
        # self.file_explorer.textPreviewRequested.connect(self.update_preview_pane)

        # Create the controller and menu bar
        self.calibration_controller = CalibrationController(self)
        self.menu_bar = MenuBar(self, viewer=self.spectro_ui, controller=self.calibration_controller)

        # Set initial sizes (when explorer is shown)
        self.splitter.setSizes([250, 800])  # Explorer, Viewer (adjust as needed)

    # def update_preview_pane(self, file_path):
    #     """Loads and displays text content in the preview pane."""
    #     try:
    #         with open(file_path, 'r', errors='ignore') as f:
    #             content = f.read(5000) # Read first 5000 chars
    #         self.preview_pane.setPlainText(content)
    #     except Exception as e:
    #         self.preview_pane.setPlainText(f"Could not preview file:\n{e}")

    def toggle_explorer(self, checked: bool):
        """Toggle the visibility of the file explorer panel"""
        if checked:
            self.file_explorer_container.show()
            # self.preview_pane.show()
            # Set sizes for the visible panels: Explorer, Viewer
            self.splitter.setSizes([250, 800])
        else:
            self.file_explorer_container.hide()
