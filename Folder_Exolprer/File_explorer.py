from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTreeView, QLineEdit, QToolBar
from PyQt6.QtCore import pyqtSignal, QModelIndex
from pathlib import Path
from .file_system_model import FileSystemModel
from PyQt6.QtWidgets import QHeaderView


class FileExplorer(QWidget):
    textPreviewRequested = pyqtSignal(str) # emit text content for preview
    fileSelected = pyqtSignal(str)  # emit path when a file is double-clicked

    def __init__(self, root_path=None):
        super().__init__()
        self.setAcceptDrops(True)
        root_path = str(Path(root_path or ".").resolve())
        self.model = FileSystemModel(root_path)

        # Toolbar
        self.toolbar = QToolBar()
        self.path_entry = QLineEdit()
        self.path_entry.setText(root_path)
        self.path_entry.returnPressed.connect(self.change_directory)
        self.toolbar.addWidget(self.path_entry)

        # Tree view
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # name stretches
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.doubleClicked.connect(self.on_double_click)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.tree)
        layout.setContentsMargins(0, 0, 0, 0)

    def load_folder(self, path):
        """Load folder from dropped path."""
        try:
            p = Path(path)
            if p.exists() and p.is_dir():
                self.model.set_root_path(str(p))
                self.path_entry.setText(str(p))
                print(f"Loaded folder: {p}")
        except Exception as e:
            print(f"Error loading folder: {e}")

    def change_directory(self):
        """Change directory from path entry."""
        try:
            path = Path(self.path_entry.text())
            if path.exists() and path.is_dir():
                self.model.set_root_path(str(path.resolve()))
                self.path_entry.setText(str(path.resolve()))
        except Exception as e:
            print(f"Error changing directory: {e}")

    def on_double_click(self, index: QModelIndex):
        """Handle double-click on tree items."""
        if not index.isValid():
            return
            
        item = index.internalPointer()
        if not item:
            return
            
        try:
            if item.is_dir:
                # Navigate to directory
                self.path_entry.setText(str(item.path))
                self.model.set_root_path(str(item.path))
            else:
                # Handle file
                file_path = str(item.path)
                if file_path.lower().endswith(('.fit', '.fits', '.fts')):
                    self.fileSelected.emit(file_path)
                else:
                    # Preview text files
                    try:
                        self.textPreviewRequested.emit(file_path)
                    except Exception:
                        self.textPreviewRequested.emit(f"Cannot preview file: {file_path}")
        except Exception as e:
            print(f"Error handling double-click: {e}")

    # --- Drag and Drop Event Handling ---
    
    def dragEnterEvent(self, event):
        """Accept drops if they contain URLs (e.g., files/folders)."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Ensure the move is accepted if it contains URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle the drop event, loading the first folder dropped."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                # Check if the path is a directory before loading
                if Path(path).is_dir():
                    self.load_folder(path)
                    event.acceptProposedAction()
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()