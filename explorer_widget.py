'''from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTreeView, QLineEdit, QToolBar, QSplitter, QTextEdit
from PyQt6.QtCore import Qt, pyqtSignal, QModelIndex, QEvent
from PyQt6.QtGui import QAction
from pathlib import Path
from .file_system_model import FileSystemModel
from ..widgets.folder_drop_area import DropArea
from PyQt6.QtWidgets import QHeaderView


class FileExplorer(QWidget):
    fileSelected = pyqtSignal(str)  # emit path when a file is double-clicked

    def __init__(self, root_path=None):
        super().__init__()
        self.setAcceptDrops(True)
        self.model = FileSystemModel(root_path)
        self.model.set_root_path("") 

        # Toolbar
        self.toolbar = QToolBar()
        self.path_entry = QLineEdit()
        self.path_entry.returnPressed.connect(self.change_directory)
        self.toolbar.addWidget(self.path_entry)

        self.toggle_drop_action = QAction("Show Drop Area", self)
        self.toggle_drop_action.setCheckable(True)
        self.toggle_drop_action.toggled.connect(self._on_toggle_drop)
        self.toolbar.addAction(self.toggle_drop_action)

        # Tree view
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # name stretches
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        # Connect double-click event (this was missing!)
        self.tree.doubleClicked.connect(self.on_double_click)

        # File preview
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)

        # Splitter (explorer + preview)
        self.splitter = QSplitter()
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(self.preview)
        self.splitter.setSizes([300, 700])
        # optionally limit width of explorer as you described
        self.tree.setMaximumWidth(600)

        # Improved drag & drop area: parent it to the main widget for better control
        self.drop_area = DropArea(self)
        self.drop_area.folderDropped.connect(self.load_folder)
        self.drop_area.hide()
        
        # Apply gray styling to drop area
        self.drop_area.setStyleSheet("""
            QWidget {
                background-color: rgba(128, 128, 128, 180);
                border: 2px dashed #888888;
                border-radius: 8px;
            }
        """)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.splitter)
        layout.setContentsMargins(0, 0, 0, 0)

        # Connect events for proper geometry updates
        self.splitter.splitterMoved.connect(self._update_drop_geometry)
        self.tree.viewport().installEventFilter(self)
        
        # Track if we're currently showing drop area due to drag
        self._drag_active = False

    def eventFilter(self, obj, event):
        """Handle tree viewport events to keep drop area geometry updated."""
        if obj is self.tree.viewport():
            if event.type() in [QEvent.Type.Resize, QEvent.Type.Move]:
                self._update_drop_geometry()
        return super().eventFilter(obj, event)

    def _update_drop_geometry(self):
        """Position the drop_area to exactly cover the tree viewport."""
        if not self.drop_area.isVisible():
            return
            
        # Get tree viewport geometry in FileExplorer coordinates
        tree_viewport = self.tree.viewport()
        tree_pos = self.tree.mapTo(self, tree_viewport.pos())
        tree_size = tree_viewport.size()
        
        # Set drop area to cover the tree viewport exactly
        self.drop_area.setGeometry(
            tree_pos.x(), 
            tree_pos.y(), 
            tree_size.width(), 
            tree_size.height()
        )
        self.drop_area.raise_()

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
        finally:
            # Always hide drop area after processing
            if self._drag_active:
                self.hide_drop_area()

    def change_directory(self):
        """Change directory from path entry."""
        try:
            path = Path(self.path_entry.text())
            if path.exists() and path.is_dir():
                self.model.set_root_path(str(path))
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
                        text = item.path.read_text(errors="ignore")[:2000]
                        self.preview.setPlainText(text)
                    except Exception as e:
                        self.preview.setPlainText(f"Cannot preview file: {e}")
        except Exception as e:
            print(f"Error handling double-click: {e}")

    # Drop area control methods
    def show_drop_area(self):
        """Show the drop area overlay over the tree."""
        self._update_drop_geometry()
        self.drop_area.show()
        self.toggle_drop_action.setChecked(True)
        self.toggle_drop_action.setText("Hide Drop Area")

    def hide_drop_area(self):
        """Hide the drop area overlay."""
        self.drop_area.hide()
        self.toggle_drop_action.setChecked(False)
        self.toggle_drop_action.setText("Show Drop Area")
        self._drag_active = False

    def _on_toggle_drop(self, checked: bool):
        """Handle toolbar toggle for drop area."""
        if checked:
            self.show_drop_area()
        else:
            self.hide_drop_area()

    def _is_position_over_tree(self, pos):
        """Check if position is over the tree widget."""
        # Convert position to tree widget coordinates
        tree_pos = self.tree.mapFromParent(pos)
        return self.tree.rect().contains(tree_pos)

    # Improved drag and drop event handling
    def dragEnterEvent(self, event):
        """Handle drag enter - show drop area if dragging over tree."""
        if event.mimeData().hasUrls():
            pos = event.position().toPoint()
            if self._is_position_over_tree(pos):
                self._drag_active = True
                self.show_drop_area()
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Handle drag move - maintain drop area visibility based on position."""
        if event.mimeData().hasUrls():
            pos = event.position().toPoint()
            if self._is_position_over_tree(pos):
                if not self._drag_active:
                    self._drag_active = True
                    self.show_drop_area()
                event.acceptProposedAction()
            else:
                if self._drag_active:
                    self.hide_drop_area()
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave - hide drop area."""
        if self._drag_active:
            self.hide_drop_area()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        """Handle drop event."""
        if event.mimeData().hasUrls():
            pos = event.position().toPoint()
            if self._is_position_over_tree(pos):
                urls = event.mimeData().urls()
                if urls:
                    path = urls[0].toLocalFile()
                    self.load_folder(path)
                    event.acceptProposedAction()
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()
        
        # Always clean up after drop
        if self._drag_active:
            self.hide_drop_area()

    def resizeEvent(self, event):
        """Handle widget resize to update drop area geometry."""
        super().resizeEvent(event)
        if self.drop_area.isVisible():
            self._update_drop_geometry()
'''