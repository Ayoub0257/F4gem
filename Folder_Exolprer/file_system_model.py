from PyQt6.QtCore import QAbstractItemModel, QModelIndex, Qt
from .file_system_item import FileSystemItem

class FileSystemModel(QAbstractItemModel):
    def __init__(self, root_path=None):
        super().__init__()
        self.rootItem = FileSystemItem(root_path) if root_path else None

    def set_root_path(self, path):
        self.beginResetModel()
        self.rootItem = FileSystemItem(path) if path else None
        self.endResetModel()

    def rowCount(self, parent):
        if not self.rootItem:
            return 0
        parent_item = parent.internalPointer() if parent.isValid() else self.rootItem
        return parent_item.child_count() if parent_item else 0

    def columnCount(self, parent):
        return 2  # Name, Size

    def index(self, row, column, parent):
        parent_item = parent.internalPointer() if parent.isValid() else self.rootItem
        if parent_item:
            child_item = parent_item.child(row)
            if child_item:
                return self.createIndex(row, column, child_item)
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        item = index.internalPointer()
        if item and item.parent and item.parent != self.rootItem:
            return self.createIndex(item.parent.row(), 0, item.parent)
        return QModelIndex()

    def data(self, index, role):
        if not index.isValid():
            return None
        item = index.internalPointer()
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return item.path.name
            elif index.column() == 1 and item.path.is_file():
                try:
                    return f"{item.path.stat().st_size // 1024} KB"
                except OSError:
                    return ""
            '''elif index.column() == 2:
                return "Folder" if item.is_dir else "File"'''
        return None

    def hasChildren(self, index):
        if not index.isValid():
            return True
        item = index.internalPointer()
        return item.is_dir
    # this methode is for header of the table 
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        """Return human-readable column headers instead of 1/2/3."""
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        labels = ["Name", "Size"]
        if 0 <= section < len(labels):
            return labels[section]
        return None