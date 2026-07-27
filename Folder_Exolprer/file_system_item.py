from pathlib import Path

class FileSystemItem:
    def __init__(self, path, parent=None):
        self.path = Path(path)
        self.parent = parent
        self.children = []
        self.is_dir = self.path.is_dir()
        self._loaded = False

    def child_count(self):
        if not self._loaded:
            self.load_children()
        return len(self.children)

    def child(self, row):
        if not self._loaded:
            self.load_children()
        return self.children[row] if 0 <= row < len(self.children) else None

    def row(self):
        return self.parent.children.index(self) if self.parent else 0

    def load_children(self):
        self.children = []
        if self.is_dir:
            try:
                for entry in sorted(self.path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                    if not entry.name.startswith("."):  # skip hidden
                        self.children.append(FileSystemItem(entry, self))
            except PermissionError:
                pass
        self._loaded = True
