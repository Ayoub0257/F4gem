from PyQt6.QtGui import QAction

class MenuBar:
    """Skeleton menu bar for Faraway"""
    def __init__(self, main_window, viewer=None, controller=None):
        self.main_window = main_window
        self.menubar = main_window.menuBar()
        self.viewer = viewer
        self.controller = controller  # Add this line
        self.setup_menus()

    def setup_menus(self):
        file_menu = self.menubar.addMenu("File")
        edit_menu = self.menubar.addMenu("Edit")
        view_menu = self.menubar.addMenu("View")
        help_menu = self.menubar.addMenu("Help")
        calibrate_menu = self.menubar.addMenu("Calibrate")

        # File → Open
        open_action = QAction("Open", self.main_window)
        open_action.triggered.connect(self.viewer.open_file_dialog)
        file_menu.addAction(open_action)

        # Calibrate → Run Calibration
        calibrate_action = QAction("Run Calibration Test", self.main_window)
        calibrate_action.triggered.connect(self.on_calibrate_triggered)
        calibrate_menu.addAction(calibrate_action)

    def on_calibrate_triggered(self):
        """Delegate calibration trigger to controller"""
        if self.controller:
            self.controller.run_calibration()
        else:
            print("No calibration controller assigned.")
