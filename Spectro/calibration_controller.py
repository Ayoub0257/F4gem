from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from .calibration_manager import CalibrationManager



class CalibrationWorker(QObject):
    """this class will run the calibration tasks in a separate thread"""

    finished = pyqtSignal()
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, folder_path, config_path):
        super().__init__()
        self.folder_path = folder_path
        self.config_path = config_path

    def run(self):
        """This function will run the calibration tasks."""
        try: 

            calib_manager = CalibrationManager(self.folder_path, progress_callback=self.progress)
            calib_manager.create_session_folder()
            calib_manager.load_configuration(self.config_path)
            calib_manager.classify_and_copy_fits()
            calib_manager.generate_master_frames()
            calib_manager.run_wavelength_calibration()
            calib_manager.calibrate_science_frames()

        except OSError as e:
            if e.errno == 28: # [Errno 28] No space left on device
                self.error.emit("Calibration failed: No space left on device. Please free up disk space and try again.")
            else:
                self.error.emit(f"A file system error occurred: {e}")
        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {e}")
        finally:
            self.finished.emit()

class CalibrationController:
    """this class will manage the calibration process and interact with the UI"""
    def __init__(self, main_window):
        self.main_window = main_window
        self.thread = None
        self.worker = None
        self._calibration_failed = False
    
    def run_calibration(self): 
        """this function will start the calibration process"""
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self.main_window, "Calibration Running", "A calibration process is already running.")
            return

        self._calibration_failed = False
        spect_folder = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Spectroscopy Data Folder")
        if not spect_folder: 
            return
        
        # QFileDialog.getOpenFileName returns a tuple (filepath, filter)
        config_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Configuration File",
            directory=spect_folder, 
            filter="Config Files (*.json *.conf)")
        if not config_path:
            return
        # setup the worker and thread
        self.thread = QThread()
        self.worker = CalibrationWorker(spect_folder, config_path)
        self.worker.moveToThread(self.thread)

        # connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.on_calibration_finished)
        self.worker.error.connect(self.on_calibration_error)
        self.thread.start()
        self.main_window.statusBar().showMessage("Calibration started...")

    def update_status(self, message: str):
        """Updates the main window's status bar with progress messages."""
        self.main_window.statusBar().showMessage(message)

    def on_calibration_finished(self):
        """Called when the worker thread finishes."""
        if self._calibration_failed:
            return
        self.main_window.statusBar().showMessage("Calibration process completed successfully.", 5000)

    def on_calibration_error(self, error_message: str):
        """Called when the worker thread emits an error."""
        self._calibration_failed = True
        QMessageBox.critical(self.main_window, "Calibration Error", error_message)
        self.main_window.statusBar().showMessage("Calibration failed.", 5000)
