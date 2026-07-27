import datetime
from astropy.io import fits
import shutil
import json
from pathlib import Path
from .master_processor import MasterProcessor
from .wavelength_processor import WavelengthProcessor
from .science_processor import ScienceProcessor
import xml.etree.ElementTree as ET
import asdf


class CalibrationManager:
    """handling session folder creation,Fits classification and metadata logging""" 
    def __init__(self, base_folder, progress_callback=None):
        self.base_folder = Path(base_folder) # base folder where all sessions will be stored
        self.progress_callback = progress_callback
        self.session_folder = None
        self.raw_folder = None
        self.master_folder = None
        self.calibrated_folder = None
        self.session_info = {k: [] for k in ["bias", "dark", "flat", "lamp", "science", "unsupported"]} # to store metadata about the session
        self.config = None # to store the loaded configuration data

    def create_session_folder(self) -> Path:
        """Create a session folder structure and return its path."""
        self._emit_progress("Creating session folder...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # this will create a unique folder name based on the current date and time
        # The session folder is now created INSIDE the working directory
        self.session_folder = self.base_folder / f"FAR_spectro_session_{timestamp}"
        self.master_folder = self.session_folder / "masters"
        self.calibrated_folder = self.session_folder / "calibrated"

        self.master_folder.mkdir(parents=True, exist_ok=True)
        self.calibrated_folder.mkdir(parents=True, exist_ok=True)

        return self.session_folder
    
    def load_configuration(self, config_path: str):
        """
        Loads a JSON configuration file, validates it, and copies it to the session folder.
        """
        self._emit_progress(f"Loading configuration from {Path(config_path).name}...")
        config_file = Path(config_path)
        if not self.session_folder:
            raise RuntimeError("A session folder must be created before loading a configuration.")
        if not config_file.exists() or not config_file.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        # For reproducibility, copy the original config file into the session folder
        shutil.copy(config_file, self.session_folder / config_file.name)

        if config_file.suffix.lower() == '.json':
            try:
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in configuration file: {config_file}")
        elif config_file.suffix.lower() == '.conf':
            try:
                self.config = self._parse_conf_file(config_file)
            except Exception as e:
                raise ValueError(f"Failed to parse .conf file: {e}")
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")

        self._emit_progress("Configuration loaded and archived.")

    def _parse_conf_file(self, config_path: Path) -> dict:
        """Parses the AudeLa-style .conf (XML) file and converts it to the standard dict format."""
        self._emit_progress(f"Parsing .conf file: {config_path.name}...")
        tree = ET.parse(config_path)
        root = tree.getroot()
        params = root.find('PARAMS').attrib

        # Helper to convert string values to the correct type
        def get_val(key, type_func, default=None):
            return type_func(params.get(key, default)) if params.get(key) is not None else default

        # Map the flat XML attributes to the nested dictionary structure
        config_dict = {
            "spectrograph": {
                "name": get_val('spectroName', str, "unknown"),
                "grating_density_l_mm": get_val('grating', float),
                "focal_length_mm": get_val('focale', float),
                "angles_deg": {
                    "alpha_incidence": get_val('alpha', float),
                    "beta_diffraction": get_val('beta', float),
                    "gamma_offset": get_val('gamma', float)
                }
            },
            "calibration": {
                "reference_pixel": { "x": get_val('refX', int), "y": get_val('refY', int) },
                "reference_wavelength_A": get_val('refLambda', float),
                "thar_line_list_A": [float(wl) for wl in get_val('lineList', str, "").split()]
            },
            "detector": {
                "camera_name": get_val('cameraName', str, "unknown"),
                "pixel_size_um": get_val('pixelSize', float) * 1000, # convert mm to um
                "dimensions_px": { "width": get_val('width', int), "height": get_val('height', int) }
            }
        }
        return config_dict

    def classify_and_copy_fits(self):
        """Classifies FITS files in the base folder and stores their full paths."""
        self._emit_progress("Classifying FITS files...")

        fit_extensions = {'.fits', '.fit', '.fts'}
        source_files = [p for p in self.base_folder.glob('*') if p.is_file() and p.suffix.lower() in fit_extensions]

        for fpath in source_files:
            try:
                hdr = fits.getheader(fpath, ext=0)
                ftype = str(hdr.get("IMAGETYP", "")).strip().upper()
                telescope = str(hdr.get("TELESCOP", "")).strip().upper()
                primary_ndim = int(hdr.get("NAXIS", 0) or 0)
            except (IOError, OSError, ValueError) as exc:
                self._emit_progress(f"Skipping unreadable FITS file '{fpath.name}': {exc}")
                self.session_info["unsupported"].append(str(fpath))
                continue

            # The calibration pipeline is intentionally limited to conventional
            # 2D detector images in the primary HDU. Multi-HDU tables and N-D
            # products (including JWST data) remain fully viewable, but must not
            # be processed as ground-based bias/dark/flat/science frames.
            if telescope == "JWST" or primary_ndim != 2:
                reason = "JWST product" if telescope == "JWST" else f"primary HDU has {primary_ndim} dimensions"
                self._emit_progress(
                    f"Viewer-only FITS file '{fpath.name}' ({reason}); "
                    "excluded from CCD calibration."
                )
                self.session_info["unsupported"].append(str(fpath))
                continue

            fname_lower = fpath.name.lower()
            if "BIAS" in ftype or any(keyword in fname_lower for keyword in ["bias", "zero"]):
                subfolder = "bias"
            elif "DARK" in ftype or "dark" in fname_lower:
                subfolder = "dark"
            elif "FLAT" in ftype or any(keyword in fname_lower for keyword in ["flat", "tung", "tungsten"]):
                subfolder = "flat"
            elif any(keyword in ftype for keyword in ["COMP", "LAMP", "THAR", "NEON", "ARGON"]) or any(keyword in fname_lower for keyword in ["thar", "neon", "argon"]):
                subfolder = "lamp"
            else:
                subfolder = "science"

            # Store the full path to the original file, DO NOT COPY
            self.session_info[subfolder].append(str(fpath))

        # Save session metadata once after processing all files
        self.save_session_json()
    
    def save_session_json(self):
        """Save session metadata to a JSON file."""
        json_path = self.session_folder / "session_info.json"
        with open(json_path, "w") as f:
            json.dump(self.session_info, f, indent=4)

    def generate_master_frames(self):
        """Orchestrates the creation of all master calibration frames."""
        if not self.session_folder:
            raise RuntimeError("Session must be created before generating master frames.")
        
        processor = MasterProcessor(self.session_folder, self.session_info, self.progress_callback)
        processor.generate_master_bias()
        processor.generate_master_darks()
        processor.generate_master_flats()
        processor.generate_master_lamp()

    def run_wavelength_calibration(self):
        """Orchestrates the wavelength calibration process."""
        if not self.session_folder or not self.config:
            raise RuntimeError("Session and configuration must be loaded first.")
        
        if not self.session_info.get('lamp'):
            raise FileNotFoundError("Wavelength calibration cannot proceed: No lamp frames (e.g., ThAr, Neon) were found in the provided data.")
            
        wl_processor = WavelengthProcessor(self.session_folder, self.config, self.progress_callback)
        dispersion_solution = wl_processor.run_wavelength_calibration()
    
    def calibrate_science_frames(self):
        """Orchestrates the calibration of all raw science frames."""
        self._emit_progress("Starting science frame calibration...")
        solution_path = self.master_folder / "dispersion_solution.asdf"
        if not solution_path.exists():
            raise FileNotFoundError("dispersion_solution.asdf not found. Cannot calibrate science frames.")

        with asdf.open(solution_path) as ff:
            dispersion_solution = ff['model']

        science_file_paths = self.session_info.get('science', [])
        for science_file_path in science_file_paths:
            sci_processor = ScienceProcessor(self.session_folder, science_file_path, dispersion_solution, self.progress_callback)
            sci_processor.run_science_calibration()

    def _emit_progress(self, message: str):
        if self.progress_callback:
            self.progress_callback.emit(message)