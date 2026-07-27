from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from scipy.signal import find_peaks
import asdf

class WavelengthProcessor:
    """
    A specialist class for performing wavelength calibration.
    It finds the dispersion solution from a master lamp image.
    """
    def __init__(self, session_folder: Path, config: dict, progress_callback=None):
        self.session_folder = session_folder
        self.config = config
        self.progress_callback = progress_callback
        self.master_folder = self.session_folder / "masters"

    def _emit_progress(self, message: str):
        if self.progress_callback:
            self.progress_callback.emit(message)

    def run_wavelength_calibration(self):
        """
        Orchestrates the entire wavelength calibration process.
        """
        self._emit_progress("Starting wavelength calibration...")
        master_lamp_path = self.master_folder / "master_lamp.fits"
        if not master_lamp_path.exists():
            raise FileNotFoundError("master_lamp.fits not found. Cannot perform wavelength calibration.")

        # 1. Extract a 1D profile from the 2D lamp image
        with fits.open(master_lamp_path) as hdul:
            lamp_data_2d = hdul[0].data
        # Sum along the spatial axis (axis 0) to get a 1D spectrum
        lamp_profile_1d = np.sum(lamp_data_2d, axis=0)

        # 2. Find peaks (emission lines) in the 1D profile
        # The 'prominence' parameter is crucial for rejecting noise
        peaks, _ = find_peaks(lamp_profile_1d, prominence=np.std(lamp_profile_1d) * 3)
        self._emit_progress(f"Found {len(peaks)} potential emission lines in the lamp spectrum.")

        # 3. Match peaks to the known line list from the config
        # This is a simplified matching algorithm. More advanced methods exist.
        known_lines = np.array(self.config['calibration']['thar_line_list_A'], dtype=float)
        if known_lines.size == 0:
            raise ValueError("No calibration line list was found in the configuration file.")

        ref_pixel = self.config['calibration']['reference_pixel']['x']
        ref_lambda = self.config['calibration']['reference_wavelength_A']

        # Estimate dispersion (Angstroms per pixel) using the reference point
        # This is a rough guess to help with matching
        initial_dispersion = 0.5  # This could be a parameter in the config file

        pixel_coords = []
        wavelength_coords = []

        for peak_px in peaks:
            # Predict wavelength based on distance from reference pixel
            predicted_lambda = ref_lambda + (peak_px - ref_pixel) * initial_dispersion
            
            # Find the closest known line from our list
            closest_known_line = known_lines[np.argmin(np.abs(known_lines - predicted_lambda))]

            # If the match is close enough (e.g., within a few Angstroms), accept it
            if abs(closest_known_line - predicted_lambda) < 5: # 5 Angstrom tolerance
                pixel_coords.append(peak_px)
                wavelength_coords.append(closest_known_line)
        
        self._emit_progress(f"Successfully matched {len(pixel_coords)} lines.")

        # 4. Fit a polynomial model to the matched points
        if len(pixel_coords) < 2:
            raise ValueError(
                "Not enough matched lamp lines to fit a wavelength solution. "
                f"Matched {len(pixel_coords)} line(s)."
            )

        fit = fitting.LinearLSQFitter()
        # Use up to a 3rd-degree polynomial, but lower the degree if few lines were matched.
        degree = min(3, len(pixel_coords) - 1)
        poly_model = models.Polynomial1D(degree=degree)
        dispersion_solution = fit(poly_model, pixel_coords, wavelength_coords)

        # 5. Save the dispersion solution model to a file for later use
        solution_path = self.master_folder / "dispersion_solution.asdf"
        tree = {'model': dispersion_solution}
        with asdf.AsdfFile(tree) as ff:
            ff.write_to(solution_path)

        self._emit_progress("Wavelength calibration successful. Dispersion solution found.")
        return dispersion_solution