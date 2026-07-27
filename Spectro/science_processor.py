from pathlib import Path
import numpy as np
import re
from astropy.io import fits
from ccdproc import CCDData, ccd_process
import astropy.units as u


class ScienceProcessor:
    """
    A specialist class for calibrating a single raw science image.
    """
    def __init__(self, session_folder: Path, science_filepath: str, dispersion_solution, progress_callback=None):
        self.session_folder = session_folder
        self.science_filepath = Path(science_filepath)
        self.dispersion_solution = dispersion_solution
        self.progress_callback = progress_callback
        self.master_folder = self.session_folder / "masters"
        self.calibrated_folder = self.session_folder / "calibrated"

    def _emit_progress(self, message: str):
        if self.progress_callback:
            self.progress_callback.emit(message)

    def _get_exposure_time(self, header: fits.Header, filename: str) -> float | None:
        """Search for exposure time first in FITS header keywords, then filename."""
        common_keywords = ['EXPTIME', 'EXPOSURE']
        for key in common_keywords:
            if key in header:
                return float(header[key])

        match = re.search(r'-(\d+\.?\d*)s-', filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _find_best_dark(self, target_exptime: float) -> tuple[CCDData | None, float | None]:
        """Finds the best master dark and returns it and its exposure time."""
        master_dark_path = self.master_folder / f"master_dark_{int(target_exptime)}s.fits"
        if master_dark_path.exists():
            self._emit_progress(f"Found exact master dark for {target_exptime}s.")
            return CCDData.read(master_dark_path, unit='adu', unit_parse_strict='silent'), target_exptime

        available_darks = list(self.master_folder.glob("master_dark_*.fits"))
        if not available_darks:
            return None, None

        candidates = []
        for dark_path in available_darks:
            try:
                candidate_exptime = float(dark_path.stem.split('_')[-1][:-1])
            except (ValueError, IndexError):
                continue
            candidates.append((abs(candidate_exptime - target_exptime), candidate_exptime, dark_path))

        if not candidates:
            raise ValueError("No readable master dark exposure times were found.")

        _, dark_exptime, best_dark_path = min(candidates, key=lambda item: item[0])
        self._emit_progress(f"No exact dark match found. Will use {best_dark_path.name} for scaling.")
        return CCDData.read(best_dark_path, unit='adu', unit_parse_strict='silent'), dark_exptime

    def _find_best_flat(self, target_exptime: float) -> CCDData:
        """Find the best master flat for the science exposure time."""
        exact_flat_path = self.master_folder / f"master_flat_{int(target_exptime)}s.fits"
        if exact_flat_path.exists():
            self._emit_progress(f"Found exact master flat for {target_exptime}s.")
            return CCDData.read(exact_flat_path, unit='adu', unit_parse_strict='silent')

        generic_flat_path = self.master_folder / "master_flat.fits"
        if generic_flat_path.exists():
            self._emit_progress("Using generic master_flat.fits.")
            return CCDData.read(generic_flat_path, unit='adu', unit_parse_strict='silent')

        raise FileNotFoundError("No master flat found. Cannot calibrate science frame.")

    def run_science_calibration(self):
        """Orchestrates the calibration of a single science frame."""
        self._emit_progress(f"Calibrating science frame: {self.science_filepath.name}...")

        # 1. Load all necessary data safely
        science_ccd = CCDData.read(self.science_filepath, unit='adu', unit_parse_strict='silent')
        master_bias = CCDData.read(self.master_folder / "master_bias.fits", unit='adu', unit_parse_strict='silent')

        # Find the correct exposure-dependent calibration frames
        exptime = self._get_exposure_time(science_ccd.header, self.science_filepath.name)
        if exptime is None:
            raise ValueError(f"Science frame '{self.science_filepath.name}' is missing an exposure time keyword.")
        
        master_dark, dark_exptime = self._find_best_dark(exptime)
        if master_dark is None:
            raise FileNotFoundError(f"No suitable master dark found to calibrate science frame '{self.science_filepath.name}'.")
        master_flat = self._find_best_flat(exptime)

        # 2. Apply calibration
        self._emit_progress("Applying BIAS, DARK, and FLAT corrections...")
        calibrated_ccd = ccd_process(
            science_ccd,
            master_bias=master_bias,
            dark_frame=master_dark,
            data_exposure=exptime * u.s,
            dark_exposure=dark_exptime * u.s,
            master_flat=master_flat
        )

        # 3. Apply wavelength calibration for the extracted 1D spectrum.
        # The current dispersion solution is a 1D x-pixel -> wavelength model.
        self._emit_progress("Applying wavelength solution to extracted 1D spectrum...")

        # 4. Save calibrated 2D spectrum
        output_filename = f"{self.science_filepath.stem}_calibrated.fits"
        output_path = self.calibrated_folder / output_filename
        calibrated_ccd.write(output_path, overwrite=True)
        self._emit_progress(f"Saved calibrated 2D spectrum to {output_filename}")

        # 5. Extract and save 1D spectrum
        spectrum_1d = np.sum(calibrated_ccd.data, axis=0)
        wavelengths_1d = self.dispersion_solution(np.arange(spectrum_1d.size))
        txt_output_path = self.calibrated_folder / f"{self.science_filepath.stem}_1D.txt"
        np.savetxt(txt_output_path, np.transpose([wavelengths_1d, spectrum_1d]),
                   header="Wavelength(A) Intensity(ADU)")
        self._emit_progress(f"Saved extracted 1D spectrum to {txt_output_path.name}")
