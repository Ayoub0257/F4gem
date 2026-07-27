from pathlib import Path
from astropy.io import fits
from ccdproc import CCDData, combine, subtract_dark, subtract_bias
import astropy.units as u
import numpy as np


class MasterProcessor:
    """
    A specialist class for creating master calibration frames (BIAS, DARK, FLAT, LAMP).
    """

    def __init__(self, session_folder: Path, session_info: dict, progress_callback=None):
        self.session_folder = session_folder
        self.session_info = session_info
        self.progress_callback = progress_callback
        self.master_folder = session_folder / "masters"

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    def _emit_progress(self, message: str):
        if self.progress_callback:
            self.progress_callback.emit(message)

    def _get_exptime_from_header(self, header: fits.Header) -> float | None:
        """Searches for common exposure time keywords in a FITS header."""
        for key in ['EXPTIME', 'EXPOSURE']:
            if key in header:
                return header[key]
        return None

    def _read_ccd_safe(self, path: str | Path) -> CCDData:
        """
        Reads a FITS file as CCDData, ignoring invalid BUNIT headers.
        """
        return CCDData.read(path, unit='adu', unit_parse_strict='silent')

    def _find_best_dark(self, target_exptime: float) -> tuple[CCDData | None, float | None]:
        """Finds the best master dark and returns it and its exposure time."""
        master_dark_path = self.master_folder / f"master_dark_{int(target_exptime)}s.fits"
        if master_dark_path.exists():
            self._emit_progress(f"Found exact master dark for {target_exptime}s.")
            dark = CCDData.read(master_dark_path, unit='adu', unit_parse_strict='silent')
            return dark, target_exptime

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
        self._emit_progress(f"No exact dark match found. Using {best_dark_path.name} for scaling.")
        dark = CCDData.read(best_dark_path, unit='adu', unit_parse_strict='silent')
        return dark, dark_exptime

    # -------------------------------------------------------------------------
    # Master Bias
    # -------------------------------------------------------------------------
    def generate_master_bias(self):
        """Generates the Master BIAS frame by median-combining all raw bias frames."""
        bias_files = self.session_info.get('bias', [])
        if not bias_files:
            self._emit_progress("No BIAS frames found. Skipping Master BIAS creation.")
            return

        self._emit_progress(f"Creating Master BIAS from {len(bias_files)} frames...")
        bias_ccds = [self._read_ccd_safe(f) for f in bias_files]

        master_bias_ccd = combine(bias_ccds, method='median', sigma_clip=True, unit='adu')
        master_bias_ccd.unit = u.adu
        master_bias_ccd.header['BUNIT'] = 'adu'
        master_bias_ccd.header['HISTORY'] = 'Master bias created by FAR_spectro.'

        master_bias_path = self.master_folder / "master_bias.fits"
        master_bias_ccd.write(master_bias_path, overwrite=True)
        self._emit_progress("Master BIAS saved.")

    # -------------------------------------------------------------------------
    # Master Darks
    # -------------------------------------------------------------------------
    def generate_master_darks(self):
        """Generates Master DARK frames, grouped by exposure time."""
        dark_files = self.session_info.get('dark', [])
        if not dark_files:
            self._emit_progress("No DARK frames found. Skipping Master DARK creation.")
            return

        master_bias_path = self.master_folder / "master_bias.fits"
        if not master_bias_path.exists():
            raise FileNotFoundError("Master BIAS not found. Cannot create Master DARKs.")

        self._emit_progress("Loading Master BIAS for dark subtraction...")
        master_bias = CCDData.read(master_bias_path, unit='adu', unit_parse_strict='silent')

        darks_by_exptime = {}
        for path in dark_files:
            header = fits.getheader(path)
            exptime = self._get_exptime_from_header(header)
            if exptime is None:
                self._emit_progress(f"Skipping dark '{Path(path).name}' (missing EXPTIME).")
                continue
            darks_by_exptime.setdefault(exptime, []).append(path)

        for exptime, file_list in darks_by_exptime.items():
            self._emit_progress(f"Creating Master DARK for {exptime}s from {len(file_list)} frames...")

            processed = []
            for dark_path in file_list:
                dark_ccd = self._read_ccd_safe(dark_path)
                dark_clean = subtract_bias(dark_ccd, master_bias)
                processed.append(dark_clean)

            master_dark = combine(processed, method='median', sigma_clip=True)
            master_dark.unit = u.adu
            master_dark.header['BUNIT'] = 'adu'
            master_dark.header['EXPTIME'] = exptime
            master_dark.header['HISTORY'] = 'Master dark created by FAR_spectro.'

            master_dark_path = self.master_folder / f"master_dark_{int(exptime)}s.fits"
            master_dark.write(master_dark_path, overwrite=True)
            self._emit_progress(f"Master DARK for {exptime}s saved.")

    # -------------------------------------------------------------------------
    # Master Flats
    # -------------------------------------------------------------------------
    def generate_master_flats(self):
        """Generates the Master FLAT frame."""
        flat_files = self.session_info.get('flat', [])
        if not flat_files:
            self._emit_progress("No FLAT frames found. Skipping Master FLAT creation.")
            return

        master_bias_path = self.master_folder / "master_bias.fits"
        if not master_bias_path.exists():
            raise FileNotFoundError("Master BIAS not found. Cannot create Master FLAT.")

        self._emit_progress("Loading Master BIAS for flat subtraction...")
        master_bias = CCDData.read(master_bias_path, unit='adu', unit_parse_strict='silent')

        flats_by_exptime = {}
        for path in flat_files:
            header = fits.getheader(path)
            exptime = self._get_exptime_from_header(header)
            if exptime is None:
                self._emit_progress(f"Skipping flat '{Path(path).name}' (missing EXPTIME).")
                continue
            flats_by_exptime.setdefault(exptime, []).append(path)

        if not flats_by_exptime:
            raise ValueError("Cannot create Master FLAT: No valid flats found.")

        for exptime, file_list in flats_by_exptime.items():
            master_dark, dark_exptime = self._find_best_dark(exptime)
            if master_dark is None:
                self._emit_progress(f"No dark found for {exptime}s flats. Skipping this group.")
                continue

            self._emit_progress(f"Creating Master FLAT for {exptime}s from {len(file_list)} frames...")
            processed_flats = []

            for flat_path in file_list:
                flat_ccd = self._read_ccd_safe(flat_path)
                flat_ccd.header['EXPTIME'] = exptime

                bias_subtracted = subtract_bias(flat_ccd, master_bias)
                calibrated_flat = subtract_dark(
                    bias_subtracted,
                    master_dark,
                    dark_exposure=dark_exptime * u.s,
                    data_exposure=exptime * u.s
                )
                processed_flats.append(calibrated_flat)

            combined_flat = combine(processed_flats, method='median', sigma_clip=True)

            master_flat = combined_flat.divide(np.mean(combined_flat.data))
            master_flat.unit = u.adu
            master_flat.header['BUNIT'] = 'adu'
            master_flat.header['HISTORY'] = 'Master flat (normalized) created by FAR_spectro.'

            master_flat.header['EXPTIME'] = exptime
            master_flat_path = self.master_folder / f"master_flat_{int(exptime)}s.fits"
            master_flat.write(master_flat_path, overwrite=True)

            # Keep the original generic filename for the existing ScienceProcessor flow.
            generic_master_flat_path = self.master_folder / "master_flat.fits"
            if not generic_master_flat_path.exists():
                master_flat.write(generic_master_flat_path, overwrite=True)

            self._emit_progress(f"Master FLAT for {exptime}s saved.")

    # -------------------------------------------------------------------------
    # Master Lamp
    # -------------------------------------------------------------------------
    def generate_master_lamp(self):
        """Generates the Master LAMP frame."""
        lamp_files = self.session_info.get('lamp', [])
        if not lamp_files:
            self._emit_progress("No LAMP frames found. Skipping Master LAMP creation.")
            return

        master_bias_path = self.master_folder / "master_bias.fits"
        if not master_bias_path.exists():
            raise FileNotFoundError("Master BIAS not found. Cannot create Master LAMP.")

        self._emit_progress("Loading Master BIAS for lamp subtraction...")
        master_bias = CCDData.read(master_bias_path, unit='adu', unit_parse_strict='silent')

        header = fits.getheader(lamp_files[0])
        exptime = self._get_exptime_from_header(header)
        if exptime is None:
            raise ValueError(f"Cannot create Master LAMP: '{Path(lamp_files[0]).name}' missing EXPTIME.")

        master_dark, dark_exptime = self._find_best_dark(exptime)
        if master_dark is None:
            raise FileNotFoundError("No suitable master dark found for lamp calibration.")

        self._emit_progress(f"Creating Master LAMP from {len(lamp_files)} frames...")
        processed_lamps = []

        for lamp_path in lamp_files:
            lamp_ccd = self._read_ccd_safe(lamp_path)
            lamp_ccd.header['EXPTIME'] = exptime

            bias_subtracted = subtract_bias(lamp_ccd, master_bias)
            calibrated_lamp = subtract_dark(
                bias_subtracted,
                master_dark,
                dark_exposure=dark_exptime * u.s,
                data_exposure=exptime * u.s
            )
            processed_lamps.append(calibrated_lamp)

        master_lamp = combine(processed_lamps, method='median', sigma_clip=True)
        master_lamp.unit = u.adu
        master_lamp.header['BUNIT'] = 'adu'
        master_lamp.header['HISTORY'] = 'Master lamp created by FAR_spectro.'

        master_lamp_path = self.master_folder / "master_lamp.fits"
        master_lamp.write(master_lamp_path, overwrite=True)
        self._emit_progress("Master LAMP saved.")
