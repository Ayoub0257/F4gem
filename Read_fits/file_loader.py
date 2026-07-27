"""File-dialog and FITS-opening helpers for the viewer."""

from __future__ import annotations

import os

import astropy.units as u
from astropy.io import fits
from PyQt6.QtWidgets import QFileDialog

from .fits_document import FitsDocument


class LoadFiles:
    @staticmethod
    def select_fits(parent=None, start_dir=""):
        """Open a file dialog to select a FITS file."""
        path, _ = QFileDialog.getOpenFileName(
            parent=parent,
            caption="Select FITS file",
            directory=start_dir,
            filter="FITS files (*.fit *.fits *.fts)",
        )
        return path or None

    @staticmethod
    def open_fits_document(path: str) -> FitsDocument:
        """Open a multi-HDU FITS document for the interactive viewer."""
        return FitsDocument.open(path)

    @staticmethod
    def safe_read_ccd(path, default_unit="adu"):
        """Read a conventional primary-HDU CCD image.

        This method is retained for code that explicitly needs ``CCDData``.
        The interactive FITS viewer uses :meth:`open_fits_document` instead.
        """
        from ccdproc import CCDData

        try:
            return CCDData.read(path, unit_parse_strict="silent")
        except Exception:
            try:
                with fits.open(path, ignore_missing_end=True) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header

                if data is None:
                    raise ValueError("No image data found in primary HDU")

                unit_str = header.get("BUNIT", default_unit)
                try:
                    unit = u.Unit(unit_str)
                except Exception:
                    unit = u.Unit(default_unit)

                return CCDData(data, meta=header, unit=unit)
            except Exception as inner_err:
                print(
                    "[Warning] FITS tolerance mode: "
                    f"could not fully parse {path} -> {inner_err}"
                )
                with fits.open(path, ignore_missing_end=True) as hdul:
                    return hdul[0].data, hdul[0].header

    @staticmethod
    def get_fit_data(path):
        """Return primary-HDU image data for legacy callers only."""
        if not path:
            return None
        if not path.lower().endswith((".fit", ".fits", ".fts")):
            raise ValueError("File is not a FITS file")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")

        from ccdproc import CCDData

        result = LoadFiles.safe_read_ccd(path)
        return result.data if isinstance(result, CCDData) else result[0]
