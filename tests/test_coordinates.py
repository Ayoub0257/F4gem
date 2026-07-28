"""Tests for the FITS-WCS coordinate layer.

Run from the project root with:
    python -m unittest -v tests.test_coordinates
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from Read_fits.coordinates import (
    PixelCoordinateProvider,
    format_cursor_readout,
)
from Read_fits.fits_document import FitsDocument


class FitsCoordinateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_2d_celestial_reference_pixel(self) -> None:
        path = self.temp_path / "celestial_2d.fits"
        data = np.zeros((40, 30), dtype=np.float32)

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [10.0, 20.0]
        wcs.wcs.cdelt = [-0.001, 0.001]
        wcs.wcs.crval = [150.0, -25.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        header = wcs.to_header()
        header["BUNIT"] = "adu"
        fits.PrimaryHDU(data=data, header=header).writeto(path)

        document = FitsDocument.open(path)
        try:
            # FITS CRPIX is one-based, while array/WCS API indices are
            # zero-based. Therefore x=9, y=19 is exactly the reference pixel.
            result = document.world_at(0, x=9, y=19)

            self.assertTrue(result.has_world_coordinates)
            self.assertEqual(result.backend, "FITS-WCS")
            self.assertAlmostEqual(float(result.world_values[0]), 150.0, places=9)
            self.assertAlmostEqual(float(result.world_values[1]), -25.0, places=9)
            self.assertEqual(document.data_unit(0), "adu")
        finally:
            document.close()

    def test_no_wcs_uses_pixel_fallback(self) -> None:
        path = self.temp_path / "no_wcs.fits"
        fits.PrimaryHDU(np.zeros((6, 8), dtype=np.float32)).writeto(path)

        document = FitsDocument.open(path)
        try:
            provider = document.coordinate_provider(0)
            self.assertIsInstance(provider, PixelCoordinateProvider)

            result = document.world_at(0, x=3, y=2)
            self.assertFalse(result.has_world_coordinates)
            self.assertEqual(result.array_indices, (2.0, 3.0))
            self.assertIn("No physical FITS-WCS", result.message or "")
        finally:
            document.close()

    def test_provider_is_cached_per_hdu(self) -> None:
        path = self.temp_path / "cached_wcs.fits"
        data = np.zeros((5, 7), dtype=np.float32)

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.cdelt = [1.0, 1.0]
        wcs.wcs.ctype = ["LINEAR", "LINEAR"]
        fits.PrimaryHDU(data=data, header=wcs.to_header()).writeto(path)

        document = FitsDocument.open(path)
        try:
            first = document.coordinate_provider(0)
            second = document.coordinate_provider(0)
            self.assertIs(first, second)
        finally:
            document.close()

    def test_3d_array_order_is_leading_y_x(self) -> None:
        path = self.temp_path / "linear_3d.fits"
        data = np.zeros((4, 5, 6), dtype=np.float32)

        wcs = WCS(naxis=3)
        wcs.wcs.crpix = [1.0, 1.0, 1.0]
        wcs.wcs.crval = [100.0, 200.0, 300.0]
        wcs.wcs.cdelt = [1.0, 10.0, 100.0]
        wcs.wcs.ctype = ["LINEAR", "LINEAR", "LINEAR"]

        fits.PrimaryHDU(data=data, header=wcs.to_header()).writeto(path)

        document = FitsDocument.open(path)
        try:
            result = document.world_at(
                0,
                x=2,
                y=3,
                leading_indices=(1,),
            )

            self.assertTrue(result.has_world_coordinates)
            # array order (z=1, y=3, x=2) is passed to the APE-14 method,
            # which maps to FITS pixel order (x=2, y=3, z=1).
            np.testing.assert_allclose(
                np.asarray(result.world_values, dtype=float),
                np.asarray([102.0, 230.0, 400.0]),
                rtol=0.0,
                atol=1e-10,
            )
        finally:
            document.close()

    def test_world_at_rejects_out_of_bounds_pixel(self) -> None:
        path = self.temp_path / "bounds.fits"
        fits.PrimaryHDU(np.zeros((5, 7), dtype=np.float32)).writeto(path)

        document = FitsDocument.open(path)
        try:
            with self.assertRaises(IndexError):
                document.world_at(0, x=7, y=0)
            with self.assertRaises(IndexError):
                document.world_at(0, x=0, y=5)
        finally:
            document.close()

    def test_cursor_text_uses_original_value_and_unit(self) -> None:
        result = PixelCoordinateProvider("No WCS").array_index_to_world((2, 3))
        text = format_cursor_readout(
            x=3,
            y=2,
            value=np.uint16(65535),
            result=result,
            data_unit="adu",
        )

        self.assertIn("Pixel: x=3, y=2", text)
        self.assertIn("Value: 65535 adu", text)
        self.assertIn("No WCS", text)


if __name__ == "__main__":
    unittest.main()
