"""Regression tests for memory-efficient FITS image access.

Run from the project root with:
    python -m unittest tests.test_fits_document
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from Read_fits.fits_document import FitsDocument


class FitsDocumentImageAccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self._temp_dir.name) / "image_access_cases.fits"

        self.float_2d = np.arange(20 * 30, dtype=np.float32).reshape(20, 30)
        self.uint16_2d = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
        self.uint16_4d = np.arange(
            2 * 3 * 10 * 12,
            dtype=np.uint16,
        ).reshape(2, 3, 10, 12)

        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(self.float_2d, name="FLOAT_2D"),
                fits.ImageHDU(self.uint16_2d, name="UINT16_2D"),
                fits.ImageHDU(self.uint16_4d, name="UINT16_4D"),
            ]
        ).writeto(self.path)

        self.document = FitsDocument.open(self.path)

    def tearDown(self) -> None:
        self.document.close()
        self._temp_dir.cleanup()

    def _index(self, name: str) -> int:
        for descriptor in self.document.descriptors:
            if descriptor.name == name:
                return descriptor.index
        raise AssertionError(f"Missing test HDU: {name}")

    def test_unscaled_2d_image(self) -> None:
        index = self._index("FLOAT_2D")
        actual = self.document.extract_2d_slice(index)
        np.testing.assert_array_equal(actual, self.float_2d)

    def test_scaled_unsigned_2d_image(self) -> None:
        index = self._index("UINT16_2D")
        hdu = self.document.hdu(index)
        self.assertTrue(self.document._requires_section_access(hdu))

        actual = self.document.extract_2d_slice(index)
        self.assertEqual(actual.dtype, np.dtype(np.uint16))
        np.testing.assert_array_equal(actual, self.uint16_2d)

    def test_scaled_unsigned_4d_reads_only_selected_plane(self) -> None:
        index = self._index("UINT16_4D")
        actual = self.document.extract_2d_slice(index, (1, 2))

        self.assertEqual(actual.shape, (10, 12))
        self.assertEqual(actual.dtype, np.dtype(np.uint16))
        np.testing.assert_array_equal(actual, self.uint16_4d[1, 2])

    def test_image_data_api_blocks_full_cube_materialization(self) -> None:
        index = self._index("UINT16_4D")
        with self.assertRaises(RuntimeError):
            self.document.data(index)


if __name__ == "__main__":
    unittest.main()
