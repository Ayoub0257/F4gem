"""FITS document abstraction used by the FAR_spectro viewer.

This module keeps the FITS file open while the viewer tab is active, exposes a
small description of every HDU, and extracts only the currently requested 2D
plane from N-dimensional image data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits


IMAGE_HDU_TYPES = (
    fits.PrimaryHDU,
    fits.ImageHDU,
    fits.CompImageHDU,
)

TABLE_HDU_TYPES = (
    fits.BinTableHDU,
    fits.TableHDU,
)


@dataclass(frozen=True)
class HDUDescriptor:
    """Lightweight information required to populate the HDU selector."""

    index: int
    name: str
    hdu_type: str
    category: str
    shape: tuple[int, ...] | None
    dtype: str | None
    row_count: int | None = None
    column_count: int | None = None

    @property
    def dimension_label(self) -> str:
        if self.category == "table":
            rows = self.row_count if self.row_count is not None else 0
            columns = self.column_count if self.column_count is not None else 0
            return f"{rows} rows × {columns} columns"

        if self.shape is None:
            return "No data"

        return " × ".join(str(size) for size in self.shape)


class FitsDocument:
    """Own an open FITS file and provide safe access to its HDUs."""

    VALID_EXTENSIONS = (".fit", ".fits", ".fts")

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.hdul: fits.HDUList | None = None
        self._descriptors: list[HDUDescriptor] = []

    @classmethod
    def open(cls, path: str | Path) -> "FitsDocument":
        document = cls(path)
        document._open()
        return document

    def _open(self) -> None:
        if not self.path.exists() or not self.path.is_file():
            raise FileNotFoundError(f"FITS file does not exist: {self.path}")

        if self.path.suffix.lower() not in self.VALID_EXTENSIONS:
            raise ValueError(f"File is not a supported FITS file: {self.path.name}")

        try:
            self.hdul = fits.open(
                self.path,
                mode="readonly",
                memmap=True,
                lazy_load_hdus=True,
                ignore_missing_end=True,
            )
            self._descriptors = self._inspect_hdus()
        except Exception:
            self.close()
            raise

        if not self._descriptors:
            self.close()
            raise ValueError("The FITS file contains no HDUs.")

    @property
    def descriptors(self) -> tuple[HDUDescriptor, ...]:
        return tuple(self._descriptors)

    def descriptor(self, index: int) -> HDUDescriptor:
        self._validate_index(index)
        return self._descriptors[index]

    def hdu(self, index: int):
        self._validate_index(index)
        assert self.hdul is not None
        return self.hdul[index]

    def data(self, index: int):
        """Return non-image HDU data.

        Image HDUs must be accessed through :meth:`extract_1d_array` or
        :meth:`extract_2d_slice`.  This prevents callers from accidentally
        materializing an entire scaled N-dimensional image in memory.
        """

        descriptor = self.descriptor(index)
        if descriptor.category in {"array_1d", "image_2d", "image_nd"}:
            raise RuntimeError(
                "Image HDUs must be read through extract_1d_array() or "
                "extract_2d_slice()."
            )
        return self.hdu(index).data

    def header(self, index: int) -> fits.Header:
        return self.hdu(index).header

    def header_text(self, index: int) -> str:
        return self.header(index).tostring(
            sep="\n",
            endcard=False,
            padding=False,
        )

    def extract_2d_slice(
        self,
        index: int,
        leading_indices: Iterable[int] = (),
    ) -> np.ndarray:
        """Extract one 2D image from a 2D or N-dimensional image HDU.

        NumPy's final two dimensions are treated as image Y and X. Every
        earlier dimension is selected with one integer index.
        """

        descriptor = self.descriptor(index)
        if descriptor.category not in {"image_2d", "image_nd"}:
            raise TypeError(
                f"HDU {index} is not image data; category={descriptor.category!r}."
            )

        shape = descriptor.shape
        if shape is None:
            raise ValueError(f"HDU {index} contains no image data.")

        indices = tuple(int(value) for value in leading_indices)
        expected = len(shape) - 2
        if len(indices) != expected:
            raise ValueError(
                f"Expected {expected} leading-axis indices for shape {shape}, "
                f"received {len(indices)}."
            )

        for axis, (value, axis_size) in enumerate(zip(indices, shape[:-2])):
            if not 0 <= value < axis_size:
                raise IndexError(
                    f"Index {value} is outside axis {axis} with size {axis_size}."
                )

        selection = indices + (slice(None), slice(None))
        return self._read_image_selection(index, selection)

    def extract_1d_array(self, index: int) -> np.ndarray:
        """Read a one-dimensional image HDU without loading unrelated data."""

        descriptor = self.descriptor(index)
        if descriptor.category != "array_1d":
            raise TypeError(
                f"HDU {index} is not a 1D image; category={descriptor.category!r}."
            )
        return self._read_image_selection(index, (slice(None),))

    def _read_image_selection(
        self,
        index: int,
        selection: tuple[int | slice, ...],
    ) -> np.ndarray:
        """Read one image selection using the cheapest correct access path.

        Ordinary, unscaled image HDUs remain memory-mapped and slicing returns
        a view onto the mapped file.  Images that contain FITS storage
        transforms (``BSCALE``, ``BZERO`` or ``BLANK``), and tile-compressed
        images, are read through Astropy's ``section`` interface.  ``section``
        decodes only the requested plane instead of forcing the complete cube
        into memory.
        """

        hdu = self.hdu(index)

        if self._requires_section_access(hdu):
            result = hdu.section[selection]
        else:
            data = hdu.data
            if data is None:
                raise ValueError(f"HDU {index} contains no image data.")
            result = data[selection]

        return np.asanyarray(result)

    @staticmethod
    def _requires_section_access(hdu: Any) -> bool:
        """Return whether direct ``hdu.data`` access is unsuitable.

        ``hdu.data`` cannot remain memory-mapped when FITS integer storage must
        be decoded through BSCALE/BZERO/BLANK.  The section interface performs
        that decoding only for the requested slice.  It is also the efficient
        path for internally tile-compressed images.
        """

        if isinstance(hdu, fits.CompImageHDU):
            return True

        header = hdu.header
        try:
            bscale = float(header.get("BSCALE", 1))
            bzero = float(header.get("BZERO", 0))
            bitpix = int(header.get("BITPIX", 0) or 0)
        except (TypeError, ValueError):
            # A malformed scaling card should be interpreted by Astropy's FITS
            # section reader rather than by the direct memory-map path.
            return True

        has_active_blank = bitpix > 0 and "BLANK" in header
        return bscale != 1.0 or bzero != 0.0 or has_active_blank

    def axis_names(self, index: int) -> list[str]:
        """Return useful labels for NumPy-order axes when metadata permits."""

        descriptor = self.descriptor(index)
        if descriptor.shape is None:
            return []

        ndim = len(descriptor.shape)
        header = self.header(index)
        names: list[str] = []

        # FITS CTYPE1 is the last NumPy axis, so reverse the FITS axis number.
        for numpy_axis in range(ndim):
            fits_axis = ndim - numpy_axis
            ctype = str(header.get(f"CTYPE{fits_axis}", "")).strip()
            names.append(ctype if ctype else "")

        primary_header = self.header(0)
        telescope = str(
            header.get("TELESCOP", primary_header.get("TELESCOP", ""))
        ).strip().upper()
        data_model = str(
            header.get("DATAMODL", primary_header.get("DATAMODL", ""))
        ).strip().lower()

        if telescope == "JWST":
            if ndim == 4:
                defaults = ["Integration", "Group", "Y", "X"]
            elif ndim == 3 and "rateints" in data_model:
                defaults = ["Integration", "Y", "X"]
            elif ndim == 3 and "cube" in data_model:
                defaults = ["Plane", "Y", "X"]
            else:
                defaults = [f"Axis {axis}" for axis in range(ndim)]
        else:
            defaults = [f"Axis {axis}" for axis in range(ndim)]
            if ndim >= 2:
                defaults[-2:] = ["Y", "X"]

        return [name or defaults[axis] for axis, name in enumerate(names)]

    def preferred_hdu_index(self) -> int:
        """Choose the most useful HDU to show when a document is opened."""

        category_priority = {
            "image_2d": 0,
            "image_nd": 1,
            "table": 2,
            "array_1d": 3,
            "empty": 4,
            "unsupported": 5,
        }
        def preference(item: HDUDescriptor):
            is_science_image = (
                item.name.strip().upper() == "SCI"
                and item.category in {"image_2d", "image_nd"}
            )
            return (
                0 if is_science_image else 1,
                category_priority.get(item.category, 99),
                item.index,
            )

        return min(self._descriptors, key=preference).index

    def close(self) -> None:
        if self.hdul is not None:
            self.hdul.close()
            self.hdul = None
        self._descriptors = []

    def _inspect_hdus(self) -> list[HDUDescriptor]:
        assert self.hdul is not None
        descriptors: list[HDUDescriptor] = []

        for index, hdu in enumerate(self.hdul):
            category = self._classify_hdu(hdu)
            name = str(getattr(hdu, "name", "") or "PRIMARY")
            hdu_type = type(hdu).__name__
            shape: tuple[int, ...] | None = None
            dtype: str | None = None
            row_count: int | None = None
            column_count: int | None = None

            if category == "table":
                row_count = int(hdu.header.get("NAXIS2", 0) or 0)
                columns = getattr(hdu, "columns", None)
                column_count = len(columns) if columns is not None else 0
                shape = (row_count, column_count)
                dtype = "table"
            elif category != "empty":
                ndim = int(hdu.header.get("NAXIS", 0) or 0)
                shape = tuple(
                    int(hdu.header.get(f"NAXIS{fits_axis}", 0) or 0)
                    for fits_axis in range(ndim, 0, -1)
                )

                # BITPIX gives a useful type without forcing large image data
                # into memory. Exact NumPy dtype is filled when the HDU is used.
                bitpix = hdu.header.get("BITPIX")
                dtype = self._dtype_from_bitpix(bitpix)

            descriptors.append(
                HDUDescriptor(
                    index=index,
                    name=name,
                    hdu_type=hdu_type,
                    category=category,
                    shape=shape,
                    dtype=dtype,
                    row_count=row_count,
                    column_count=column_count,
                )
            )

        return descriptors

    @staticmethod
    def _classify_hdu(hdu: Any) -> str:
        if isinstance(hdu, TABLE_HDU_TYPES):
            return "table"

        if isinstance(hdu, IMAGE_HDU_TYPES):
            ndim = int(hdu.header.get("NAXIS", 0) or 0)
            if ndim == 0:
                return "empty"
            if ndim == 1:
                return "array_1d"
            if ndim == 2:
                return "image_2d"
            if ndim >= 3:
                return "image_nd"

        return "unsupported"

    @staticmethod
    def _dtype_from_bitpix(bitpix: Any) -> str | None:
        mapping = {
            8: "uint8",
            16: "int16",
            32: "int32",
            64: "int64",
            -32: "float32",
            -64: "float64",
        }
        try:
            return mapping.get(int(bitpix), f"BITPIX={bitpix}")
        except (TypeError, ValueError):
            return None

    def _validate_index(self, index: int) -> None:
        if self.hdul is None:
            raise RuntimeError("The FITS document is closed.")
        if not 0 <= index < len(self.hdul):
            raise IndexError(f"HDU index {index} is outside this FITS document.")
