"""Coordinate-system support for the FAR_spectro FITS viewer.

The viewer works internally with NumPy array order. For a normal image this is
``(y, x)``; for an N-dimensional image it is
``(leading_axis_0, ..., y, x)``. Astropy's APE-14 WCS interface provides
``array_index_to_world_values`` specifically for this ordering.

This module supports ordinary FITS-WCS, pixel-only fallback, equatorial
RA/Dec formatting, manual celestial-coordinate parsing, inverse WCS lookup,
and constant-RA/constant-Dec overlay generation. The provider abstraction is
kept compatible with the shared Astropy WCS API so a GWCS backend can be added
later without redesigning the viewer UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS


COORDINATE_FORMAT_BOTH = "both"
COORDINATE_FORMAT_SEXAGESIMAL = "sexagesimal"
COORDINATE_FORMAT_DECIMAL = "decimal"
COORDINATE_FORMAT_OPTIONS = (
    ("Both", COORDINATE_FORMAT_BOTH),
    ("Sexagesimal", COORDINATE_FORMAT_SEXAGESIMAL),
    ("Decimal degrees", COORDINATE_FORMAT_DECIMAL),
)


@dataclass(frozen=True)
class CoordinateResult:
    """Result of converting one NumPy-array position to world coordinates."""

    backend: str
    array_indices: tuple[float, ...]
    has_world_coordinates: bool
    world_values: tuple[Any, ...] = ()
    world_axis_names: tuple[str, ...] = ()
    physical_types: tuple[str | None, ...] = ()
    units: tuple[str, ...] = ()
    ctypes: tuple[str, ...] = ()
    message: str | None = None


@dataclass(frozen=True)
class PixelCoordinateResult:
    """Result of an inverse celestial-world-to-array transformation."""

    backend: str
    success: bool
    array_coordinates: tuple[float, ...] = ()
    x: float | None = None
    y: float | None = None
    inside_image: bool = False
    message: str | None = None


@dataclass(frozen=True)
class CelestialLineResult:
    """Sampled pixel polyline for a constant-RA or constant-Dec curve."""

    backend: str
    success: bool
    x_values: np.ndarray | None = None
    y_values: np.ndarray | None = None
    message: str | None = None


class CoordinateProvider:
    """Interface shared by FITS-WCS, future GWCS, and pixel fallback."""

    backend = "pixel"
    frame_name = "Unknown"

    @property
    def has_celestial_coordinates(self) -> bool:
        return False

    def array_index_to_world(
        self,
        array_indices: Sequence[float],
    ) -> CoordinateResult:
        raise NotImplementedError

    def celestial_to_array_coordinates(
        self,
        *,
        ra_deg: float,
        dec_deg: float,
        reference_array_indices: Sequence[float],
        image_shape: Sequence[int],
    ) -> PixelCoordinateResult:
        return PixelCoordinateResult(
            backend=self.backend,
            success=False,
            message="This HDU has no celestial RA/Dec coordinate solution.",
        )

    def celestial_line(
        self,
        *,
        ra_deg: float | None,
        dec_deg: float | None,
        reference_array_indices: Sequence[float],
        image_shape: Sequence[int],
        samples: int = 192,
    ) -> CelestialLineResult:
        return CelestialLineResult(
            backend=self.backend,
            success=False,
            message="This HDU has no celestial RA/Dec coordinate solution.",
        )


class PixelCoordinateProvider(CoordinateProvider):
    """Fallback provider used when no valid physical WCS is available."""

    backend = "pixel"

    def __init__(self, reason: str = "No world-coordinate solution is available."):
        self.reason = reason

    def array_index_to_world(
        self,
        array_indices: Sequence[float],
    ) -> CoordinateResult:
        indices = tuple(float(value) for value in array_indices)
        return CoordinateResult(
            backend=self.backend,
            array_indices=indices,
            has_world_coordinates=False,
            message=self.reason,
        )


class WCSAPIProvider(CoordinateProvider):
    """Provider for objects implementing Astropy's low-level APE-14 WCS API."""

    def __init__(
        self,
        wcs_object: Any,
        *,
        backend: str,
        ctypes: Sequence[str] = (),
        frame_name: str = "Unknown",
    ):
        self._wcs = wcs_object
        self.backend = backend
        self.frame_name = frame_name or "Unknown"
        self._pixel_n_dim = int(wcs_object.pixel_n_dim)
        self._world_n_dim = int(wcs_object.world_n_dim)

        self._world_axis_names = _normalise_text_sequence(
            getattr(wcs_object, "world_axis_names", ()),
            self._world_n_dim,
        )
        self._physical_types = _normalise_optional_text_sequence(
            getattr(wcs_object, "world_axis_physical_types", ()),
            self._world_n_dim,
        )
        self._units = _normalise_text_sequence(
            getattr(wcs_object, "world_axis_units", ()),
            self._world_n_dim,
        )
        self._ctypes = _normalise_text_sequence(ctypes, self._world_n_dim)

        self._ra_axis = _find_equatorial_axis(
            self._physical_types,
            self._ctypes,
            target="ra",
        )
        self._dec_axis = _find_equatorial_axis(
            self._physical_types,
            self._ctypes,
            target="dec",
        )

    @property
    def pixel_n_dim(self) -> int:
        return self._pixel_n_dim

    @property
    def has_celestial_coordinates(self) -> bool:
        return self._ra_axis is not None and self._dec_axis is not None

    def array_index_to_world(
        self,
        array_indices: Sequence[float],
    ) -> CoordinateResult:
        indices = tuple(float(value) for value in array_indices)

        if len(indices) != self._pixel_n_dim:
            return CoordinateResult(
                backend=self.backend,
                array_indices=indices,
                has_world_coordinates=False,
                message=(
                    f"The coordinate solution expects {self._pixel_n_dim} pixel "
                    f"axes, but {len(indices)} array indices were supplied."
                ),
            )

        try:
            transformed = self._wcs.array_index_to_world_values(*indices)
            values = _normalise_world_values(transformed, self._world_n_dim)
        except Exception as exc:
            return CoordinateResult(
                backend=self.backend,
                array_indices=indices,
                has_world_coordinates=False,
                message=f"World-coordinate conversion failed: {exc}",
            )

        visible_axes = tuple(
            bool(name or physical_type or ctype)
            for name, physical_type, ctype in zip(
                self._world_axis_names,
                self._physical_types,
                self._ctypes,
            )
        )
        has_defined_value = any(
            visible and _is_defined_scalar(value)
            for visible, value in zip(visible_axes, values)
        )

        return CoordinateResult(
            backend=self.backend,
            array_indices=indices,
            has_world_coordinates=has_defined_value,
            world_values=values,
            world_axis_names=self._world_axis_names,
            physical_types=self._physical_types,
            units=self._units,
            ctypes=self._ctypes,
            message=(
                None
                if has_defined_value
                else "The WCS returned no defined physical coordinate at this pixel."
            ),
        )

    def celestial_to_array_coordinates(
        self,
        *,
        ra_deg: float,
        dec_deg: float,
        reference_array_indices: Sequence[float],
        image_shape: Sequence[int],
    ) -> PixelCoordinateResult:
        if not self.has_celestial_coordinates:
            return PixelCoordinateResult(
                backend=self.backend,
                success=False,
                message="This WCS does not define both RA and Dec axes.",
            )

        reference = tuple(float(value) for value in reference_array_indices)
        if len(reference) != self._pixel_n_dim:
            return PixelCoordinateResult(
                backend=self.backend,
                success=False,
                message=(
                    f"The WCS expects {self._pixel_n_dim} reference array "
                    f"coordinates, received {len(reference)}."
                ),
            )

        base = self.array_index_to_world(reference)
        if not base.has_world_coordinates:
            return PixelCoordinateResult(
                backend=self.backend,
                success=False,
                message=base.message or "Could not establish a WCS reference point.",
            )

        world_values = list(base.world_values)
        assert self._ra_axis is not None
        assert self._dec_axis is not None
        world_values[self._ra_axis] = self._degrees_to_native_world(
            self._ra_axis,
            float(ra_deg) % 360.0,
        )
        world_values[self._dec_axis] = self._degrees_to_native_world(
            self._dec_axis,
            float(dec_deg),
        )

        try:
            pixel_values = self._wcs.world_to_pixel_values(*world_values)
            pixel_tuple = _normalise_pixel_values(pixel_values, self._pixel_n_dim)
            array_coordinates = tuple(reversed(pixel_tuple))
        except Exception as exc:
            return PixelCoordinateResult(
                backend=self.backend,
                success=False,
                message=f"Inverse WCS conversion failed: {exc}",
            )

        if not all(np.isfinite(float(value)) for value in array_coordinates):
            return PixelCoordinateResult(
                backend=self.backend,
                success=False,
                array_coordinates=array_coordinates,
                message="The coordinate does not map to a finite image position.",
            )

        if len(array_coordinates) < 2:
            return PixelCoordinateResult(
                backend=self.backend,
                success=False,
                array_coordinates=array_coordinates,
                message="The WCS did not return two image coordinates.",
            )

        y_value = float(array_coordinates[-2])
        x_value = float(array_coordinates[-1])
        height, width = tuple(int(size) for size in image_shape)[-2:]
        inside = 0.0 <= x_value < float(width) and 0.0 <= y_value < float(height)

        leading_reference = np.asarray(reference[:-2], dtype=float)
        leading_result = np.asarray(array_coordinates[:-2], dtype=float)
        maps_to_slice = True
        if leading_reference.size:
            maps_to_slice = bool(
                leading_reference.shape == leading_result.shape
                and np.all(np.abs(leading_result - leading_reference) <= 0.75)
            )

        if not maps_to_slice:
            message = "The coordinate maps to a different N-dimensional slice."
        elif not inside:
            message = "The coordinate is outside the current image bounds."
        else:
            message = None

        return PixelCoordinateResult(
            backend=self.backend,
            success=maps_to_slice,
            array_coordinates=array_coordinates,
            x=x_value,
            y=y_value,
            inside_image=inside and maps_to_slice,
            message=message,
        )

    def celestial_line(
        self,
        *,
        ra_deg: float | None,
        dec_deg: float | None,
        reference_array_indices: Sequence[float],
        image_shape: Sequence[int],
        samples: int = 192,
    ) -> CelestialLineResult:
        if not self.has_celestial_coordinates:
            return CelestialLineResult(
                backend=self.backend,
                success=False,
                message="This WCS does not define both RA and Dec axes.",
            )

        if (ra_deg is None) == (dec_deg is None):
            return CelestialLineResult(
                backend=self.backend,
                success=False,
                message="Supply exactly one fixed coordinate for a coordinate line.",
            )

        reference = tuple(float(value) for value in reference_array_indices)
        if len(reference) != self._pixel_n_dim:
            return CelestialLineResult(
                backend=self.backend,
                success=False,
                message="The WCS reference dimensionality is invalid.",
            )

        base = self.array_index_to_world(reference)
        if not base.has_world_coordinates:
            return CelestialLineResult(
                backend=self.backend,
                success=False,
                message=base.message or "Could not establish a WCS reference point.",
            )

        height, width = tuple(int(size) for size in image_shape)[-2:]
        boundary_ra, boundary_dec = self._sample_image_boundary(
            reference,
            height=height,
            width=width,
        )
        if boundary_ra.size < 2 or boundary_dec.size < 2:
            return CelestialLineResult(
                backend=self.backend,
                success=False,
                message="Could not determine the celestial footprint of this image.",
            )

        sample_count = max(32, min(int(samples), 1024))
        reference_ra = float(np.nanmedian(boundary_ra))
        unwrapped_boundary_ra = _unwrap_degrees(boundary_ra, reference_ra)
        ra_min = float(np.nanmin(unwrapped_boundary_ra))
        ra_max = float(np.nanmax(unwrapped_boundary_ra))
        dec_min = float(np.nanmin(boundary_dec))
        dec_max = float(np.nanmax(boundary_dec))

        if ra_deg is not None:
            fixed_ra = _align_degrees_to_reference(float(ra_deg), reference_ra)
            ra_samples = np.full(sample_count, fixed_ra, dtype=float)
            dec_samples = np.linspace(dec_min, dec_max, sample_count, dtype=float)
        else:
            assert dec_deg is not None
            ra_samples = np.linspace(ra_min, ra_max, sample_count, dtype=float)
            dec_samples = np.full(sample_count, float(dec_deg), dtype=float)

        world_arrays: list[Any] = []
        assert self._ra_axis is not None
        assert self._dec_axis is not None
        for axis, value in enumerate(base.world_values):
            if axis == self._ra_axis:
                world_arrays.append(self._degrees_array_to_native_world(axis, ra_samples))
            elif axis == self._dec_axis:
                world_arrays.append(self._degrees_array_to_native_world(axis, dec_samples))
            else:
                world_arrays.append(np.full(sample_count, value))

        try:
            pixel_values = self._wcs.world_to_pixel_values(*world_arrays)
            pixel_tuple = _normalise_pixel_arrays(pixel_values, self._pixel_n_dim)
            array_arrays = tuple(reversed(pixel_tuple))
            y_values = np.asarray(array_arrays[-2], dtype=float)
            x_values = np.asarray(array_arrays[-1], dtype=float)
        except Exception as exc:
            return CelestialLineResult(
                backend=self.backend,
                success=False,
                message=f"Could not project the celestial coordinate line: {exc}",
            )

        finite = np.isfinite(x_values) & np.isfinite(y_values)
        near_image = (
            (x_values >= -1.0)
            & (x_values <= float(width))
            & (y_values >= -1.0)
            & (y_values <= float(height))
        )
        valid = finite & near_image

        leading_arrays = array_arrays[:-2]
        for axis, leading in enumerate(leading_arrays):
            expected = reference[axis]
            valid &= np.abs(np.asarray(leading, dtype=float) - expected) <= 0.75

        x_plot = x_values.copy()
        y_plot = y_values.copy()
        x_plot[~valid] = np.nan
        y_plot[~valid] = np.nan

        # Break lines across projection discontinuities instead of drawing a
        # misleading long segment through the image.
        finite_pair = np.isfinite(x_plot[:-1]) & np.isfinite(x_plot[1:])
        jumps = np.zeros_like(x_plot, dtype=bool)
        if finite_pair.any():
            distance = np.hypot(np.diff(x_plot), np.diff(y_plot))
            threshold = max(float(width), float(height)) * 0.35
            jumps[1:] = finite_pair & (distance > threshold)
            x_plot[jumps] = np.nan
            y_plot[jumps] = np.nan

        if np.count_nonzero(np.isfinite(x_plot) & np.isfinite(y_plot)) < 2:
            return CelestialLineResult(
                backend=self.backend,
                success=False,
                x_values=x_plot,
                y_values=y_plot,
                message="The requested coordinate line does not cross this image.",
            )

        return CelestialLineResult(
            backend=self.backend,
            success=True,
            x_values=x_plot,
            y_values=y_plot,
        )

    def _sample_image_boundary(
        self,
        reference: tuple[float, ...],
        *,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample all image edges with one vectorized WCS transformation."""

        edge_samples = 40
        x_axis = np.linspace(0.0, max(0.0, width - 1.0), edge_samples)
        y_axis = np.linspace(0.0, max(0.0, height - 1.0), edge_samples)

        boundary_x = np.concatenate(
            [
                x_axis,
                x_axis,
                np.zeros_like(y_axis),
                np.full_like(y_axis, max(0.0, width - 1.0)),
            ]
        )
        boundary_y = np.concatenate(
            [
                np.zeros_like(x_axis),
                np.full_like(x_axis, max(0.0, height - 1.0)),
                y_axis,
                y_axis,
            ]
        )
        count = boundary_x.size

        array_inputs: list[np.ndarray] = [
            np.full(count, float(value), dtype=float)
            for value in reference[:-2]
        ]
        array_inputs.extend([boundary_y, boundary_x])

        assert self._ra_axis is not None
        assert self._dec_axis is not None
        try:
            transformed = self._wcs.array_index_to_world_values(*array_inputs)
            world_arrays = _normalise_world_arrays(transformed, self._world_n_dim)
            ra_values = self._native_world_array_to_degrees(
                self._ra_axis,
                world_arrays[self._ra_axis],
            )
            dec_values = self._native_world_array_to_degrees(
                self._dec_axis,
                world_arrays[self._dec_axis],
            )
        except Exception:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        valid = np.isfinite(ra_values) & np.isfinite(dec_values)
        return (
            np.asarray(ra_values[valid], dtype=float),
            np.asarray(dec_values[valid], dtype=float),
        )

    def _native_world_to_degrees(self, axis: int, value: Any) -> float:
        unit_text = self._units[axis] if axis < len(self._units) else ""
        unit = _safe_angular_unit(unit_text)
        return float((float(value) * unit).to_value(u.deg))

    def _native_world_array_to_degrees(
        self,
        axis: int,
        values: np.ndarray,
    ) -> np.ndarray:
        unit_text = self._units[axis] if axis < len(self._units) else ""
        unit = _safe_angular_unit(unit_text)
        return np.asarray((np.asarray(values, dtype=float) * unit).to_value(u.deg))

    def _degrees_to_native_world(self, axis: int, value_deg: float) -> float:
        unit_text = self._units[axis] if axis < len(self._units) else ""
        unit = _safe_angular_unit(unit_text)
        return float((float(value_deg) * u.deg).to_value(unit))

    def _degrees_array_to_native_world(
        self,
        axis: int,
        values_deg: np.ndarray,
    ) -> np.ndarray:
        unit_text = self._units[axis] if axis < len(self._units) else ""
        unit = _safe_angular_unit(unit_text)
        return np.asarray((np.asarray(values_deg) * u.deg).to_value(unit), dtype=float)


class FitsWCSProvider(WCSAPIProvider):
    """APE-14 provider backed by :class:`astropy.wcs.WCS`."""

    def __init__(self, wcs_object: WCS, frame_name: str):
        ctypes = tuple(
            str(value or "").strip()
            for value in getattr(wcs_object.wcs, "ctype", ())
        )
        super().__init__(
            wcs_object,
            backend="FITS-WCS",
            ctypes=ctypes,
            frame_name=frame_name,
        )


def build_fits_coordinate_provider(
    header: fits.Header,
    hdul: fits.HDUList,
    data_shape: Sequence[int],
) -> CoordinateProvider:
    """Build a cached coordinate provider for one image HDU."""

    shape = tuple(int(size) for size in data_shape)
    if not shape:
        return PixelCoordinateProvider("The selected HDU contains no image axes.")

    if not _header_has_meaningful_wcs(header, len(shape)):
        return PixelCoordinateProvider(
            "No physical FITS-WCS keywords were found in this HDU."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            try:
                wcs = WCS(
                    header=header,
                    fobj=hdul,
                    relax=True,
                    fix=True,
                    preserve_units=True,
                )
            except TypeError as exc:
                if "preserve_units" not in str(exc):
                    raise
                wcs = WCS(
                    header=header,
                    fobj=hdul,
                    relax=True,
                    fix=True,
                )
    except Exception as exc:
        return PixelCoordinateProvider(f"Could not construct FITS-WCS: {exc}")

    if int(wcs.pixel_n_dim) != len(shape):
        return PixelCoordinateProvider(
            "The FITS-WCS dimensionality does not match the image: "
            f"WCS has {wcs.pixel_n_dim} pixel axes, image shape is {shape}."
        )

    frame_name = _frame_name_from_header(header)
    return FitsWCSProvider(wcs, frame_name=frame_name)


def parse_ra_input(text: str) -> float:
    """Parse RA as decimal degrees or sexagesimal hours and return degrees."""

    cleaned = _clean_angle_text(text)
    if not cleaned:
        raise ValueError("RA is empty.")

    try:
        if _looks_sexagesimal(cleaned, is_ra=True):
            angle = Angle(cleaned, unit=u.hourangle)
            value = float(angle.to_value(u.deg))
        else:
            value = float(cleaned)
    except Exception as exc:
        raise ValueError(
            "Invalid RA. Use decimal degrees (for example 188.73625) or "
            "sexagesimal hours (for example 12:34:56.7)."
        ) from exc

    if not np.isfinite(value):
        raise ValueError("RA must be finite.")
    return value % 360.0


def parse_dec_input(text: str) -> float:
    """Parse declination as decimal or sexagesimal degrees."""

    cleaned = _clean_angle_text(text)
    if not cleaned:
        raise ValueError("Dec is empty.")

    try:
        if _looks_sexagesimal(cleaned, is_ra=False):
            angle = Angle(cleaned, unit=u.deg)
            value = float(angle.to_value(u.deg))
        else:
            value = float(cleaned)
    except Exception as exc:
        raise ValueError(
            "Invalid Dec. Use decimal degrees (for example -24.31167) or "
            "sexagesimal degrees (for example -24:18:42)."
        ) from exc

    if not np.isfinite(value):
        raise ValueError("Dec must be finite.")
    if not -90.0 <= value <= 90.0:
        raise ValueError("Declination must be between -90 and +90 degrees.")
    return value


def format_cursor_readout(
    *,
    x: int,
    y: int,
    value: Any,
    result: CoordinateResult,
    data_unit: str | None = None,
    coordinate_format: str = COORDINATE_FORMAT_BOTH,
) -> str:
    """Create a compact one-line cursor status string."""

    parts = [
        f"Pixel: x={int(x)}, y={int(y)}",
        f"Value: {format_data_value(value, data_unit)}",
    ]
    world_text = format_world_coordinates(
        result,
        coordinate_format=coordinate_format,
    )
    if world_text:
        parts.append(world_text)
    return " | ".join(parts)


def format_world_coordinate_lines(
    result: CoordinateResult,
    *,
    coordinate_format: str = COORDINATE_FORMAT_BOTH,
) -> list[str]:
    """Format world coordinates as separate lines for the side panel."""

    if not result.has_world_coordinates:
        return [result.message or "No world-coordinate solution"]

    mode = normalise_coordinate_format(coordinate_format)
    formatted: list[str] = []
    handled: set[int] = set()
    ra_axis = _find_equatorial_axis(result.physical_types, result.ctypes, "ra")
    dec_axis = _find_equatorial_axis(result.physical_types, result.ctypes, "dec")

    if ra_axis is not None and dec_axis is not None:
        try:
            ra_deg = _coordinate_value_in_degrees(result, ra_axis)
            dec_deg = _coordinate_value_in_degrees(result, dec_axis)
            formatted.extend(_format_equatorial_pair(ra_deg, dec_deg, mode))
            handled.update({ra_axis, dec_axis})
        except Exception:
            # Preserve the generic formatting path when a malformed unit or
            # non-scalar value prevents sexagesimal conversion.
            pass

    for axis, value in enumerate(result.world_values):
        if axis in handled:
            continue

        name = result.world_axis_names[axis] if axis < len(result.world_axis_names) else ""
        physical_type = (
            result.physical_types[axis] if axis < len(result.physical_types) else None
        )
        unit = result.units[axis] if axis < len(result.units) else ""
        ctype = result.ctypes[axis] if axis < len(result.ctypes) else ""

        if not (name or physical_type or ctype):
            continue

        label = _axis_label(
            axis=axis,
            name=name,
            physical_type=physical_type,
            ctype=ctype,
        )
        formatted.append(f"{label}: {_format_world_value(value, unit)}")

    if not formatted:
        return [result.message or "No defined world coordinate"]
    return formatted


def format_world_coordinates(
    result: CoordinateResult,
    *,
    coordinate_format: str = COORDINATE_FORMAT_BOTH,
) -> str:
    """Format scientifically identified world axes in one line."""

    return " | ".join(
        format_world_coordinate_lines(
            result,
            coordinate_format=coordinate_format,
        )
    )


def format_data_value(value: Any, unit: str | None = None) -> str:
    """Format an original science-array value without display scaling."""

    if np.ma.is_masked(value):
        text = "masked"
    else:
        scalar = _to_python_scalar(value)
        if isinstance(scalar, complex):
            text = f"{scalar.real:.7g}{scalar.imag:+.7g}j"
        elif isinstance(scalar, (bool, np.bool_)):
            text = str(bool(scalar))
        elif isinstance(scalar, (int, np.integer)):
            text = str(int(scalar))
        elif isinstance(scalar, (float, np.floating)):
            numeric = float(scalar)
            text = "undefined" if not np.isfinite(numeric) else f"{numeric:.7g}"
        else:
            text = str(scalar)

    clean_unit = str(unit or "").strip()
    return f"{text} {clean_unit}" if clean_unit else text


def normalise_coordinate_format(value: str) -> str:
    clean = str(value or "").strip().lower()
    aliases = {
        "both": COORDINATE_FORMAT_BOTH,
        "sexagesimal": COORDINATE_FORMAT_SEXAGESIMAL,
        "decimal": COORDINATE_FORMAT_DECIMAL,
        "decimal degrees": COORDINATE_FORMAT_DECIMAL,
    }
    return aliases.get(clean, COORDINATE_FORMAT_BOTH)


def _format_equatorial_pair(
    ra_deg: float,
    dec_deg: float,
    mode: str,
) -> list[str]:
    ra_decimal = f"{ra_deg:.8f}°"
    dec_decimal = f"{dec_deg:+.8f}°"
    ra_sexagesimal = Angle(ra_deg, unit=u.deg).to_string(
        unit=u.hourangle,
        sep=":",
        precision=3,
        pad=True,
    )
    dec_sexagesimal = Angle(dec_deg, unit=u.deg).to_string(
        unit=u.deg,
        sep=":",
        precision=2,
        pad=True,
        alwayssign=True,
    )

    if mode == COORDINATE_FORMAT_SEXAGESIMAL:
        return [
            f"RA: {ra_sexagesimal}",
            f"Dec: {dec_sexagesimal}",
        ]
    if mode == COORDINATE_FORMAT_DECIMAL:
        return [
            f"RA: {ra_decimal}",
            f"Dec: {dec_decimal}",
        ]
    return [
        f"RA: {ra_sexagesimal}  ({ra_decimal})",
        f"Dec: {dec_sexagesimal}  ({dec_decimal})",
    ]


def _coordinate_value_in_degrees(result: CoordinateResult, axis: int) -> float:
    value = float(_to_python_scalar(result.world_values[axis]))
    unit_text = result.units[axis] if axis < len(result.units) else ""
    unit = _safe_angular_unit(unit_text)
    return float((value * unit).to_value(u.deg))


def _header_has_meaningful_wcs(header: fits.Header, data_ndim: int) -> bool:
    try:
        wcs_axes = int(header.get("WCSAXES", 0) or 0)
    except (TypeError, ValueError):
        wcs_axes = 0

    axis_count = max(int(data_ndim), wcs_axes)
    return any(
        str(header.get(f"CTYPE{axis}", "") or "").strip()
        for axis in range(1, axis_count + 1)
    )


def _frame_name_from_header(header: fits.Header) -> str:
    radesys = str(
        header.get("RADESYS", header.get("RADECSYS", "")) or ""
    ).strip()
    equinox = header.get("EQUINOX")
    if radesys and equinox not in (None, ""):
        return f"{radesys} (equinox {equinox})"
    if radesys:
        return radesys
    if equinox not in (None, ""):
        return f"Equatorial (equinox {equinox})"
    return "Image WCS frame"


def _normalise_world_values(value: Any, world_n_dim: int) -> tuple[Any, ...]:
    if world_n_dim == 1:
        raw_values = (value,)
    else:
        try:
            raw_values = tuple(value)
        except TypeError:
            raw_values = (value,)

    normalised = tuple(_to_python_scalar(item) for item in raw_values)
    if len(normalised) < world_n_dim:
        normalised += (np.nan,) * (world_n_dim - len(normalised))
    return normalised[:world_n_dim]


def _normalise_world_arrays(value: Any, world_n_dim: int) -> tuple[np.ndarray, ...]:
    if world_n_dim == 1:
        raw_values = (value,)
    else:
        try:
            raw_values = tuple(value)
        except TypeError:
            raw_values = (value,)

    arrays = tuple(np.asarray(item, dtype=float) for item in raw_values)
    if len(arrays) < world_n_dim:
        arrays += tuple(np.asarray(np.nan) for _ in range(world_n_dim - len(arrays)))
    return arrays[:world_n_dim]


def _normalise_pixel_values(value: Any, pixel_n_dim: int) -> tuple[float, ...]:
    if pixel_n_dim == 1:
        raw_values = (value,)
    else:
        try:
            raw_values = tuple(value)
        except TypeError:
            raw_values = (value,)

    values = tuple(float(np.asarray(item).reshape(())) for item in raw_values)
    if len(values) < pixel_n_dim:
        values += (np.nan,) * (pixel_n_dim - len(values))
    return values[:pixel_n_dim]


def _normalise_pixel_arrays(value: Any, pixel_n_dim: int) -> tuple[np.ndarray, ...]:
    if pixel_n_dim == 1:
        raw_values = (value,)
    else:
        try:
            raw_values = tuple(value)
        except TypeError:
            raw_values = (value,)

    arrays = tuple(np.asarray(item, dtype=float) for item in raw_values)
    if len(arrays) < pixel_n_dim:
        arrays += tuple(np.asarray(np.nan) for _ in range(pixel_n_dim - len(arrays)))
    return arrays[:pixel_n_dim]


def _normalise_text_sequence(values: Any, length: int) -> tuple[str, ...]:
    try:
        sequence = tuple(str(value or "").strip() for value in values)
    except TypeError:
        sequence = ()

    if len(sequence) < length:
        sequence += ("",) * (length - len(sequence))
    return sequence[:length]


def _normalise_optional_text_sequence(
    values: Any,
    length: int,
) -> tuple[str | None, ...]:
    try:
        sequence = tuple(
            str(value).strip() if value not in (None, "") else None
            for value in values
        )
    except TypeError:
        sequence = ()

    if len(sequence) < length:
        sequence += (None,) * (length - len(sequence))
    return sequence[:length]


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return value.reshape(()).item()
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_defined_scalar(value: Any) -> bool:
    if np.ma.is_masked(value):
        return False

    scalar = _to_python_scalar(value)
    if isinstance(scalar, complex):
        return bool(np.isfinite(scalar.real) and np.isfinite(scalar.imag))
    if isinstance(scalar, (float, np.floating, int, np.integer)):
        return bool(np.isfinite(scalar))
    return scalar is not None


def _find_equatorial_axis(
    physical_types: Sequence[str | None],
    ctypes: Sequence[str],
    target: str,
) -> int | None:
    target_physical = "pos.eq.ra" if target == "ra" else "pos.eq.dec"
    target_prefix = "RA" if target == "ra" else "DEC"

    for axis, physical_type in enumerate(physical_types):
        if str(physical_type or "").strip().lower() == target_physical:
            return axis
    for axis, ctype in enumerate(ctypes):
        if str(ctype or "").strip().upper().startswith(target_prefix):
            return axis
    return None


def _axis_label(
    *,
    axis: int,
    name: str,
    physical_type: str | None,
    ctype: str,
) -> str:
    physical_labels = {
        "pos.eq.ra": "RA",
        "pos.eq.dec": "Dec",
        "pos.galactic.lon": "Galactic longitude",
        "pos.galactic.lat": "Galactic latitude",
        "em.wl": "Wavelength",
        "em.freq": "Frequency",
        "em.energy": "Energy",
        "spect.dopplerveloc": "Velocity",
        "time": "Time",
    }

    physical_key = str(physical_type or "").strip().lower()
    if physical_key in physical_labels:
        return physical_labels[physical_key]

    clean_name = str(name or "").strip()
    if clean_name:
        return clean_name

    clean_ctype = str(ctype or "").strip().upper()
    if clean_ctype.startswith("RA"):
        return "RA"
    if clean_ctype.startswith("DEC"):
        return "Dec"
    if clean_ctype.startswith(("WAVE", "AWAV")):
        return "Wavelength"
    if clean_ctype.startswith("FREQ"):
        return "Frequency"
    if clean_ctype.startswith(("VELO", "VRAD", "VOPT", "ZOPT")):
        return "Velocity"
    if clean_ctype.startswith("TIME"):
        return "Time"
    if clean_ctype:
        return clean_ctype.split("-")[0].title()

    return f"World {axis + 1}"


def _format_world_value(value: Any, unit: str) -> str:
    scalar = _to_python_scalar(value)

    if np.ma.is_masked(scalar):
        text = "masked"
    elif isinstance(scalar, complex):
        if np.isfinite(scalar.real) and np.isfinite(scalar.imag):
            text = f"{scalar.real:.9g}{scalar.imag:+.9g}j"
        else:
            text = "undefined"
    elif isinstance(scalar, (float, np.floating, int, np.integer)):
        numeric = float(scalar)
        text = f"{numeric:.9g}" if np.isfinite(numeric) else "undefined"
    else:
        text = str(scalar)

    clean_unit = str(unit or "").strip()
    return f"{text} {clean_unit}" if clean_unit else text


def _safe_angular_unit(unit_text: str) -> u.UnitBase:
    clean = str(unit_text or "").strip()
    if not clean:
        return u.deg
    try:
        unit = u.Unit(clean)
        if not unit.is_equivalent(u.deg):
            return u.deg
        return unit
    except Exception:
        return u.deg


def _clean_angle_text(text: str) -> str:
    return (
        str(text or "")
        .strip()
        .replace("−", "-")
        .replace("°", "d")
        .replace("′", "m")
        .replace("″", "s")
    )


def _looks_sexagesimal(text: str, *, is_ra: bool) -> bool:
    lower = text.lower()
    if ":" in lower:
        return True
    if is_ra and any(token in lower for token in ("h", "m", "s")):
        return True
    if not is_ra and any(token in lower for token in ("d", "m", "s")):
        return True
    return False


def _align_degrees_to_reference(value: float, reference: float) -> float:
    return float(value + 360.0 * round((reference - value) / 360.0))


def _unwrap_degrees(values: np.ndarray, reference: float) -> np.ndarray:
    aligned = np.asarray(
        [_align_degrees_to_reference(float(value), reference) for value in values],
        dtype=float,
    )
    radians = np.unwrap(np.deg2rad(aligned))
    return np.rad2deg(radians)
