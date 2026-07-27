"""Create one FITS file that exercises every FAR_spectro viewer mode.

Run from the project root:
    python tests/create_fits_viewer_test_file.py
"""

from pathlib import Path

import numpy as np
from astropy.io import fits


def main() -> None:
    rng = np.random.default_rng(42)

    primary = fits.PrimaryHDU()

    image_2d = fits.ImageHDU(
        data=rng.normal(size=(100, 200)).astype(np.float32),
        name="IMAGE_2D",
    )

    # Unsigned integer FITS images are stored using FITS BZERO/BSCALE cards.
    # These HDUs exercise the optimized section-reading path that avoids the
    # Astropy memmap/scaling conflict.
    scaled_uint16_2d_data = np.arange(80 * 120, dtype=np.uint16).reshape(80, 120)
    scaled_uint16_2d = fits.ImageHDU(
        data=scaled_uint16_2d_data,
        name="UINT16_2D",
    )

    cube_3d = fits.ImageHDU(
        data=rng.normal(size=(6, 80, 120)).astype(np.float32),
        name="CUBE_3D",
    )

    jwst_like_4d = fits.ImageHDU(
        data=rng.normal(size=(3, 5, 32, 64)).astype(np.float32),
        name="SCI",
    )
    jwst_like_4d.header["TELESCOP"] = "JWST"
    jwst_like_4d.header["DATAMODL"] = "RampModel"

    scaled_uint16_4d_data = np.arange(
        2 * 3 * 24 * 32,
        dtype=np.uint16,
    ).reshape(2, 3, 24, 32)
    scaled_uint16_4d = fits.ImageHDU(
        data=scaled_uint16_4d_data,
        name="UINT16_4D",
    )

    array_1d = fits.ImageHDU(
        data=np.sin(np.linspace(0, 8 * np.pi, 500)).astype(np.float32),
        name="ARRAY_1D",
    )

    scalar_columns = [
        fits.Column(
            name="WAVELENGTH",
            format="D",
            unit="um",
            array=np.linspace(1.0, 5.0, 100),
        ),
        fits.Column(
            name="FLUX",
            format="D",
            unit="Jy",
            array=rng.random(100),
        ),
    ]
    spectrum_table = fits.BinTableHDU.from_columns(
        scalar_columns,
        name="SPECTRUM",
    )

    vector_column = fits.Column(
        name="VECTOR_CELL",
        format="10E",
        array=rng.random((8, 10)).astype(np.float32),
    )
    vector_table = fits.BinTableHDU.from_columns(
        [vector_column],
        name="VECTOR_TABLE",
    )

    output = Path(__file__).with_name("fits_viewer_cases.fits")
    fits.HDUList(
        [
            primary,
            image_2d,
            scaled_uint16_2d,
            cube_3d,
            jwst_like_4d,
            scaled_uint16_4d,
            array_1d,
            spectrum_table,
            vector_table,
        ]
    ).writeto(output, overwrite=True)

    print(f"Created: {output}")


if __name__ == "__main__":
    main()
