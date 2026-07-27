# FITS viewer tests

Create the interactive test file:

```bash
python tests/create_fits_viewer_test_file.py
```

Run the document-access regression tests:

```bash
python -m unittest tests.test_fits_document
```

The generated FITS file includes ordinary 2D data, unsigned 16-bit 2D and 4D
images using FITS `BZERO`/`BSCALE`, a 3D cube, a JWST-like 4D cube, a 1D array,
and FITS tables.
