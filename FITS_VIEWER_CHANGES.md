# Multi-HDU and N-dimensional FITS viewer

## Existing files modified

- `Read_fits/file_loader.py`: opens a complete `FitsDocument` instead of reducing every file to `hdul[0].data`.
- `Read_fits/viewer.py`: routes each selected HDU to the correct display and preserves the existing companion `*_1D.txt` spectrum workflow.
- `Read_fits/image_display.py`: remains the final 2D renderer, with additional validation and robust handling of NaN, masked, constant, and complex arrays.
- `Spectro/calibration_manager.py`: keeps JWST, table, empty-primary, and N-dimensional products viewer-only so they are not accidentally sent through the ground-based CCD calibration pipeline.

## New files

- `Read_fits/fits_document.py`: owns the open HDU list, describes HDUs, provides headers, and extracts one 2D plane from N-dimensional images.
- `Read_fits/hdu_selector.py`: selects an HDU/extension from a multi-extension FITS file.
- `Read_fits/axis_selector.py`: dynamically creates one selector for every axis before image Y/X.
- `Read_fits/table_viewer.py`: virtualized `QTableView` for FITS ASCII and binary tables.
- `Read_fits/header_viewer.py`: read-only FITS header display.
- `Read_fits/array_plot.py`: plots one-dimensional image HDUs.
- `tests/create_fits_viewer_test_file.py`: generates a synthetic file containing all supported cases.

## Display routing

- Empty HDU -> header viewer
- 1D image HDU -> line plot
- 2D image HDU -> existing image display
- 3D+ image HDU -> axis selectors -> selected 2D plane -> existing image display
- `TableHDU` / `BinTableHDU` -> table viewer
- Unknown HDU -> explanatory message plus header access

## Scaled-image memmap regression fix

The document reader still opens FITS files with `memmap=True`. Image access is
now adaptive:

- ordinary unscaled images use direct memory-mapped slicing;
- images requiring `BSCALE`, `BZERO`, or `BLANK` decoding use
  `ImageHDU.section[...]`;
- internally tile-compressed images also use `section[...]`;
- only the selected 2D plane of an N-dimensional image is decoded;
- direct full-image access through `FitsDocument.data()` is blocked for image
  HDUs to prevent accidental full-cube materialization.

This restores conventional unsigned-integer CCD FITS support without globally
disabling memory mapping for large cubes. Regression tests are in
`tests/test_fits_document.py`.
