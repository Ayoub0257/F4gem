# FAR_spectro project structure

## Application shell

- `main_spectro.py` — creates the Qt application and launches `MainWindow`.
- `UI/main_window.py` — assembles the main window, explorer/viewer splitter, menu bar, and calibration controller.
- `UI/menu_bar.py` — defines File/Open and calibration menu actions.

## FITS viewing (`Read_fits/`)

- `viewer.py` — top-level FITS viewing controller and Qt container. Opens files in a worker thread and routes the selected HDU to the correct viewer.
- `file_loader.py` — file dialog plus FITS document opening; retains the legacy `CCDData` reader for callers that explicitly need it.
- `fits_document.py` — owns the open `HDUList`, inspects extensions, chooses the initial HDU, reads headers, and extracts 2D planes from N-dimensional arrays.
- `hdu_selector.py` — HDU/extension combo box and selected-HDU metadata.
- `axis_selector.py` — dynamic selectors for leading dimensions of 3D, 4D, and higher-dimensional arrays.
- `image_display.py` — final renderer for one 2D numerical plane using PyQtGraph and Astropy normalization.
- `table_viewer.py` — virtualized viewer/model for `TableHDU` and `BinTableHDU` data.
- `array_plot.py` — plots one-dimensional image HDUs.
- `header_viewer.py` — read-only FITS-header display.
- `drop_area.py` — initial FITS drag-and-drop target.
- `GUI_utils.py` — shared error dialog helper.
- `control_panel.py` — older/optional file-control widget; currently not used by `FitsViewer`.

## File explorer (`Folder_Exolprer/`)

- `File_explorer.py` — active explorer widget used by `MainWindow`; navigates folders and emits FITS paths on double-click.
- `file_system_model.py` — Qt item model for filesystem rows and columns.
- `file_system_item.py` — lazy filesystem node representation.
- `DropArea.py` — folder-drop overlay used by the older explorer implementation.
- `explorer_widget.py` — older, more elaborate explorer implementation; `MainWindow` currently imports `Folder_Exolprer/File_explorer.py` instead.

## Spectroscopic calibration (`Spectro/`)

- `calibration_controller.py` — connects calibration UI actions to a worker thread and reports progress/errors.
- `calibration_manager.py` — creates session folders, loads configuration, classifies files, and orchestrates calibration stages. It now excludes viewer-only JWST/N-D/table products from the CCD pipeline.
- `master_processor.py` — creates master bias, exposure-dependent darks/flats, and master lamp frames.
- `wavelength_processor.py` — detects/matches lamp lines and writes the dispersion solution.
- `science_processor.py` — applies bias/dark/flat corrections, saves calibrated 2D frames, extracts 1D spectra, and applies the dispersion model.
- `spectrum_plot.py` — displays extracted wavelength/intensity text spectra.

## Test utility

- `tests/create_fits_viewer_test_file.py` — creates one synthetic FITS file containing an empty primary HDU, 2D image, 3D cube, JWST-like 4D image, 1D array, scalar table, and vector-cell table.
