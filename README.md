# XGIS Deconvolution Tools

This repository contains tools for processing and analyzing SIXTE  simulated data from the XGIS coded mask instrument on the THESEUS satellite. The core script, `deconvolve.py`, reconstructs sky images, spectra, and lightcurves from input event FITS files by cross-correlating the detector image with the mask pattern. It supports generating standard FITS outputs for sky maps and lightcurves, as well as PHA files for spectral analysis, utilizing `astropy`, `numpy`, and `scipy`.

For a detailed explanation of the instrument geometry, the deconvolution algorithm (correlation via FFT), and usage examples, have a look at the  project documentation located in the `documentation/` directory. The main document, [`documentation/main.pdf`](documentation/main.pdf), provides a comprehensive guide to the physical principles behind the mask decoding and the implementation details within the SIXTE simulation framework.

