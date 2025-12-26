#! /bin/env  python3

import os
import argparse
import sys
import numpy as np
from scipy import ndimage
from astropy.io import fits 
from astropy.wcs import WCS

# Local imports
from writeData import write_img_file, write_spectrum_file, write_lc_file
from iros import reconst_new, make_img

def main():
    # Parsing input parameters: 
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Deconvolve coded mask event data to generate sky images, spectra, and lightcurves.

This script processes an input event FITS file from a coded mask instrument.
It performs the following steps:
1. Loads the event data and the corresponding mask and weight patterns.
2. Reconstructs a sky image by cross-correlating the detector image with the mask pattern (balanced correlation).
3. Generates a sky map in FITS format centered on the source.
4. Extracts a spectrum (PHA file) for a specified source position by iterating over energy bins.
5. Extracts a lightcurve (FITS file) for the source by iterating over time bins.

The reconstruction uses a 'reconst_new' function which implies a re-weighted or specific reconstruction algorithm suitable for this instrument.
""",
        epilog="""
Example usage:
  python deconvolve.py --ra_src 10.5 --dec_src 20.2 --outfile result_obs1 --infile events.fits

Input Files:
  - Event FITS file (specified by --infile)
  - Mask file (specified by --mask, defaults to internal path)
  - Weight file (specified by --weight, defaults to internal path)

Output Files:
  - <outfile>_sky.fits: Reconstructed sky image.
  - test_spec.pha: Extracted spectrum (currently hardcoded name, see script).
  - test_lc.fits: Extracted lightcurve (currently hardcoded name, see script).
"""
    )

    home = os.environ['HOME']

    parser.add_argument('--ra_src', dest='ra_src', type=float, help='Right Ascension of the target source for spectrum/LC extraction (in degrees).')
    parser.add_argument('--dec_src', dest='dec_src', type=float, help='Declination of the target source for spectrum/LC extraction (in degrees).')
    parser.add_argument('--outfile', dest='outfile', type=str, help='Prefix string for the output sky image filename (e.g., "output" -> "output_sky.fits").')
    parser.add_argument('--infile', dest='infile', type=str, help='Path to the input FITS event file containing photon data.')
    parser.add_argument('--mask', dest='mask', type=str, default=home+"/userdata/coded_mask/theseus-xgis/instruments/theseus-xgis/instdata/msk_55_4752313.fits", help='Path to the mask FITS file.')
    parser.add_argument('--weight', dest='weight', type=str, default=home+"/userdata/coded_mask/weight/weight_di.fits", help='Path to the weight FITS file.')
    parser.add_argument('--arf', dest='arf', type=str, default=home+"/userdata/coded_mask/theseus-xgis/instruments/theseus-xgis/instdata/XGIS_X_theta_0_phi_0.arf", help='Path to the ARF file.')
    parser.add_argument('--rmf', dest='rmf', type=str, default=home+"/userdata/coded_mask/theseus-xgis/instruments/theseus-xgis/instdata/rmfgrid/xgis_x_theta_0_phi_0_sixte_20251029.rmf", help='Path to the RMF file.')
    parser.add_argument('--no_lc', dest='no_lc', action='store_true', help='Do not generate lightcurve.')
    parser.add_argument('--no_sp', dest='no_sp', action='store_true', help='Do not generate spectrum.')
         
    args = parser.parse_args()

    # Input Validation
    for field, path in [('infile', args.infile), ('mask', args.mask), ('weight', args.weight)]:
        if not path:
            print(f"Error: --{field} is required.", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(path):
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    def recon (mask, img, w):           
        return reconst_new(mask,img,w)  
        return full_reconstruction(mask,img,w) 
            
    dpixel = 5e-3   # pixel size in m
    dmask = 0.63  # mask distance win m
    da = dpixel / dmask * (180/np.pi) # angular pixel size in deg
    # - - - - - - - - - - - - - - - - - - 
    mname = args.mask
    with fits.open(mname) as hdul:
        img_hdu = hdul[0]
        mask = img_hdu.data.astype(float) # Ensure float for calculations
    
    # Resample mask to match pixel size
    mask = ndimage.zoom(mask, 2, order=0)
    mask = np.transpose(mask)
    shape = mask.shape

    # Loading the weighting matrix 
    weight_file = args.weight 
    with fits.open(weight_file) as hdul:
        img_hdu = hdul[0]
        weight = img_hdu.data.astype(float) # Ensure float for calculations

    iname = args.infile
    with fits.open(iname) as tmp:
        # Reading ra dec from input event file
        ra=tmp[1].header['RA_PNT']
        dec=tmp[1].header['DEC_PNT']
        exposure = tmp[1].header['EXPOSURE']
        cols = tmp[1].columns
        data = tmp[1].data

    img = make_img(data)
    sky = recon(mask, img, weight)
    write_img_file(args.outfile + "_sky.fits", np.flipud(sky), ra, dec);

    # Mark a source position 
    W = WCS(args.outfile + "_sky.fits")
    x,y = W.world_to_pixel_values(args.ra_src, args.dec_src)
    x = int(x)
    y = int(y)
    print ("Source position at pixel (x,y) = ", x, y)

    # let loop over time, generate images and reconstruct them 
    nbin = 100
    tstart  = min(data['TIME']) 
    tstop =  max(data['TIME']) 

    if not args.no_sp:
        # Generating a spectrum for each detector channel 
        # (make variable for higher resolution RMFs)
        minPHA = 1
        maxPHA = 57
        e_edges = list(range(minPHA, maxPHA + 2, 1))  # ensure step >=1

        n_bins = max(len(e_edges) - 1, 1)
        spec = np.zeros(n_bins, dtype=float)
        
        print("Extracting spectrum...")
        for i in range(n_bins):
            emin_i = e_edges[i]
            emax_i = e_edges[i + 1]
            im = make_img(data, emin=emin_i, emax=emax_i)
            # ensure no NaNs
            im = np.nan_to_num(im, nan=0.0)
            sky = recon(mask, im, weight)
            # handle NaNs in sky
            sky = np.nan_to_num(sky, nan=0.0)
            x_min = max(0, x - 10)                                                       
            x_max = min(sky.shape[1], x + 10) # Note: x is column (index 1)  
            y_min = max(0, y - 10)                                                       
            y_max = min(sky.shape[0], y + 10) # Note: y is row (index 0)     
            sky = sky[x_min:x_max,y_min:y_max]
            res = np.max(sky) if sky.size > 0 else 0.0
            spec[i] = float(res)

        emids = np.ones(len(e_edges)-1)
        for i in range(len(e_edges)-1):
            emids[i] = .5*( e_edges[i]+e_edges[i+1] )

        # Save as a phafile 

        arffile = args.arf
        rmffile = args.rmf

        write_spectrum_file(args.outfile + "_spec.pha", spec, exposure, arffile, rmffile)

    # Generating a lightcurve
    if not args.no_lc:
        nlc = 100
        emin = 0
        emax = 57
        minT = int(np.min(data['TIME']))
        maxT = int(np.max(data['TIME']))
        # create nlcs time bins using linspace (nlc bins)
        t_edges = np.linspace(minT, maxT, nlc + 1).astype(int)
        # ensure unique and sorted
        t_edges = np.unique(t_edges)

        n_tbins = max(len(t_edges) - 1, 1)
        lc = np.zeros(n_tbins, dtype=float)

        for i in range(n_tbins):
            tmin_i = t_edges[i]
            tmax_i = t_edges[i + 1]
            im = make_img(data, emin=emin, emax=emax, tmin=tmin_i, tmax=tmax_i)
            im = np.nan_to_num(im, nan=0.0)
            sky = reconst_new(mask, im, weight)
            sky = np.nan_to_num(sky, nan=0.0)
            x_min = max(0, x - 10)                                                       
            x_max = min(sky.shape[1], x + 10) # Note: x is column (index 1)  
            y_min = max(0, y - 10)                                                       
            y_max = min(sky.shape[0], y + 10) # Note: y is row (index 0)     
            sky = sky[x_min:x_max,y_min:y_max]
            lc[i] = float(np.max(sky)) if sky.size > 0 else 0.0

        # Generating times for lighcurve bin: 
        tmids = np.ones(len(t_edges)-1)
        for i in range(len(t_edges)-1):
            tmids[i] = .5*( t_edges[i]+t_edges[i+1] )

        # Save as a lccfile 
        write_lc_file(args.outfile + "_lc.fits", tmids, lc, np.sqrt(lc))

if __name__ == "__main__":
    main()