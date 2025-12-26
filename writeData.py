from astropy.io import fits 
import numpy as np


def write_spectrum_file(filename, spec, exposure, arffile, rmffile):
    # Writing phafile with spec and exp,arf,rmf header
    header = {
        'EXPOSURE': exposure,
        'ANCRFILE': arffile,
        'RESPFILE': rmffile
    }

    col_channel = fits.Column(name='CHANNEL',
                              format='J',         
                              array=np.arange(len(spec)) )

    col_counts = fits.Column(name='COUNTS',
                             format='J',
                             array=spec.astype(int),
                             unit='count')      
    cols = fits.ColDefs([col_channel, col_counts])
    bintable_hdu = fits.BinTableHDU.from_columns(cols)

    bintable_hdu.header['EXTNAME'] = ('SPECTRUM', 'Name of this binary table extension')
    bintable_hdu.header['TELESCOP'] = ('THESEUS', 'Mission name')
    bintable_hdu.header['INSTRUME'] = ('XGIS', 'Instrument name')
    for key, value in header.items():
        bintable_hdu.header[key] = value

    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu, bintable_hdu])

    hdul.writeto(filename, overwrite=True)
    hdul.close()
    return 1

def write_lc_file(filename, times, rates, errors):
    # Writing lcfile with times, rates, errors
    header = {
        'TELESCOP': 'THESEUS',
        'INSTRUME': 'XGIS',
        'TIMEUNIT': 's',
        'RATEUNIT': 'counts/s'
    }
    errors = np.sqrt(rates)  # simple Poisson errors

    col_time = fits.Column(name='TIME',
                           format='D',         
                           array=times )

    col_rate = fits.Column(name='RATE',
                           format='E',
                           array=rates,
                           unit='counts/s')      

    col_error = fits.Column(name='ERROR',
                            format='E',
                            array=errors,
                            unit='counts/s')      

    cols = fits.ColDefs([col_time, col_rate, col_error])
    bintable_hdu = fits.BinTableHDU.from_columns(cols)  
    for key, value in header.items():
        bintable_hdu.header[key] = value        
    hdul = fits.HDUList([fits.PrimaryHDU(), bintable_hdu])
    hdul.writeto(filename, overwrite=True)
    hdul.close()
    return 1

def write_img_file(filename, data, ra, dec):
    # Writing image file with WCS header
    da = 50.0  # total angular size in degrees
    # angular size of one pixel and 
    dp = (0.5 / 63.0) * (180 / np.pi)  # pixel scale in degrees;
    data = np.fliplr(data)
    header = {'CTYPE1': 'RA---TAN',
           'CTYPE2': 'DEC--TAN',
           'CUNIT1': 'deg',
           'CUNIT2': 'deg',
           'CRPIX1': data.shape[1] / 2,
           'CRPIX2': data.shape[0] / 2,
           'CRVAL1': ra,  # Reference RA value 
           'CRVAL2': dec,  # Reference DEC value
           'CDELT1': dp,  # Pixel scale in RA
           'CDELT2': dp   # Pixel scale in DEC
           }   
    hdu = fits.PrimaryHDU(data)
    for key, value in header.items():
        hdu.header[key] = value        
    hdu.writeto(filename, overwrite=True)
    return 1
        