#! /bin/env python3
from astropy.io import fits # Correct import from original
import numpy as np
from numpy.fft import fft2, ifft2, fftshift # For FFT-based correlation
import os
from pimage import plotImage


# import xml.etree.ElementTree as ET # Not used in the provided snippet
verb=0

def make_img(data, emin=0, emax=150, tmin=None, tmax=None, **kwargs):
    # Work with uppercase FITS field names (as in your table)
    pha = data['PHA']
    time = data['TIME']
    det = data['CHIP_ID']
    rawx = data['RAWX']
    rawy = data['RAWY']

    sel = (pha >= emin) & (pha < emax)
    if tmin is not None:
        sel &= (time >= tmin)
    if tmax is not None:
        sel &= (time < tmax)

    # If no events in selection, return zeros array to avoid NaNs
    if not np.any(sel):
        return np.zeros((90, 90), dtype=float)

    x = np.floor(det[sel].astype(float) / 10.0) * 9 + rawx[sel]
    y = (det[sel] % 10) * 9 + rawy[sel]
    bin_x  = np.arange(0, 91, 1)
    bin_y  = np.arange(0, 91, 1)

    H, xedges, yedges = np.histogram2d(x, y, bins=[bin_x, bin_y])
    # H is the 2D histogram counts/density; return it as float array
    return np.array(H, dtype=float)


# --- FFT-based Cross-Correlation Function ---
def CCF2Df_fft(image_matrix, kernel_matrix, desired_output_shape):
    """   
    Performs 2D cross-correlation using FFTs.
    image_matrix: The larger matrix (image) to correlate against.
    kernel_matrix: The smaller matrix (kernel/template) to slide over the image.
    desired_output_shape: The shape of the output correlation map to be extracted.
    """   
#    image_matrix.shape[0], image_matrix.shape[1] = image_matrix.shape
#    kernel_matrix.shape[0], kernel_matrix.shape[1] = kernel_matrix.shape

    # Determine padded dimensions for linear correlation (shape of 'full' correlation)
    fft_calc_rows = image_matrix.shape[0] + kernel_matrix.shape[0] - 1
    fft_calc_cols = image_matrix.shape[1] + kernel_matrix.shape[1] - 1
    
    # For optimal FFT performance, one might pad to the next fast FFT length.wieder
    fft_processing_shape = (fft_calc_rows, fft_calc_cols)
    # desired_output_shape=fft_processing_shape # Full shape Careful
    if verb > 1:
        print(f"CCF2Df: image_shape={image_matrix.shape}, kernel_shape={kernel_matrix.shape}, fft_processing_shape={fft_processing_shape}, desired_output_shape={desired_output_shape}")

   
    print( image_matrix.shape, kernel_matrix.shape, fft_processing_shape)
    if image_matrix.shape[0] < kernel_matrix.shape[0]: 
        image_matrix = np.pad(image_matrix, (( kernel_matrix.shape[0] - image_matrix.shape[0])//2, 
                                             ( kernel_matrix.shape[1] - image_matrix.shape[1])//2),
                                               mode='constant', constant_values=0)
    if image_matrix.shape[0] > kernel_matrix.shape[0]: 
        kernel_matrix = np.pad(kernel_matrix, (( image_matrix.shape[0] - kernel_matrix.shape[0])//2,
                                                (image_matrix.shape[1] - kernel_matrix.shape[1])//2),
                                                mode='constant', constant_values=0)

    print("Image:", image_matrix.shape, "Mask:", kernel_matrix.shape, fft_processing_shape)

    ft_image =  fft2(image_matrix, s=fft_processing_shape)
    ft_kernel = fft2(kernel_matrix, s=fft_processing_shape)

    product_freq_domain = ft_image * np.conj(ft_kernel)

    correlation_map_full = ifft2(product_freq_domain)

    correlation_map_full_real = np.real(correlation_map_full)  
   # out_r, out_c = desired_output_shape
    ccf_map_final = fftshift(correlation_map_full_real )
    
    if verb > 0:
        p_idx = np.unravel_index(np.argmax(ccf_map_final), ccf_map_final.shape)
        print(f"  CCF2Df: Peak at {p_idx} val {ccf_map_final[p_idx]} shape {ccf_map_final.shape}")

    # Crop to image_matrix shape (centered) to match the expected coordinate frame of full_reconstruction
    # This ensures that zero lag corresponds to the center of the output array
    out_r, out_c = image_matrix.shape
    curr_r, curr_c = ccf_map_final.shape
    
    start_r = (curr_r - out_r) // 2
    start_c = (curr_c - out_c) // 2
    
    return ccf_map_final[start_r : start_r + out_r, start_c : start_c + out_c]


def reconst_new(mask, det, w):
 # mask needs to contain only 0 or 1 
    
    gp = np.zeros_like(mask, dtype=float)
    gp[mask > 0.5] = 1.0
    
    gm = np.zeros_like(mask, dtype=float)
    gm[mask <= 0.5] = -1.0
    
    # Determine common FFT size
    fft_rows = det.shape[0] + mask.shape[0] - 1
    fft_cols = det.shape[1] + mask.shape[1] - 1
    fft_shape = (fft_rows, fft_cols)

    # Pad all inputs to fft_shape
    # det_padded = np.pad(det, ...)
    
    matg =  mask.shape[0] +det.shape[0] # #  fft_shape[0] # weights.shape[0]
        
    detpad = np.pad(det,(( matg -  det.shape[0])//2, (mask.shape[1] - det.shape[1])//2), mode='constant', constant_values=0)
    wpad =   np.pad(w,  (( matg -    w.shape[0])//2, (mask.shape[1] - w.shape[1])//2), mode='constant', constant_values=0)
    gp =     np.pad(gp, (( matg -   gp.shape[0])//2, (mask.shape[1] - gp.shape[1])//2),  mode='constant', constant_values=0)
    gm =     np.pad(gm, (( matg -   gm.shape[0])//2, (mask.shape[1] - gm.shape[1])// 2), mode='constant', constant_values=0)  

    detw = detpad*wpad
    
    fftgp =   fft2(gp)
    fftgm =   fft2(gm)
    fftw =    fft2(wpad)
    fftdetw = fft2(detw)
    
    # from writeData import write_img_file 
    # write_img_file("gm.fits", mask, 0,0)

    norm = 1 # detw.shape[0]*detw.shape[1]  #hmm... 
    gpw =    fftshift(norm * np.real(fft2(fftgp   * np.conj(fftw))))
    gmw =    fftshift(norm * np.real(fft2(fftgm   * np.conj(fftw))))
    detwgp = fftshift(norm * np.real(ifft2(fftdetw * np.conj(fftgp)))) # actual signal using  ifft
    detwgm = fftshift(norm * np.real(ifft2(fftdetw * np.conj(fftgm))))

    b = (gpw-1)/(gmw+1e-6) # will avoid devision by zero 
    sky  = detwgp  - b * detwgm
    
    res = {"sky":sky, "gpw":gpw, "gmw":gmw, "detwgp":detwgp, "detwgm":detwgm, "b":b , "fftgp":fftgp, "fftgm":fftgm, "fftw":fftw, "fftdetw":fftdetw}
    return np.flipud(sky)  # Make sure image is oriented correctly 

def full_reconstruction(mask, img, weights):
    from scipy import ndimage

    det_rows = img.shape[0]
    det_cols = img.shape[1]

    # Original mask dimensions
    mask_rows_orig = mask.shape[0]
    mask_cols_orig = mask.shape[1]

    if verb > 0:
        print(f"Detector dimensions: {det_rows} (rows) x {det_cols} (cols)")
        print(f"Original mask dimensions: {mask_rows_orig} (rows) x {mask_cols_orig} (cols)")

    # Dimensions of the padded mask G
    img_size = mask_rows_orig + 2 * det_rows

    # Padding of the weights matrix:
    w_pad = (img_size - weights.shape[0])//2
    W = np.pad(weights, w_pad, mode='constant', constant_values=0)

    img_pad = (img_size - img.shape[0])//2
    img = np.pad(img, img_pad, mode='constant', constant_values=0)

    G_pad = (img_size - mask.shape[0])//2
    
    # Calculate Gplus/Gminus from original mask to ensure padding is 0
    Gplus_orig  = np.where(mask >  0,  1.0, 0.0)
    Gminus_orig = np.where(mask <= 0, -1.0, 0.0)
    
    mask = np.pad(mask, G_pad, mode='constant', constant_values=0)
    Gplus = np.pad(Gplus_orig, G_pad, mode='constant', constant_values=0)
    Gminus = np.pad(Gminus_orig, G_pad, mode='constant', constant_values=0)

    # Convert mask values from 0 (closed) / 1 (open) to -1 (closed) / +1 (open)
    G = mask

    # --- Derived Masks (Vectorized) ---
    # Gplus and Gminus already calculated and padded with 0

    Gschmidt = np.ones_like(G)
    N_total_G = G.size
    Nt_open_original_mask = np.sum(mask) # Sum of 0/1 values in original mask

    if Nt_open_original_mask == 0:
        if verb >=0: print("Warning: Original mask has no open elements (Nt_open_original_mask is zero).")
        val_open = np.inf # Avoid division by zero
    else:
        val_open = 1.0 * N_total_G / Nt_open_original_mask

    if N_total_G == Nt_open_original_mask:
        if verb >=0: print("Warning: Padded mask effectively has no closed elements for Gschmidt denominator.")
        val_closed = -np.inf # Avoid division by zero
    else:
        val_closed = -1.0 * N_total_G / (N_total_G - Nt_open_original_mask)

    Gschmidt[G > 0] = val_open
    Gschmidt[G <= 0] = val_closed

    # These define the output map size for correlation results
    sky_map_rows = mask_rows_orig + det_rows
    sky_map_cols = mask_cols_orig + det_cols
    sky_shape = (sky_map_rows, sky_map_cols)

    # Initialize sky-related arrays
    #skyp = np.zeros(sky_shape, dtype=float)
    #skyn = np.zeros(sky_shape, dtype=float)
    sky  = np.zeros(sky_shape, dtype=float)

    # Weighting matrix (e.g., for dead pixels)
    # Other arrays, initialized to the sky map shape
    A_map  = np.zeros_like(sky)
    V_map  = np.zeros_like(sky)
    Bal_map = np.zeros_like(sky)

    # W = 1 - W
    img_weighted = img * W 

    # Calculate Bal_map
    if verb >= 0: print("Calculating Bal_map...")
    ccf_W_Gplus =  CCF2Df_fft(Gplus,  W, sky_shape)
    ccf_W_Gminus = CCF2Df_fft(Gminus, W, sky_shape)

    # Handle potential division by zero for Bal_map
    denominator_bal = ccf_W_Gminus
    threshold_bal = 1e-6 * np.max(np.abs(denominator_bal)) if np.max(np.abs(denominator_bal)) > 0 else 1e-9
    Bal_map = np.divide(ccf_W_Gplus, denominator_bal,
                        out=np.zeros_like(ccf_W_Gplus),
                        where=np.abs(denominator_bal) > threshold_bal)
    
    if verb >=0 and np.any(np.abs(denominator_bal) <= threshold_bal):
        print("Warning: Division by zero (or small value) encountered in Bal_map calculation.")

    if verb >= 0: print("Calculating A_map...")  #  A = (G+ · M) * W − Bal · ((G− · M) * W )

    term_A1_mask = mask * Gplus
    ccf_W_termA1 = CCF2Df_fft(term_A1_mask, W, sky_shape)
    term_A2_mask = mask * Gminus
    ccf_W_termA2 = CCF2Df_fft(term_A2_mask, W, sky_shape)

    tmp = term_A1_mask
    A_map = ccf_W_termA1 - Bal_map * ccf_W_termA2

    # Calculate skyp and skyn components
    if verb >= 0: print("Calculating skyp and skyn components...")
    skyp_component = CCF2Df_fft(Gplus, img_weighted, sky_shape)
    skyn_component = Bal_map * CCF2Df_fft(Gminus, img_weighted, sky_shape)

    # Calculate final sky image
    if verb >= 0: print("Calculating final sky image...")
    denominator_sky = A_map
    
    # Avoid division by very small numbers (noise amplification at edges)
    threshold = 1e-6 * np.max(np.abs(denominator_sky)) if np.max(np.abs(denominator_sky)) > 0 else 1e-9
    
    sky = np.divide((skyp_component - skyn_component), denominator_sky,
                    out=np.zeros_like(skyp_component),
                    where=np.abs(denominator_sky) > threshold)
                    
    if verb >=0 and np.any(np.abs(denominator_sky) <= threshold):
        print("Warning: Division by zero (or small value) encountered in final sky image calculation.")

    # Calculate Variance V_map
    if verb >= 0: print("Calculating Variance (V_map)...")
    Gplus_squared = np.square(Gplus)
    Gminus_squared = np.square(Gminus)

    ccf_imgW_GplusSq = CCF2Df_fft(Gplus_squared, img_weighted, sky_shape)
    ccf_imgW_GminusSq = CCF2Df_fft(Gminus_squared, img_weighted, sky_shape)

    V_map = ccf_imgW_GplusSq + np.square(Bal_map) * ccf_imgW_GminusSq
    # Normalize Variance
    V_map = np.divide(V_map, np.square(denominator_sky),
                      out=np.zeros_like(V_map),
                      where=np.abs(denominator_sky) > threshold)

    # Crop to sky_shape
    r_start = (img_size - sky_map_rows) // 2
    c_start = (img_size - sky_map_cols) // 2
    sky = sky[r_start:r_start+sky_map_rows, c_start:c_start+sky_map_cols]
    V_map = V_map[r_start:r_start+sky_map_rows, c_start:c_start+sky_map_cols]

    print("Shape of sky image here:", sky.shape) 
    # --- Sky Grid Generation & Plotting (Placeholders/Assumptions) ---

    result = {"sky" : sky , 
                 "G" : G ,
                 "W" : W,
                 "A_map" : A_map,
                 "V_map" : V_map,
                 "Bal_map" : Bal_map,
                 "skyp" : skyp_component,
                 "skyn" : skyn_component,
                 "denominator" : denominator_sky,
                 "img" : img,
                 "img_w" : img_weighted,
                 "Gplus" : Gplus,
                 "Gminus" : Gminus,
                 "tmp" : tmp
   }

    return result["sky"]


def plotting(): 
        import matplotlib.pyplot as plt
        
        # Placeholder values if 'igrid' and 'par.mdist' are not available:
        pixel_size_x = 0.005
        pixel_size_y = 0.005
        mask_distance = 0.631

        num_sky_pixels_x = sky.shape[1]
        num_sky_pixels_y = sky.shape[0]
        if verb >= 0: print(f"Number of sky pixels: X={num_sky_pixels_x}, Y={num_sky_pixels_y}")

        angular_pixel_x_rad = np.arctan(pixel_size_x / mask_distance)
        angular_pixel_y_rad = np.arctan(pixel_size_y / mask_distance)
        print(f"Pixel size: X={pixel_size_x}, Y={pixel_size_y} [m]")
        print(f"Angular pixel size in radians: X={angular_pixel_x_rad}, Y={angular_pixel_y_rad}")    
        half_fov_x_rad = (num_sky_pixels_x / 2.0) * angular_pixel_x_rad
        half_fov_y_rad = (num_sky_pixels_y / 2.0) * angular_pixel_y_rad

        print(f"Half FOV in radians: X={half_fov_x_rad}, Y={half_fov_y_rad}")
        print(f"Half FOV in degrees: X={half_fov_x_rad * (180.0 / np.pi)}, Y={half_fov_y_rad * (180.0 / np.pi)}")
        print(f"Angular pixel size in radians: X={angular_pixel_x_rad}, Y={angular_pixel_y_rad}")
        
        x_sky_grid_deg = np.linspace(-half_fov_x_rad, half_fov_x_rad, num_sky_pixels_x) * (180.0 / np.pi)
        y_sky_grid_deg = np.linspace(-half_fov_y_rad, half_fov_y_rad, num_sky_pixels_y) * (180.0 / np.pi)
        print("Number of pixerls: X=", num_sky_pixels_x, "Y=", num_sky_pixels_y)
     
        # = CCF2Df_fft(img, mask, sky_shape)  # Assuming this is the final sky image
        if verb >= 0: print("Plotting sky image...")
        plt.figure(figsize=(10, 8))
        plt.imshow( sky, origin='lower', cmap='viridis', aspect='auto', extent=[x_sky_grid_deg.min(), x_sky_grid_deg.max(), y_sky_grid_deg.min(), y_sky_grid_deg.max()])
        plt.colorbar(label="Sky Intensity")
        plt.xlabel("Sky X [deg]")
        plt.ylabel("Sky Y [deg]")
        plt.title("Reconstructed Sky Image")
        plt.savefig('sky_image.pdf')
        plt.close()
        print("Done plotting")

def writing():  
        hdu = fits.PrimaryHDU(data=sky) 
        hdul = fits.HDUList([hdu])
        os.remove('sky_image.fits') if os.path.exists('sky_image.fits') else None
        hdul.writeto('sky_image.fits')   