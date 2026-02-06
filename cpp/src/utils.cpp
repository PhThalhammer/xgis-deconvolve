#include "utils.h"
#include <cmath>
#include <iostream>
#include <cstring>

Matrix2D resample_mask(const Matrix2D& input, int nsample) {
    int factor = 2 * nsample;
    size_t new_rows = input.rows * factor;
    size_t new_cols = input.cols * factor;
    Matrix2D output(new_rows, new_cols);

    for (size_t r = 0; r < new_rows; ++r) {
        for (size_t c = 0; c < new_cols; ++c) {
            size_t src_r = r / factor;
            size_t src_c = c / factor;
            if (src_r < input.rows && src_c < input.cols) {
                output(r, c) = input(src_r, src_c);
            }
        }
    }
    return output;
}

// Standard WCS TAN projection (gnomonic)
void world_to_pixel(double ra, double dec, double crval1, double crval2, double cdelt1, double cdelt2, double crpix1, double crpix2, double& x, double& y) {
    double deg2rad = PI / 180.0;
    double ra_r = ra * deg2rad;
    double dec_r = dec * deg2rad;
    double crval1_r = crval1 * deg2rad;
    double crval2_r = crval2 * deg2rad;

    // Relative RA
    double d_ra = ra_r - crval1_r;
    
    // Standard TAN projection formulas
    double denom = std::sin(dec_r) * std::sin(crval2_r) + std::cos(dec_r) * std::cos(crval2_r) * std::cos(d_ra);
    
    // Avoid division by zero
    if (std::abs(denom) < 1e-9) {
        x = 0; y = 0; return; 
    }

    double xi = (std::cos(dec_r) * std::sin(d_ra)) / denom;
    double eta = (std::sin(dec_r) * std::cos(crval2_r) - std::cos(dec_r) * std::sin(crval2_r) * std::cos(d_ra)) / denom;

    // Convert to degrees
    xi /= deg2rad;
    eta /= deg2rad;

    // Convert to pixel coordinates
    // Note: FITS uses 1-based indexing for CRPIX usually, but we work in 0-based for internal logic.
    // Ideally we match what Python's astropy WCS does.
    // x = (xi / cdelt1) + (crpix1 - 1); // If crpix is 1-based
    // y = (eta / cdelt2) + (crpix2 - 1);
    
    // Assuming the CRPIX passed in is 0-based from our C++ logic or we adjust it. 
    // The python code calculates CRPIX as shape[1]/2 which is 0-based friendly (center of array).
    // Let's assume CRPIX is the pixel coordinate of the reference point.
    x = (xi / cdelt1) + crpix1;
    y = (eta / cdelt2) + crpix2;
}

Matrix2D pad_array(const Matrix2D& input, size_t new_rows, size_t new_cols) {
    Matrix2D output(new_rows, new_cols, 0.0);
    size_t r_offset = (new_rows - input.rows) / 2;
    size_t c_offset = (new_cols - input.cols) / 2;

    for (size_t r = 0; r < input.rows; ++r) {
        for (size_t c = 0; c < input.cols; ++c) {
            output(r + r_offset, c + c_offset) = input(r, c);
        }
    }
    return output;
}

Matrix2D crop_center(const Matrix2D& input, size_t target_rows, size_t target_cols) {
    Matrix2D output(target_rows, target_cols);
    size_t r_start = (input.rows - target_rows) / 2;
    size_t c_start = (input.cols - target_cols) / 2;

    for (size_t r = 0; r < target_rows; ++r) {
        for (size_t c = 0; c < target_cols; ++c) {
            output(r, c) = input(r + r_start, c + c_start);
        }
    }
    return output;
}

void flipud(Matrix2D& mat) {
    size_t rows = mat.rows;
    size_t cols = mat.cols;
    for (size_t r = 0; r < rows / 2; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            std::swap(mat(r, c), mat(rows - 1 - r, c));
        }
    }
}

void fliplr(Matrix2D& mat) {
    size_t rows = mat.rows;
    size_t cols = mat.cols;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols / 2; ++c) {
            std::swap(mat(r, c), mat(r, cols - 1 - c));
        }
    }
}

ComplexMatrix fft_2d(const Matrix2D& input) {
    size_t rows = input.rows;
    size_t cols = input.cols;
    ComplexMatrix output(rows, cols);
    
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    
    // Copy real input to complex buffer
    for (size_t i = 0; i < rows * cols; ++i) {
        in[i][0] = input.data[i];
        in[i][1] = 0.0;
    }
    
    fftw_plan p = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    
    for (size_t i = 0; i < rows * cols; ++i) {
        output.data[i] = std::complex<double>(out[i][0], out[i][1]);
    }
    
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    
    return output;
}

Matrix2D ifft_2d(const ComplexMatrix& input) {
    size_t rows = input.rows;
    size_t cols = input.cols;
    Matrix2D output(rows, cols);
    
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    
    for (size_t i = 0; i < rows * cols; ++i) {
        in[i][0] = input.data[i].real();
        in[i][1] = input.data[i].imag();
    }
    
    fftw_plan p = fftw_plan_dft_2d(rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    
    double norm = 1.0 / (rows * cols); // FFTW does not normalize
    
    for (size_t i = 0; i < rows * cols; ++i) {
        output.data[i] = out[i][0] * norm; // Take real part
    }
    
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    
    return output;
}

Matrix2D ifft_shift(const Matrix2D& input) {
    // Swap quadrants
    size_t rows = input.rows;
    size_t cols = input.cols;
    Matrix2D output(rows, cols);
    
    size_t r_mid = rows / 2;
    size_t c_mid = cols / 2;
    
    // Top-Left -> Bottom-Right
    // Top-Right -> Bottom-Left
    // Bottom-Left -> Top-Right
    // Bottom-Right -> Top-Left
    
    // Logic: new_r = (r + mid) % rows
    
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            size_t new_r = (r + r_mid) % rows;
            size_t new_c = (c + c_mid) % cols;
            output(new_r, new_c) = input(r, c);
        }
    }
    return output;
}

// Stub for ccf_fft_2d if needed, or remove from header if unused.
Matrix2D ccf_fft_2d(const Matrix2D& img, const Matrix2D& kernel) {
    return Matrix2D(0,0);
}
