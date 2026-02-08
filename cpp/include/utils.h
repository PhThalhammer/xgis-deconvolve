#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <complex>
#include <fftw3.h>

// Constants
constexpr double PI = 3.14159265358979323846;

// Simple 2D Matrix wrapper
class Matrix2D {
public:
    size_t rows;
    size_t cols;
    std::vector<double> data;

    Matrix2D() : rows(0), cols(0) {}
    Matrix2D(size_t r, size_t c, double val = 0.0) : rows(r), cols(c), data(r * c, val) {}

    double& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }

    const double& operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    void fill(double val) {
        std::fill(data.begin(), data.end(), val);
    }
    
    // Returns flattened data pointer
    double* ptr() { return data.data(); }
    const double* ptr() const { return data.data(); }
};

// Complex Matrix wrapper for FFT
class ComplexMatrix {
public:
    size_t rows;
    size_t cols;
    std::vector<std::complex<double>> data;

    ComplexMatrix() : rows(0), cols(0) {}
    ComplexMatrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {}

    std::complex<double>& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }
    
    const std::complex<double>& operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }
};

// FFT Helpers
ComplexMatrix fft_2d(const Matrix2D& input);
Matrix2D ifft_2d(const ComplexMatrix& input); // Returns real part, shifted? Python's ifft2 returns complex, we usually want real.
Matrix2D ifft_shift(const Matrix2D& input);

// Simple struct for Event data
struct EventData {
    std::vector<double> time;
    std::vector<int> pha;
    std::vector<int> det_id;
    std::vector<int> rawx;
    std::vector<int> rawy;
    double ra_pnt;
    double dec_pnt;
    double exposure;
};

// Utils functions
Matrix2D resample_mask(const Matrix2D& input, int nsample);
void world_to_pixel(double ra, double dec, double crval1, double crval2, double cdelt1, double cdelt2, double crpix1, double crpix2, double& x, double& y);

// FFT Helper
// Performs C = A * conj(B) in freq domain, then IFFT
// If pad_to_shape is provided, pads inputs to that size.
Matrix2D ccf_fft_2d(const Matrix2D& img, const Matrix2D& kernel);

// Helpers for array manipulation
Matrix2D pad_array(const Matrix2D& input, size_t new_rows, size_t new_cols);
Matrix2D pad_asym(const Matrix2D& input, size_t pad_before, size_t pad_after);
Matrix2D crop_center(const Matrix2D& input, size_t target_rows, size_t target_cols);
void flipud(Matrix2D& mat);
void fliplr(Matrix2D& mat);

#endif // UTILS_H
