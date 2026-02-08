#include "iros.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// Replicates numpy.histogram2d for the specific case in iros.py
Matrix2D make_img(const EventData& data, double emin, double emax, double tmin, double tmax) {
    // 90x90 grid
    Matrix2D img(90, 90, 0.0);
    
    size_t n = data.pha.size();
    for (size_t i = 0; i < n; ++i) {
        // Filter
        if (data.pha[i] < emin || data.pha[i] >= emax) continue;
        if (tmin != -1.0 && data.time[i] < tmin) continue;
        if (tmax != -1.0 && data.time[i] >= tmax) continue; // Python uses < tmax, but logic might vary. Using >= for exclusion to match Python's 'sel &= (time < tmax)'
        // Python: (time >= tmin) & (time < tmax)
        
        double det_val = (double)data.det_id[i];
        double rawx_val = (double)data.rawx[i];
        double rawy_val = (double)data.rawy[i];
        
        // x = np.floor(det / 10.0) * 9 + rawx
        // y = (det % 10) * 9 + rawy
        int x_idx = (int)(std::floor(det_val / 10.0) * 9.0 + rawx_val);
        int y_idx = (int)((int)det_val % 10 * 9 + rawy_val);
        
        // Check bounds (0 to 90)
        // Python bins are 0..91 with step 1. So indices 0..89.
        if (x_idx >= 0 && x_idx < 90 && y_idx >= 0 && y_idx < 90) {
            // Histogram count
            // Python histogram2d returns counts.
            img(x_idx, y_idx) += 1.0; 
            // Note: Python histogram2d(x, y). output[i, j] is sum of samples where x in bin i and y in bin j.
            // So img(x_idx, y_idx) is correct.
        }
    }
    return img;
}

Matrix2D reconst_new(const Matrix2D& mask, const Matrix2D& det, const Matrix2D& w) {
    // gp = mask > 0.5 ? 1.0 : 0.0
    Matrix2D gp(mask.rows, mask.cols, 0.0);
    Matrix2D gm(mask.rows, mask.cols, 0.0);
    
    for (size_t i = 0; i < mask.rows * mask.cols; ++i) {
        if (mask.data[i] > 0.5) gp.data[i] = 1.0;
        else gm.data[i] = -1.0; // Python: gm[mask <= 0.5] = -1.0
    }
    
    // Padding logic
    // matg = mask.shape[0] + det.shape[0]
    size_t matg = mask.rows + det.rows;
    // axis 0 target size: matg
    // axis 1 target size: mask.cols
    
    // Pad det
    size_t p1_det = (matg - det.rows) / 2;
    size_t p2_det = (mask.cols - det.cols) / 2;
    Matrix2D detpad = pad_asym(det, p1_det, p2_det);
    // Pad w
    Matrix2D wpad = pad_asym(w, p1_det, p2_det);
    // Pad gp
    size_t p1_mask = (matg - mask.rows) / 2;
    size_t p2_mask = (mask.cols - mask.cols) / 2;
    Matrix2D gppad = pad_asym(gp, p1_mask, p2_mask);
    // Pad gm
    Matrix2D gmpad = pad_asym(gm, p1_mask, p2_mask);
    
    // detw = detpad * wpad
    Matrix2D detw(detpad.rows, detpad.cols);
    for (size_t i = 0; i < detw.data.size(); ++i) {
        detw.data[i] = detpad.data[i] * wpad.data[i];
    }
    
    // FFTs
    ComplexMatrix fftgp = fft_2d(gppad);
    ComplexMatrix fftgm = fft_2d(gmpad);
    ComplexMatrix fftw = fft_2d(wpad);
    ComplexMatrix fftdetw = fft_2d(detw);
    
    double norm = 1.0;
    
    // gpw = fftshift(norm * real(fft2(fftgp * conj(fftw))))
    // Calculate product fftgp * conj(fftw)
    ComplexMatrix prod_gpw(fftgp.rows, fftgp.cols);
    for (size_t i = 0; i < prod_gpw.data.size(); ++i) {
        prod_gpw.data[i] = fftgp.data[i] * std::conj(fftw.data[i]);
    }
    // "fft2" (forward) of product
    // Note: Python's fft2 is forward. We implemented fft_2d (forward).
    // But fft_2d takes Matrix2D (Real). We need fft_2d for Complex.
    
    auto fft_complex_forward = [](const ComplexMatrix& in) -> ComplexMatrix {
        ComplexMatrix out(in.rows, in.cols);
        fftw_complex* fin = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * in.rows * in.cols);
        fftw_complex* fout = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * in.rows * in.cols);
        for(size_t i=0; i<in.data.size(); ++i) {
            fin[i][0] = in.data[i].real();
            fin[i][1] = in.data[i].imag();
        }
        fftw_plan p = fftw_plan_dft_2d(in.rows, in.cols, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        for(size_t i=0; i<out.data.size(); ++i) {
            out.data[i] = std::complex<double>(fout[i][0], fout[i][1]);
        }
        fftw_destroy_plan(p);
        fftw_free(fin);
        fftw_free(fout);
        return out;
    };
    
    ComplexMatrix gpw_c = fft_complex_forward(prod_gpw);
    
    // gmw = ... fft2(fftgm * conj(fftw))
    ComplexMatrix prod_gmw(fftgm.rows, fftgm.cols);
    for (size_t i = 0; i < prod_gmw.data.size(); ++i) {
        prod_gmw.data[i] = fftgm.data[i] * std::conj(fftw.data[i]);
    }
    ComplexMatrix gmw_c = fft_complex_forward(prod_gmw);

    // detwgp = ... ifft2(fftdetw * conj(fftgp))
    ComplexMatrix prod_detwgp(fftdetw.rows, fftdetw.cols);
    for (size_t i = 0; i < prod_detwgp.data.size(); ++i) {
        prod_detwgp.data[i] = fftdetw.data[i] * std::conj(fftgp.data[i]);
    }
    // ifft2
    Matrix2D detwgp = ifft_2d(prod_detwgp); // Returns real part
    
    // detwgm = ... ifft2(fftdetw * conj(fftgm))
    ComplexMatrix prod_detwgm(fftdetw.rows, fftdetw.cols);
    for (size_t i = 0; i < prod_detwgm.data.size(); ++i) {
        prod_detwgm.data[i] = fftdetw.data[i] * std::conj(fftgm.data[i]);
    }
    Matrix2D detwgm = ifft_2d(prod_detwgm);

    // Apply Real and Shift
    // gpw (real part)
    Matrix2D gpw(gpw_c.rows, gpw_c.cols);
    for(size_t i=0; i<gpw.data.size(); ++i) gpw.data[i] = gpw_c.data[i].real();
    gpw = ifft_shift(gpw);

    Matrix2D gmw(gmw_c.rows, gmw_c.cols);
    for(size_t i=0; i<gmw.data.size(); ++i) gmw.data[i] = gmw_c.data[i].real();
    gmw = ifft_shift(gmw);
    
    detwgp = ifft_shift(detwgp);
    detwgm = ifft_shift(detwgm);
    
    // b = (gpw - 1) / (gmw + 1e-6)
    Matrix2D b(gpw.rows, gpw.cols);
    for (size_t i = 0; i < b.data.size(); ++i) {
        b.data[i] = (gpw.data[i] - 1.0) / (gmw.data[i] + 1e-6);
    }
    
    // sky = detwgp - b * detwgm
    Matrix2D sky(detwgp.rows, detwgp.cols);
    for (size_t i = 0; i < sky.data.size(); ++i) {
        sky.data[i] = detwgp.data[i] - b.data[i] * detwgm.data[i];
    }
    
    // return np.flipud(sky)
    flipud(sky);
    return sky;
}
