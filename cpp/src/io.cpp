#include "io.h"
#include <CCfits/CCfits>
#include <iostream>
#include <cstdio>
#include <cmath>

using namespace CCfits;

Matrix2D read_fits_image(const std::string& filename) {
    try {
        FITS fits(filename, Read, true);
        PHDU& phdu = fits.pHDU();
        
        std::valarray<double> contents;
        phdu.read(contents);
        
        long rows = phdu.axis(1);
        long cols = phdu.axis(0);
        
        Matrix2D img(rows, cols);
        // CCfits reads row by row?
        // FITS is Fortran order (column-major) usually?
        // CCfits documentation: "The array is filled with the image data. The first element of the array is the value of the first pixel in the image."
        // (1,1), (2,1), ... (cols, 1), (1, 2)...
        // So it fills row 0 (all cols), then row 1...
        // Matches our Matrix2D (row-major).
        
        for (size_t i = 0; i < contents.size(); ++i) {
            img.data[i] = contents[i];
        }
        return img;
    } catch (FITS::CantOpen& e) {
        std::cerr << "Error reading FITS file: " << filename << " " << e.message() << std::endl;
        exit(1);
    } catch (FitsException& e) {
        std::cerr << "Fits Exception: " << e.message() << std::endl;
        exit(1);
    }
}

EventData read_event_file(const std::string& filename) {
    EventData ev;
    try {
        FITS fits(filename, Read, true);
        ExtHDU& table = fits.extension(1);
        
        // Read header
        try {
            table.readKey("RA_PNT", ev.ra_pnt);
            table.readKey("DEC_PNT", ev.dec_pnt);
            table.readKey("EXPOSURE", ev.exposure);
        } catch (...) {
            std::cerr << "Warning: Missing header keys in " << filename << std::endl;
            ev.ra_pnt = 0; ev.dec_pnt = 0; ev.exposure = 1.0;
        }
        
        long rows = table.rows();
        
        // Read columns
        // Assuming column names: TIME, PHA, CHIP_ID, RAWX, RAWY
        // Note: Python script uses uppercase names.
        
        std::valarray<double> time_va;
        std::valarray<int> pha_va; // Python PHA is int? usually.
        std::valarray<int> chip_va;
        std::valarray<int> rawx_va;
        std::valarray<int> rawy_va;
        
        // We need to support different types. PHA might be integer or byte.
        // using read column.
        
        table.column("TIME").read(time_va, 0, rows);
        
        // Check types for PHA. Python code 'data['PHA']'.
        // If it's a TINT/TSHORT/TBYTE.
        table.column("PHA").read(pha_va, 0, rows);
        
        table.column("CHIP_ID").read(chip_va, 0, rows);
        table.column("RAWX").read(rawx_va, 0, rows);
        table.column("RAWY").read(rawy_va, 0, rows);
        
        // Copy to vector
        ev.time.assign(std::begin(time_va), std::end(time_va));
        ev.pha.assign(std::begin(pha_va), std::end(pha_va));
        ev.det_id.assign(std::begin(chip_va), std::end(chip_va));
        ev.rawx.assign(std::begin(rawx_va), std::end(rawx_va));
        ev.rawy.assign(std::begin(rawy_va), std::end(rawy_va));
        
    } catch (FitsException& e) {
        std::cerr << "Fits Exception reading events: " << e.message() << std::endl;
        exit(1);
    }
    return ev;
}

void write_img_file(const std::string& filename, const Matrix2D& data, double ra, double dec) {
    try {
        std::string filename_clean = filename;
        // Check if exists and remove
        std::remove(filename.c_str());
        
        // Prepare data: fliplr
        Matrix2D out_data = data;
        fliplr(out_data);
        
        long naxis = 2;
        long naxes[2] = { (long)out_data.cols, (long)out_data.rows };
        
        // Create FITS
        FITS fits(filename, DOUBLE_IMG, naxis, naxes);
        PHDU& phdu = fits.pHDU();
        
        std::valarray<double> array(out_data.data.size());
        for(size_t i=0; i<out_data.data.size(); ++i) array[i] = out_data.data[i];
        
        phdu.write(1, phdu.axis(0)*phdu.axis(1), array); // Write flat
        
        // Header
        double dp = (0.5 / 63.0) * (180.0 / PI);
        
        phdu.addKey("CTYPE1", "RA---TAN", "");
        phdu.addKey("CTYPE2", "DEC--TAN", "");
        phdu.addKey("CUNIT1", "deg", "");
        phdu.addKey("CUNIT2", "deg", "");
        phdu.addKey("CRPIX1", out_data.cols / 2.0, "");
        phdu.addKey("CRPIX2", out_data.rows / 2.0, "");
        phdu.addKey("CRVAL1", ra, "");
        phdu.addKey("CRVAL2", dec, "");
        phdu.addKey("CDELT1", dp, "");
        phdu.addKey("CDELT2", dp, "");
        
    } catch (FitsException& e) {
        std::cerr << "Error writing image: " << e.message() << std::endl;
    }
}

void write_spectrum_file(const std::string& filename, const std::vector<double>& spec, double exposure, const std::string& arffile, const std::string& rmffile) {
    try {
        std::remove(filename.c_str());
        FITS fits(filename, Write);
        
        // Table columns
        std::vector<string> colName(2);
        std::vector<string> colForm(2);
        std::vector<string> colUnit(2);
        
        colName[0] = "CHANNEL"; colForm[0] = "1J"; colUnit[0] = "";
        colName[1] = "COUNTS";  colForm[1] = "1J"; colUnit[1] = "count";
        
        Table* table = fits.addTable("SPECTRUM", spec.size(), colName, colForm, colUnit);
        
        std::valarray<int> channel(spec.size());
        std::valarray<int> counts(spec.size());
        
        for(size_t i=0; i<spec.size(); ++i) {
            channel[i] = i;
            counts[i] = (int)spec[i];
        }
        
        table->column("CHANNEL").write(channel, 1);
        table->column("COUNTS").write(counts, 1);
        
        table->addKey("EXPOSURE", exposure, "");
        table->addKey("ANCRFILE", arffile, "");
        table->addKey("RESPFILE", rmffile, "");
        table->addKey("TELESCOP", "THESEUS", "");
        table->addKey("INSTRUME", "XGIS", "");
        
    } catch (FitsException& e) {
        std::cerr << "Error writing spectrum: " << e.message() << std::endl;
    }
}

void write_lc_file(const std::string& filename, const std::vector<double>& times, const std::vector<double>& rates, const std::vector<double>& errors) {
    try {
        std::remove(filename.c_str());
        FITS fits(filename, Write);
        
        std::vector<string> colName(3);
        std::vector<string> colForm(3);
        std::vector<string> colUnit(3);
        
        colName[0] = "TIME";  colForm[0] = "1D"; colUnit[0] = "s";
        colName[1] = "RATE";  colForm[1] = "1E"; colUnit[1] = "counts/s";
        colName[2] = "ERROR"; colForm[2] = "1E"; colUnit[2] = "counts/s";
        
        Table* table = fits.addTable("LIGHTCURVE", times.size(), colName, colForm, colUnit);
        
        std::valarray<double> t_va(times.size());
        std::valarray<float> r_va(rates.size());
        std::valarray<float> e_va(errors.size());
        
        for(size_t i=0; i<times.size(); ++i) {
            t_va[i] = times[i];
            r_va[i] = (float)rates[i];
            e_va[i] = (float)errors[i];
        }
        
        table->column("TIME").write(t_va, 1);
        table->column("RATE").write(r_va, 1);
        table->column("ERROR").write(e_va, 1);
        
        table->addKey("TELESCOP", "THESEUS", "");
        table->addKey("INSTRUME", "XGIS", "");
        table->addKey("TIMEUNIT", "s", "");
        table->addKey("RATEUNIT", "counts/s", "");
        
    } catch (FitsException& e) {
        std::cerr << "Error writing LC: " << e.message() << std::endl;
    }
}
