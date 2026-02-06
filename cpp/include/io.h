#ifndef IO_H
#define IO_H

#include "utils.h"
#include <string>
#include <vector>

void write_img_file(const std::string& filename, const Matrix2D& data, double ra, double dec);
void write_spectrum_file(const std::string& filename, const std::vector<double>& spec, double exposure, const std::string& arffile, const std::string& rmffile);
void write_lc_file(const std::string& filename, const std::vector<double>& times, const std::vector<double>& rates, const std::vector<double>& errors);
EventData read_event_file(const std::string& filename);
Matrix2D read_fits_image(const std::string& filename);

#endif // IO_H
