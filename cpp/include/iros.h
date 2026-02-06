#ifndef IROS_H
#define IROS_H

#include "utils.h"

Matrix2D make_img(const EventData& data, double emin, double emax, double tmin = -1.0, double tmax = -1.0);
Matrix2D reconst_new(const Matrix2D& mask, const Matrix2D& det, const Matrix2D& w);

#endif // IROS_H
