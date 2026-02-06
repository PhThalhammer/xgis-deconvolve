#include "utils.h"
#include "iros.h"
#include "io.h"
#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>
#include <cstdlib>
#include <algorithm>
#include <iomanip>

struct Args {
    double ra_src = 0.0;
    double dec_src = 0.0;
    std::string outfile;
    std::string infile;
    std::string mask;
    std::string weight;
    std::string arf;
    std::string rmf;
    bool no_lc = false;
    bool no_sp = false;
    int nbins = 100;
    int nsample = 1;
};

void print_help(const char* progname) {
    std::cout << "Usage: " << progname << " [options]"  << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --ra_src <deg>    Right Ascension of source " << std::endl;
    std::cout << "  --dec_src <deg>   Declination of source " << std::endl;
    std::cout << "  --outfile <str>   Output filename prefix " << std::endl;
    std::cout << "  --infile <str>    Input FITS event file " << std::endl;
    std::cout << "  --mask <str>      Mask FITS file " << std::endl;
    std::cout << "  --weight <str>    Weight FITS file " << std::endl;
    std::cout << "  --arf <str>       ARF directory/file " << std::endl;
    std::cout << "  --rmf <str>       RMF directory/file " << std::endl;
    std::cout << "  --no_lc           Do not generate lightcurve " << std::endl;
    std::cout << "  --no_sp           Do not generate spectrum " << std::endl; 
    std::cout << "  --nbins <int>     Number of time bins (default 100) " << std::endl;
    std::cout << "  --nsample <int>   Oversampling factor (default 1) " << std::endl;
    std::cout << "  --help            Show this help " << std::endl;
}

int main(int argc, char* argv[]) {
    Args args;
    const char* home_env = std::getenv("HOME");
    std::string home = home_env ? home_env : "";
    
    // Defaults
    args.mask = home + "/userdata/coded_mask/theseus-xgis/instruments/theseus-xgis/instdata/msk_55_4752313.fits";
    args.weight = home + "/userdata/coded_mask/weight/weight_di.fits";
    args.arf = home + "/userdata/coded_mask/theseus-xgis/instruments/theseus-xgis/instdata/arfgrid/";
    args.rmf = home + "/userdata/coded_mask/theseus-xgis/instruments/theseus-xgis/instdata/rmfgrid/";

    static struct option long_options[] = {
        {"ra_src", required_argument, 0, 'r'},
        {"dec_src", required_argument, 0, 'd'},
        {"outfile", required_argument, 0, 'o'},
        {"infile", required_argument, 0, 'i'},
        {"mask", required_argument, 0, 'm'},
        {"weight", required_argument, 0, 'w'},
        {"arf", required_argument, 0, 'a'},
        {"rmf", required_argument, 0, 'R'},
        {"no_lc", no_argument, 0, 'L'},
        {"no_sp", no_argument, 0, 'S'},
        {"nbins", required_argument, 0, 'n'},
        {"nsample", required_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'r': args.ra_src = std::atof(optarg); break;
            case 'd': args.dec_src = std::atof(optarg); break;
            case 'o': args.outfile = optarg; break;
            case 'i': args.infile = optarg; break;
            case 'm': args.mask = optarg; break;
            case 'w': args.weight = optarg; break;
            case 'a': args.arf = optarg; break;
            case 'R': args.rmf = optarg; break;
            case 'L': args.no_lc = true; break;
            case 'S': args.no_sp = true; break;
            case 'n': args.nbins = std::atoi(optarg); break;
            case 's': args.nsample = std::atoi(optarg); break;
            case 'h': print_help(argv[0]); return 0;
            default: print_help(argv[0]); return 1;
        }
    }

    if (args.infile.empty() || args.outfile.empty()) {
        std::cerr << "Error: --infile and --outfile are required. ";
        print_help(argv[0]);
        return 1;
    }

    std::cout << "Loading mask: " << args.mask << std::endl;
    Matrix2D mask = read_fits_image(args.mask);
    
    // Resample mask
    mask = resample_mask(mask, args.nsample);
    // Python: mask = np.transpose(mask)
    // In our Matrix2D (row-major), transpose:
    Matrix2D mask_T(mask.cols, mask.rows);
    for(size_t r=0; r<mask.rows; ++r) {
        for(size_t c=0; c<mask.cols; ++c) {
            mask_T(c, r) = mask(r, c);
        }
    }
    mask = mask_T;

    std::cout << "Loading weight: " << args.weight << std::endl;
    Matrix2D weight = read_fits_image(args.weight);

    std::cout << "Loading events: " << args.infile << std::endl;
    EventData ev = read_event_file(args.infile);
    
    double ra_pnt = ev.ra_pnt;
    double dec_pnt = ev.dec_pnt;
    double exposure = ev.exposure;

    std::cout << "Making detector image..." << std::endl;
    
    double min_time = 0;
    double max_time = 0;
    if (!ev.time.empty()) {
        auto mm = std::minmax_element(ev.time.begin(), ev.time.end());
        min_time = *mm.first;
        max_time = *mm.second;
    }
    
    double tstop_preview = min_time + 0.1 * (max_time - min_time);
    Matrix2D img = make_img(ev, 0, 150, min_time, tstop_preview); // Defaults emin 0, emax 150 in Python
    
    std::cout << "Making sky image..." << std::endl;
    Matrix2D sky = reconst_new(mask, img, weight);
    
    // Python: write_img_file(..., np.flipud(sky), ra, dec)
    Matrix2D sky_flipped = sky;
    flipud(sky_flipped);
    write_img_file(args.outfile + "_sky.fits", sky_flipped, ra_pnt, dec_pnt);
    
    double dp = (0.5 / 63.0) * (180.0 / PI);
    double crpix1 = sky.cols / 2.0;
    double crpix2 = sky.rows / 2.0;
    
    double src_x, src_y;
    world_to_pixel(args.ra_src, args.dec_src, ra_pnt, dec_pnt, dp, dp, crpix1, crpix2, src_x, src_y);
    
    int ix = (int)src_x;
    int iy = (int)src_y;
    std::cout << "Source position at pixel (x,y) = " << ix << ", " << iy << std::endl;

    // Spectrum
    if (!args.no_sp) {
        std::cout << "Extracting spectrum..." << std::endl;
        int minPHA = 0;
        int maxPHA = 56;
        int n_bins = maxPHA - minPHA + 1; // +2 in range means last edge is +1? 
        
        std::vector<double> spec(n_bins);
        
        for (int i = 0; i < n_bins; ++i) {
            double emin_i = minPHA + i;
            double emax_i = minPHA + i + 1;
            
            Matrix2D im = make_img(ev, emin_i, emax_i);
            Matrix2D sky_i = reconst_new(mask, im, weight);
            
            int x_min = std::max(0, ix - 10);
            int x_max = std::min((int)sky_i.rows, ix + 10); // Using rows for x slice?
            int y_min = std::max(0, iy - 10);
            int y_max = std::min((int)sky_i.cols, iy + 10); // Using cols for y slice?
            
            double max_val = 0.0;
            bool any_data = false;
            
            for (int r = x_min; r < x_max; ++r) {
                for (int c = y_min; c < y_max; ++c) {
                    double val = sky_i(r, c);
                    if (!any_data || val > max_val) {
                        max_val = val;
                        any_data = true;
                    }
                }
            }
            spec[i] = any_data ? max_val : 0.0;
        }
        
        // Angles for ARF/RMF
        double theta = std::atan2(iy, ix) * 180.0 / PI;
        // Phi = distance in degrees?
        double phi = 0.0;
        {
             double r1 = ra_pnt * PI/180; double d1 = dec_pnt * PI/180;
             double r2 = args.ra_src * PI/180; double d2 = args.dec_src * PI/180;
             double c = std::sin(d1)*std::sin(d2) + std::cos(d1)*std::cos(d2)*std::cos(r1-r2);
             phi = std::acos(c) * 180.0 / PI;
        }
        
        std::cout << "Theta (deg): " << theta << std::endl;
        std::cout << "Phi (deg): " << phi << std::endl;
        
        // Construct filename strings
        int theta_step = (int)(std::round(theta/15.0)*15);
        int phi_step = (int)(std::round(phi/45.0)*45);
        
        std::string arffile = args.arf + "/XGIS_X_theta_" + std::to_string(theta_step) + "_phi_" + std::to_string(phi_step) + ".arf";
        std::string rmffile = args.rmf + "/xgis_x_theta_" + std::to_string(theta_step) + "_phi_" + std::to_string(phi_step) + "_sixte_20251029.rmf";
        
        write_spectrum_file(args.outfile + "_spec.pha", spec, exposure, arffile, rmffile);
    }
    
    // Lightcurve
    if (!args.no_lc) {
        std::cout << "Extracting lightcurve..." << std::endl;
        // Python: t_edges = linspace(minT, maxT, nlc + 1)
        double duration = max_time - min_time;
        std::vector<double> tmids;
        std::vector<double> rates;
        std::vector<double> errors;
        
        for (int i = 0; i < args.nbins; ++i) {
            double tmin_i = min_time + i * (duration / args.nbins);
            double tmax_i = min_time + (i + 1) * (duration / args.nbins);
            
            // make_img(emin=0, emax=57)
            Matrix2D im = make_img(ev, 0, 57, tmin_i, tmax_i);
            Matrix2D sky_i = reconst_new(mask, im, weight);
            
            int x_min = std::max(0, ix - 10);
            int x_max = std::min((int)sky_i.rows, ix + 10);
            int y_min = std::max(0, iy - 10);
            int y_max = std::min((int)sky_i.cols, iy + 10);
            
            double max_val = 0.0;
            bool any_data = false;
            for (int r = x_min; r < x_max; ++r) {
                for (int c = y_min; c < y_max; ++c) {
                    double val = sky_i(r, c);
                    if (!any_data || val > max_val) {
                        max_val = val;
                        any_data = true;
                    }
                }
            }
            double rate = any_data ? max_val : 0.0;
            
            tmids.push_back((tmin_i + tmax_i) / 2.0);
            rates.push_back(rate);
            errors.push_back(std::sqrt(std::max(0.0, rate)));
        }
        
        write_lc_file(args.outfile + "_lc.fits", tmids, rates, errors);
    }

    return 0;
}
