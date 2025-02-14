import numpy as np
import pickle

from astropy.io import fits
from astropy.time import Time
from astropy import coordinates as coord
import jplephem
import astropy.units as u

class AperturePhotometry:
    def __init__(self):
        """
        Initializes the class with default parameters, such as readout noise,
        gain, and precomputed calibration files (bias, flat fields, etc.).
        """
        self.readout_noise = 7.38  # [e] Photoelectrons per pixel
        self.gain = 2.73  # [e/ADU] Conversion factor between ADU and electrons

        # Bias-related calibration data
        self.bias_std = 1.31  # Standard deviation of bias [e]
        self.median_bias = pickle.load(open('median_bias.p', 'rb'))
        self.median_bias_error = pickle.load(open('median_bias_error.p', 'rb'))

        # Flat-field-related calibration data
        self.median_normalised_flat = pickle.load(open('median_normalized_flat.p', 'rb'))
        self.median_normalised_flat_errors = pickle.load(open('median_normalized_flat_errors.p', 'rb'))

        # Science frame file paths and metadata
        self.science_path = './science/'  # Path to science data
        self.science_list = np.genfromtxt(self.science_path + 'science.list', dtype=str)  # List of science frames
        self.science_size = len(self.science_list)  # Number of science frames

        # Define the X and Y axes for pixel indexing
        ylen, xlen = np.shape(self.median_bias)
        X_axis = np.arange(0, xlen, 1)  # X pixel coordinates
        Y_axis = np.arange(0, ylen, 1)  # Y pixel coordinates
        self.X, self.Y = np.meshgrid(X_axis, Y_axis)  # Create 2D coordinate grids
        self.X_axis = X_axis
        self.Y_axis = Y_axis

        #self.target = coord.SkyCoord("20:13:31.63", "+65:09:44'", unit=(u.hourangle, u.deg), frame='icrs')
        self.target = coord.SkyCoord("13:57:33", "+43:29:36", unit=(u.hourangle, u.deg), frame='icrs')
        #self.observatory_location = ('45.8472d', '11.569d')  # asiago observatory coordinates
        self.observatory_location = ('45.84848d', '11.56900d')  # Coordinates from the header for Asiago Observatory


    def provide_aperture_parameters(self, sky_inner_radius, sky_outer_radius, aperture_radius, x_initial, y_initial):
        """
        Loads parameters required for aperture photometry and centroid refinement.

        Parameters:
        - sky_inner_radius: Inner radius for background annulus [pixels].
        - sky_outer_radius: Outer radius for background annulus [pixels].
        - aperture_radius: Radius for aperture photometry [pixels].
        - x_initial: Initial X-coordinate of the target [pixels].
        - y_initial: Initial Y-coordinate of the target [pixels].
        """
        self.sky_inner_radius = sky_inner_radius
        self.sky_outer_radius = sky_outer_radius
        self.aperture_radius = aperture_radius
        self.x_initial = x_initial
        self.y_initial = y_initial

    def compute_centroid(self, science_data, x_target_initial, y_target_initial, maximum_number_of_iterations=20):
        """
        Refines the centroid of a target star by iteratively calculating weighted averages.
        """
        for i_iter in range(maximum_number_of_iterations):
            if i_iter == 0:
                x_target_previous = x_target_initial
                y_target_previous = y_target_initial
            else:
                x_target_previous = x_target_refined
                y_target_previous = y_target_refined

            # 2D array with the distance of each pixel from the target star 
            target_distance = np.sqrt((self.X - x_target_previous)**2 + (self.Y - y_target_previous)**2)
            # Selection of the pixels within the inner radius
            annulus_sel = (target_distance < self.sky_inner_radius)

            # weighted sum of coordinates
            weighted_X = np.sum(science_data[annulus_sel] * self.X[annulus_sel])
            weighted_Y = np.sum(science_data[annulus_sel] * self.Y[annulus_sel])

            # sum of weights
            total_flux = np.sum(science_data[annulus_sel])

            #refined determination of coordinates.
            x_target_refined = weighted_X / total_flux
            y_target_refined = weighted_Y / total_flux

            #stopping if convergence is below 0.1%.
            percent_variance_x = (x_target_refined - x_target_previous) / x_target_previous * 100
            percent_variance_y = (y_target_refined - y_target_previous) / y_target_previous * 100

            # exit condition: both percent variance are smaller than 0.1%
            if np.abs(percent_variance_x) < 0.1 and np.abs(percent_variance_y) < 0.1:
                break

        return x_target_refined, y_target_refined

    def compute_sky_background(self, science_data, science_data_errors, x_pos, y_pos):
        """
        Calculates the background sky flux using an annulus around the target.
        """
        target_distance = np.sqrt((self.X - x_pos)**2 + (self.Y - y_pos)**2)
        
        annulus_selection = (target_distance > self.sky_inner_radius) & (target_distance <= self.sky_outer_radius)

        sky_flux_median = np.median(science_data[annulus_selection])

        N_sky = np.sum(annulus_selection)
        sky_flux_error = np.sqrt(np.sum(science_data_errors[annulus_selection]**2)) / N_sky

        
        return sky_flux_median, sky_flux_error

    def determine_FWHM_axis(self, reference_axis, normalised_cumulative_distribution):
        # Find the closest point to NCD= 0.15865 (-1 sigma)
        NCD_index_left = np.argmin(np.abs(normalised_cumulative_distribution-0.15865))
    
        # Find the closest point to NCD= 0.84135 (+1 sigma)
        NCD_index_right = np.argmin(np.abs(normalised_cumulative_distribution-0.84135))

        # We model the NCD around the -1sigma value with a polynomial curve. 
        # The independent variable is actually the normalized cumulative distribution, 
        # the dependent variable is the pixel position
        p_fitted = np.polynomial.Polynomial.fit(normalised_cumulative_distribution[NCD_index_left-1: NCD_index_left+2],
                                            reference_axis[NCD_index_left-1: NCD_index_left+2],
                                            deg=2)

        # We get a more precise estimate of the pixel value corresponding to the -1sigma position
        pixel_left = p_fitted(0.15865)

        # We repeat the step for the 1sigma value
        p_fitted = np.polynomial.Polynomial.fit(normalised_cumulative_distribution[NCD_index_right-1: NCD_index_right+2],
                                            reference_axis[NCD_index_right-1: NCD_index_right+2],
                                            deg=2)
        pixel_right = p_fitted(0.84135)

        FWHM_factor = 2 * np.sqrt(2 * np.log(2)) # = 2.35482
        FWHM = (pixel_right-pixel_left)/2. * FWHM_factor

        return FWHM

    def compute_fwhm(self, science_data, x_pos, y_pos, radius):
        """
        Compute FWHM along X and Y directions for the star located at x_pos, y_pos.
        We select pixels within the chosen radius and compute cumulative sums.
        """
        target_distance = np.sqrt((self.X - x_pos)**2 + (self.Y - y_pos)**2)
        sel = (target_distance < radius)

        total_flux = np.nansum(science_data * sel)
        flux_x = np.nansum(science_data * sel, axis=0)
        flux_y = np.nansum(science_data * sel, axis=1)

        cumulative_sum_x = np.cumsum(flux_x) / total_flux
        cumulative_sum_y = np.cumsum(flux_y) / total_flux

        FWHM_x = self.determine_FWHM_axis(self.X_axis, cumulative_sum_x)
        FWHM_y = self.determine_FWHM_axis(self.Y_axis, cumulative_sum_y)

        return FWHM_x, FWHM_y

    def aperture_photometry(self):
        """
        Performs aperture photometry for all science frames.
        """
        # Initialize arrays for results
        self.airmass = np.empty(self.science_size)
        self.exptime = np.empty(self.science_size)
        self.julian_date = np.empty(self.science_size)

        self.aperture = np.empty(self.science_size)
        self.aperture_errors = np.empty(self.science_size)
        self.sky_background = np.empty(self.science_size)
        self.sky_background_errors = np.empty(self.science_size)

        self.x_position = np.empty(self.science_size)
        self.y_position = np.empty(self.science_size)

        self.x_fwhm = np.empty(self.science_size)
        self.y_fwhm = np.empty(self.science_size)

        x_ref_init = self.x_initial
        y_ref_init = self.y_initial

        for ii_science, science_name in enumerate(self.science_list):
            # Open science frame and extract data
            science_fits = fits.open(self.science_path + science_name)
            hdr = science_fits[0].header
            self.airmass[ii_science] = hdr['AIRMASS']
            self.exptime[ii_science] = hdr['EXPTIME']
            self.julian_date[ii_science] = hdr['JD']

            science_data = science_fits[0].data * self.gain
            science_fits.close()

            # Correct frame
            science_corrected, science_corrected_errors = self.correct_science_frame(science_data)

            # Refine centroid position
            x_refined, y_refined = self.compute_centroid(science_corrected, x_ref_init, y_ref_init)

            # Compute sky background
            sky_median, sky_error = self.compute_sky_background(science_corrected, science_corrected_errors,
                                                                x_refined, y_refined)
            self.sky_background[ii_science] = sky_median
            self.sky_background_errors[ii_science] = sky_error

            # Subtract sky
            science_sky_corrected = science_corrected - sky_median
            science_sky_corrected_errors = np.sqrt(science_corrected_errors**2 + sky_error**2)

            # Recompute centroid after sky subtraction
            x_refined, y_refined = self.compute_centroid(science_sky_corrected, x_refined, y_refined)

            # Aperture photometry
            target_distance = np.sqrt((self.X - x_refined)**2 + (self.Y - y_refined)**2)
            aperture_selection = (target_distance < self.aperture_radius)
            self.aperture[ii_science] = np.sum(science_sky_corrected[aperture_selection])
            self.aperture_errors[ii_science] = np.sqrt(np.sum((science_sky_corrected_errors[aperture_selection])**2))

            self.x_position[ii_science] = x_refined
            self.y_position[ii_science] = y_refined

            # Compute FWHM
            fwhm_x, fwhm_y = self.compute_fwhm(science_sky_corrected, x_refined, y_refined,
                                               radius=self.sky_inner_radius)
            self.x_fwhm[ii_science] = fwhm_x
            self.y_fwhm[ii_science] = fwhm_y

            # Update initial guess for next iteration
            x_ref_init = x_refined
            y_ref_init = y_refined

         # Convert JD to BJD_TDB (mid exposure)
        jd_mid = self.julian_date + self.exptime/86400./2.
        tm = Time(jd_mid, format='jd', scale='utc', location=self.observatory_location)
        ltt_bary = tm.light_travel_time(self.target, ephemeris='jpl')
        self.bjd_tdb = tm.tdb + ltt_bary

        # For compatibility
        self.x_refined = self.x_position
        self.y_refined = self.y_position


    def correct_science_frame(self, science_data):
        """
        Applies bias subtraction and flat-field correction to the raw science frame.
        """
        # Bias subtraction
        science_debiased = science_data - self.median_bias

        science_corrected = science_debiased / self.median_normalised_flat
        # Compute errors
        science_debiased_errors = np.sqrt(
            self.readout_noise**2 + self.median_bias_error**2 + science_debiased)
        

        capture = (science_debiased != 0) & (self.median_normalised_flat != 0)
        science_corrected_errors = np.zeros_like(science_corrected)
        science_corrected_errors[capture] = science_corrected[capture] * np.sqrt(
            (science_debiased_errors[capture] / science_debiased[capture])**2 +
            (self.median_normalised_flat_errors[capture] / self.median_normalised_flat[capture])**2
        )
        # For invalid pixels:
        science_corrected_errors[~capture] = 0.0

        return science_corrected, science_corrected_errors


    