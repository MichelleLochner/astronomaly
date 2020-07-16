import numpy as np
import cv2
from astronomaly.base.base_pipeline import PipelineStage
from astronomaly.base import logging_tools


def find_contours(img, n_sigma):
    """
    Finds the contours of an image that meet a threshold, defined as the 
    n_sigma * standard deviation of the image.

    Parameters
    ----------
    img : np.ndarray
        Input image (must be greyscale)
    n_sigma : float
        Number of standard deviations above zero to threshold.

    Returns
    -------
    contours
        opencv description of contours (each contour is a list of x,y values
        and there may be several contours, given as a list of lists)
    hierarchy
        opencv description of how contours relate to each other (see opencv 
        documentation)
    """

    thresh = n_sigma * img.std() + np.median(img)
    img_bin = np.zeros(img.shape, dtype=np.uint8)

    img_bin[img <= thresh] = 0
    img_bin[img > thresh] = 1

    contours, hierarchy = cv2.findContours(img_bin, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def fit_ellipse(contour, image, return_params=False):
    """
    Fits an ellipse to a contour and returns a binary image representation of
    the ellipse.

    Parameters
    ----------
    contour : np.ndarray
        Array of x,y values describing the contours (as returned by opencv's
        findCountours function)
    image : np.ndarray
        The original image the contour was fit to.
    return_params : bool
        If true also returns the parameters of the fitted ellipse

    Returns
    -------
    np.ndarray
        2d binary image with representation of the ellipse
    """

    thickness = -1
    y_npix = image.shape[0]
    x_npix = image.shape[1]
    ellipse_arr = np.zeros([y_npix, x_npix], dtype=np.float)

    # Sets some defaults for when the fitting fails
    default_return_params = [np.nan] * 5 

    try:
        ((x0, y0), (maj_axis, min_axis), theta) = cv2.fitEllipse(contour)
    except cv2.error as e:
        logging_tools.log('fit_ellipse failed with cv2 error:' + e.msg)
        if return_params:
            return ellipse_arr, default_return_params
        else:
            return ellipse_arr

    ellipse_params = x0, y0, maj_axis, min_axis, theta

    # Sometimes the ellipse fitting function produces insane values
    # if not (0 <= x0 <= x_npix) or not (0 <= y0 <= y_npix):
    #     print('Ellipse fitting failed')
    #     if return_params:
    #         return ellipse_arr, default_return_params
    #     else:
    #         return ellipse_arr

    x0 = int(np.round(x0))
    y0 = int(np.round(y0))
    maj_axis = int(np.round(maj_axis))
    min_axis = int(np.round(min_axis))
    theta = int(np.round(theta))

    cv2.ellipse(ellipse_arr, (x0, y0), (maj_axis // 2, min_axis // 2), 
                theta, 0, 360, (1, 1, 1), thickness)

    if return_params:
        return ellipse_arr, ellipse_params
    else:
        return ellipse_arr


def get_ellipse_leastsq(contour, image):
    """
    Fits an ellipse to a (single) contour and returns the sum of the
    differences squared between the fitted ellipse and contour (normalised).

    Parameters
    ----------
    contour : np.ndarray
        Array of x,y values describing the contours (as returned by opencv's
        findCountours function)
    image : np.ndarray
        The original image the contour was fit to.

    Returns
    -------
    float
        sum((ellipse-contour)^2)/number_of_pixels
    """
    thickness = -1
    y_npix = image.shape[0]
    x_npix = image.shape[1]

    contour_arr = np.zeros([y_npix, x_npix], dtype=np.float)
    cv2.drawContours(contour_arr, [contour], 0, (1, 1, 1), thickness)

    ellipse_arr, params = fit_ellipse(contour, image, return_params=True)

    if np.any(np.isnan(params)):
        res = np.nan
    else:
        arr_diff = ellipse_arr - contour_arr
        res = np.sum((arr_diff)**2) / np.prod(contour_arr.shape)

    return [res] + list(params)


def draw_contour(contour, image, filled=False):
    """
    Draws a contour onto an image for diagnostic purposes

    Parameters
    ----------
    contour : np.ndarray
        Array of x,y values describing the contours (as returned by opencv's
        findCountours function)
    image : np.ndarray
        The original image the contour was fit to.
    filled : bool, optional
        If true will fill in the contour otherwise will return an outline.
    Returns
    -------
    np.ndarray
        The image with the drawn contour on top
    """

    if filled:
        thickness = -1
        contour_arr = np.zeros([image.shape[0], image.shape[1]])
    else:
        thickness = 1
        contour_arr = image.copy()

    cv2.drawContours(contour_arr, [contour], 0, (1, 1, 1), thickness)

    return contour_arr


def extract_contour(contours, x0, y0):
    """
    Utility function to determine which contour contains the points given.
    Note by default this will only return the first contour it finds to contain
    x0, y0.

    Parameters
    ----------
    contours : np.ndarray
        Array of x,y values describing the contours (as returned by opencv's
        findCountours function)
    x0 : int
        x value to test
    y0 : int
        y value to test

    Returns
    -------
    contour : np.ndarray
        Returns the single contour that contains (x0,y0)
    """

    for c in contours:
        if cv2.pointPolygonTest(c, (x0, y0), False) == 1:
            return c

    print('No contour found around points given')
    raise TypeError


def get_hu_moments(img):
    """
    Extracts the Hu moments for an image. Note this often works best with
    simple, clean shapes like filled contours.

    Parameters
    ----------
    img : np.ndarray
        Input image (must be 2d, no channel information)

    Returns
    -------
    np.ndarray
        The 7 Hu moments for the image
    """
    moms = cv2.moments(img)
    hu_feats = cv2.HuMoments(moms)
    hu_feats = hu_feats.flatten()

    return hu_feats


class EllipseFitFeatures(PipelineStage):
    def __init__(self, sigma_levels=[1, 2, 3, 4, 5], channel=None, 
                 central_contour=True, **kwargs):
        """
        Computes a fit to an ellipse for an input image. Translation and 
        rotation invariate features.

        Parameters
        ----------
        sigma_levels : array-like
            The levels at which to calculate the contours in numbers of
            standard deviations of the image.
        channel : int
            Specify which channel to use for multiband images
        central_contour : bool
            If true will only use the contour surrounding the centre of the 
            image
        """

        super().__init__(sigma_levels=sigma_levels, channel=channel, 
                         central_contour=central_contour, **kwargs)

        self.sigma_levels = sigma_levels
        self.labels = []
        feat_labs = ['Residual_%d', 'Offset_%d', 'Aspect_%d', 'Theta_%d']
        for f in feat_labs:
            for n in sigma_levels:
                self.labels.append(f % n)
        self.channel = channel
        self.central_contour = central_contour

    def _execute_function(self, image):
        """
        Does the work in actually extracting the ellipse fitted features

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        array
            Contains the extracted ellipse fitted features
        """

        # First check the array is normalised since opencv will cry otherwise
        if len(image.shape) > 2:
            if self.channel is None:
                raise ValueError('Contours cannot be determined for \
                                  multi-channel images, please set the \
                                  channel kwarg.')
            else:
                this_image = image[:, :, self.channel]
        else:
            this_image = image

        x0 = y0 = -1
        x_cent = this_image.shape[0]
        y_cent = this_image.shape[1]

        feats = []
        # Start with the closest in contour (highest sigma)
        sigma_levels = np.sort(self.sigma_levels)[::-1] 

        failed = False
        failure_message = ""

        for n in sigma_levels:
            contours, hierarchy = find_contours(this_image, n_sigma=n)

            x_contours = np.zeros(len(contours))
            y_contours = np.zeros(len(contours))

            found = False

            # First attempt to find the central point of the inner most contour
            if x0 == -1 and len(contours) != 0:
                for k in range(len(contours)):
                    M = cv2.moments(contours[k])
                    try:
                        x_contours[k] = int(M["m10"] / M["m00"])
                        y_contours[k] = int(M["m01"] / M["m00"])
                    except ZeroDivisionError:
                        pass
                x_diff = x_contours - x_cent
                y_diff = y_contours - y_cent
                r_diff = np.sqrt(x_diff**2 + y_diff**2)

                ind = np.argmin(r_diff)

                x0 = x_contours[ind]
                y0 = y_contours[ind]

            for c in contours:

                if x0 == -1:
                    # This happens if a 5-sigma contour isn't found
                    # Usually because of a bright artifact on the edge
                    failed = True
                    failure_message = "Failed to detect all n_sigma contours. \
                                       Applying a Gaussian window may help."

                else:
                    in_contour = cv2.pointPolygonTest(c, (x0, y0), False)

                    if in_contour == 1 and not found:
                        params = get_ellipse_leastsq(c, this_image)
                        # Params return in this order:
                        # residual, x0, y0, maj_axis, min_axis, theta
                        if np.any(np.isnan(params)):
                            failed = True
                        else:
                            if params[3] == 0 or params[4] == 0:
                                aspect = 1
                            else:
                                aspect = params[4] / params[3]

                            if aspect < 1:
                                aspect = 1 / aspect
                            if aspect > 100:
                                aspect = 1

                            new_params = params[:3] + [aspect] + [params[-1]]
                            feats.append(new_params)
                            found = True

            if not found:
                failed = True
                failure_message = "No contour found for n_sigma=" + str(n)
            if failed:
                feats.append([np.nan] * 5)
                logging_tools.log(failure_message)

        # Now we have the leastsq value, x0, y0, aspect_ratio, theta for each 
        # sigma
        # Normalise things relative to the highest sigma value
        # If there were problems with any sigma levels, set all values to NaNs
        if np.any(np.isnan(feats)):
            return [np.nan] * 4 * len(self.sigma_levels)
        else:
            max_ind = np.argmax(self.sigma_levels)

            residuals = []
            dist_to_centre = []
            aspect = []
            theta = []

            x0_max_sigma = feats[max_ind][1]
            y0_max_sigma = feats[max_ind][2]
            aspect_max_sigma = feats[max_ind][3]
            theta_max_sigma = feats[max_ind][4]

            for n in range(len(feats)):
                prms = feats[n]
                residuals.append(prms[0])
                if prms[1] == 0 or prms[2] == 0:
                    r = 0
                else:
                    x_diff = prms[1] - x0_max_sigma
                    y_diff = prms[2] - y0_max_sigma
                    r = np.sqrt((x_diff)**2 + (y_diff)**2)
                dist_to_centre.append(r)
                aspect.append(prms[3] / aspect_max_sigma)
                theta_diff = np.abs(prms[4] - theta_max_sigma) % 360
                # Because there's redundancy about which way an ellipse 
                # is aligned, we always take the acute angle
                if theta_diff > 90:
                    theta_diff -= 90
                theta.append(theta_diff)

            return np.hstack((residuals, dist_to_centre, aspect, theta))


class HuMomentsFeatures(PipelineStage):
    def __init__(self, sigma_levels=[1, 2, 3, 4, 5], channel=None, 
                 central_contour=False, **kwargs):
        """
        Computes the Hu moments for the contours at specified levels in an
        image. 

        Parameters
        ----------
        sigma_levels : array-like
            The levels at which to calculate the contours in numbers of
            standard deviations of the image.
        channel : int
            Specify which channel to use for multiband images
        central_contour : bool
            If true will only use the contour surrounding the centre of the 
            image
        """

        super().__init__(sigma_levels=sigma_levels, channel=channel, 
                         central_contour=central_contour, **kwargs)

        self.sigma_levels = sigma_levels
        self.channel = channel
        self.central_contour = central_contour

        hu_labels = ['I%d' % i for i in range(7)]
        sigma_labels = ['level%d' % n for n in sigma_levels]
        self.labels = []
        for s in sigma_labels:
            for h in hu_labels:
                self.labels.append(s + '_' + h)

    def _execute_function(self, image):
        """
        Does the work in actually extracting the Hu moments

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        array
            Contains the Hu moments for each contour level
        """

        # First check the array is normalised since opencv will cry otherwise
        if len(image.shape) > 2:
            if self.channel is None:
                raise ValueError('Contours cannot be determined for \
                                  multi-channel images, please set the \
                                  channel kwarg.')
            else:
                this_image = image[:, :, self.channel]
        else:
            this_image = image

        if self.central_contour:
            x0 = this_image.shape[0] // 2
            y0 = this_image.shape[1] // 2
        else:
            x0 = y0 = -1
        feats = []

        for n in self.sigma_levels:
            contours, hierarchy = find_contours(this_image, n_sigma=n)
            found = False

            for c in contours:
                # Only take the contour in the centre of the image

                if x0 == -1:
                    # We haven't set which contour we're going to look at
                    # default to the largest
                    lengths = [len(cont) for cont in contours]
                    largest_cont = contours[np.argmax(lengths)]
                    M = cv2.moments(largest_cont)
                    x0 = int(M["m10"] / M["m00"])
                    y0 = int(M["m01"] / M["m00"])

                in_contour = cv2.pointPolygonTest(c, (x0, y0), False)

                if in_contour == 1 and not found:
                    contour_img = draw_contour(c, this_image)
                    feats.append(get_hu_moments(contour_img))
                    found = True

            if not found:
                feats.append([0] * 7)
        feats = np.hstack(feats)

        return feats
