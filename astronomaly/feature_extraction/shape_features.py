import numpy as np
import cv2
from astronomaly.base.base_pipeline import PipelineStage
from astronomaly.base import logging_tools

def find_contours(img, threshold):
    """
    Finds the contours of an image that meet a threshold

    Parameters
    ----------
    img : np.ndarray
        Input image (must be greyscale)
    threshold : float
        What threshold to use

    Returns
    -------
    contours
        opencv description of contours (each contour is a list of x,y values
        and there may be several contours, given as a list of lists)
    hierarchy
        opencv description of how contours relate to each other (see opencv 
        documentation)
    """

    img_bin = np.zeros(img.shape, dtype=np.uint8)

    img_bin[img <= threshold] = 0
    img_bin[img > threshold] = 1

    contours, hierarchy = cv2.findContours(img_bin, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def fit_ellipse(contour, image, return_params=False, filled=True):
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

    if filled:
        thickness = -1
        y_npix = image.shape[0]
        x_npix = image.shape[1]
        ellipse_arr = np.zeros([y_npix, x_npix], dtype=np.float)
    else:
        thickness = 1
        ellipse_arr = image.copy()

    # Sets some defaults for when the fitting fails
    default_return_params = [np.nan] * 5 
    raised_error = False

    try:
        ((x0, y0), (maj_axis, min_axis), theta) = cv2.fitEllipse(contour)
        ellipse_params = x0, y0, maj_axis, min_axis, theta

        if np.any(np.isnan(ellipse_params)) or y0 < 0 or x0 < 0:
            raised_error = True
            logging_tools.log('fit_ellipse failed with unknown error:')

        #if y0 < 0:
        #    raised_error = True
        #    logging_tools.log('fit_ellipse failed with unknown error:')

    except cv2.error as e:
        logging_tools.log('fit_ellipse failed with cv2 error:' + e.msg)
        raised_error = True

    if raised_error:
        if return_params:
            return ellipse_arr, default_return_params
        else:
            return ellipse_arr

    x0 = int(np.round(x0))
    y0 = int(np.round(y0))
    maj_axis = int(np.round(maj_axis))
    min_axis = int(np.round(min_axis))
    theta = int(np.round(theta))

    #print(ellipse_params, 'PARAMS')
    #print(x0,y0,'x0 y0')

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


def check_extending_ellipses(img, threshold, return_params=False):
    """
    Checks and flags images when the contour extends beyond the image size.
    Used to check whether the image size (window size) must be increased.

    Parameters
    ----------
    img : np.ndarray
        Input image (must be 2d, no channel information)
    threshold : 
        Threshold values for drawing the outermost contour.
    return_params : bool
        If true also returns the parameters of the fitted ellipse

    Returns
    -------
    boolean
        Value that flags whether the ellipse extending beyond the image or not.
    """

    width = img.shape[0]
    height = img.shape[1]

    new_width = width * 3
    new_height = height * 3 

    blank_canvas = np.zeros((new_width,new_height), dtype=np.float)

    contours, hierarchy = find_contours(img, threshold)
    
    # Sets some defaults for when the fitting fails
    default_return_params = [np.nan] * 5 
    raised_error = False

    try:
        ((x0, y0), (maj_axis, min_axis), theta) = cv2.fitEllipse(np.float32(contours[0]))
        ellipse_params = x0, y0, maj_axis, min_axis, theta

        if np.any(np.isnan(ellipse_params)) or y0 < 0 or x0 < 0:
            raised_error = True
            logging_tools.log('fit_ellipse failed with unknown error:')

    except cv2.error as e:
        logging_tools.log('fit_ellipse failed with cv2 error:' + e.msg)
        raised_error = True

    if raised_error:
        contour_extends = False
        return contour_extends

    x0_new = int(np.round(x0)) + (int(width))
    y0_new = int(np.round(y0)) + (int(height))
    maj_axis = int(np.round(maj_axis))
    min_axis = int(np.round(min_axis))
    theta = int(np.round(theta))
    
    ellipse = cv2.ellipse(blank_canvas, (x0_new, y0_new), (maj_axis // 2, min_axis // 2), theta, 0, 360, (1, 1, 1), 1)
    ellipse[int(width*1):int(width*2), int(height*1):int(height*2)] = 0

    if ellipse.any() != 0:
        dif = np.sqrt( (x0 - width/2)**2 + (y0 - height/2)**2 )
        new_window = int((max(min_axis, maj_axis) + dif) * 1.25)
        contour_extends = True
        return contour_extends, new_window
    else:
        contour_extends = False
        return contour_extends    


class EllipseFitFeatures(PipelineStage):
    def __init__(self, percentiles=[90, 70, 50, 0], channel=None, extending_ellipse=False, **kwargs):
        """
        Computes a fit to an ellipse for an input image. Translation and 
        rotation invariate features. Warning: it's strongly recommended to
        apply a sigma-clipping transform before running this feature extraction
        algorithm.

        Parameters
        ----------
        channel : int
            Specify which channel to use for multiband images
        percentiles : array-like
            What percentiles to use as thresholds for the ellipses
        extending_ellipse : boolean
            Activates the check that determins whether or not the outermost ellipse 
            extends beyond the image
        """

        super().__init__(percentiles=percentiles, channel=channel, extending_ellipse=extending_ellipse, **kwargs)

        self.percentiles = percentiles
        self.labels = []
        feat_labs = ['Residual_%d', 'Offset_%d', 'Aspect_%d', 'Theta_%d']
        for f in feat_labs:
            for n in percentiles:
                self.labels.append(f % n)
        self.channel = channel
        self.extending_ellipse = extending_ellipse

        print(extending_ellipse)

        if extending_ellipse:
            self.labels.append('Warning_Open_Ellipse')
            self.labels.append('Recommended_Window_Size')

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

        # Get rid of possible NaNs
        # this_image = np.nan_to_num(this_image)

        x0 = y0 = -1
        x_cent = this_image.shape[0] // 2
        y_cent = this_image.shape[1] // 2

        warning_open_ellipses = []
        new_window = []
        all_contours = []
        feats = []
        stop = False

        upper_limit = 300
        scale = [i for i in np.arange(100, upper_limit + 1, 1)]

        # Start with the closest in contour (highest percentile)
        percentiles = np.sort(self.percentiles)[::-1] 

        if np.all(this_image == 0):
            failed = True
            failure_message = "Invalid cutout for feature extraction"
        else:
            failed = False
            failure_message = ""

        for a in scale:
            drawn_contours = []
            all_ellipses = []
            lst = []
            feats = []

            for p in percentiles:
                lst.append(p)
                width = int(image.shape[1] * a / 100)
                height = int(image.shape[0] * a / 100)
                dim = (width, height)
                resize = cv2.resize(this_image, dim, interpolation=cv2.INTER_AREA)

                if failed:
                    contours = []
                else:
                    thresh = np.percentile(resize[resize > 0], p)
                    contours, hierarchy = find_contours(resize, thresh)

                    x_contours = np.zeros(len(contours))
                    y_contours = np.zeros(len(contours))

                # First attempt to find the central point of the inner most contour
                if len(contours) != 0:
                    for k in range(len(contours)):
                        M = cv2.moments(contours[k])
                        try:
                            x_contours[k] = int(M["m10"] / M["m00"])
                            y_contours[k] = int(M["m01"] / M["m00"])
                        except ZeroDivisionError:
                            pass
                    if x0 == -1:
                        x_diff = x_contours - x_cent
                        y_diff = y_contours - y_cent
                    else:
                        x_diff = x_contours - x0
                        y_diff = y_contours - y0

                    # Will try to find the CLOSEST contour to the central one
                    r_diff = np.sqrt(x_diff**2 + y_diff**2)

                    ind = np.argmin(r_diff)

                    if x0 == -1:
                        x0 = x_contours[ind]
                        y0 = y_contours[ind]

                    c = contours[ind]

                    if len(c) < 5:
                        break

                    params = get_ellipse_leastsq(c, resize)

                    # Check whether or not the outermost ellipse extends beyond the image
                    if self.extending_ellipse and p == percentiles[-1]:
                        check = check_extending_ellipses(resize, thresh)
                        if check:
                            new_window.append(int(check[1]))
                            warning_open_ellipses.append(1)
                        else:
                            new_window.append(image.shape[0])
                            warning_open_ellipses.append(0)
                        
                    #ellipse_arr, param = fit_ellipse(c, resize, return_params=True, filled=False)
                    ellipse_arr = fit_ellipse(c, resize, return_params=False, filled=False)

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

                    all_ellipses.append(ellipse_arr)
                    all_contours.append(c)

                    draw = draw_contour(c, resize)
                    drawn_contours.append(draw)

                else:
                    failed = True
                    failure_message = "No contour found"

                if failed:
                    feats.append([np.nan] * 5)
                    logging_tools.log(failure_message)

                # Now we have the leastsq value, x0, y0, aspect_ratio, theta for each 
                # sigma
                # Normalise things relative to the highest threshold value
                # If there were problems with any sigma levels, set all values to NaNs
                if np.any(np.isnan(feats)):
                    return [np.nan] * 4 * len(self.percentiles)
                else:
                    max_ind = np.argmax(self.percentiles)

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
                        features = np.hstack((residuals, dist_to_centre, aspect, theta))

                    if len(lst) == len(percentiles):
                        stop = True

            if stop:
                break
            if a == upper_limit:
                features = [np.nan] * 4 * len(self.percentiles) 

        features = np.append(features,warning_open_ellipses)
        features = np.append(features,new_window)

        return features


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
