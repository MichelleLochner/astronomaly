import numpy as np
from astronomaly.base.base_pipeline import PipelineStage
import cv2


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

    thresh = n_sigma * img.std()
    img_bin = np.zeros(img.shape, dtype=np.uint8)

    img_bin[img <= thresh] = 0
    img_bin[img > thresh] = 1

    contours, hierarchy = cv2.findContours(img_bin, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def fit_ellipse(contour, image):
    """
    Fits an ellipse to a (single) contour.

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
    try:
        ((x0, y0), (maj_axis, min_axis), theta) = cv2.fitEllipse(contour)
    except cv2.error as e:
        print(e)
        print('Setting fit to zero')
        return 0

    x0 = int(np.round(x0))
    y0 = int(np.round(y0))
    maj_axis = int(np.round(maj_axis))
    min_axis = int(np.round(min_axis))
    theta = int(np.round(theta))

    y_npix = image.shape[0]
    x_npix = image.shape[1]

    ellipse_arr = np.zeros([y_npix, x_npix], dtype=np.float)
    contour_arr = np.zeros([y_npix, x_npix], dtype=np.float)

    cv2.ellipse(ellipse_arr, (x0, y0), (maj_axis // 2, min_axis // 2), 
                theta, 0, 360, (1, 1, 1), thickness)
    cv2.drawContours(contour_arr, [contour], 0, (1, 1, 1), thickness)
    res = np.sum((ellipse_arr - contour_arr)**2) / np.prod(contour_arr.shape)

    return res


class EllipseFitFeatures(PipelineStage):
    def __init__(self, sigma_levels=[1, 2, 3, 4, 5], channel=None, **kwargs):
        """
        Computes the power spectral density for an input image. Translation and 
        rotation invariate features.

        Parameters
        ----------
        sigma_levels : array-like
            The levels at which to calculate the contours in numbers of
            standard deviations of the image.
        """

        super().__init__(sigma_levels=sigma_levels, channel=channel, **kwargs)

        self.sigma_levels = sigma_levels
        self.labels = ['Contour_%d' % n for n in sigma_levels]
        self.channel = channel

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

        feats = []
        for n in self.sigma_levels:
            contours, hierarchy = find_contours(this_image, n_sigma=n)
            found = False

            for c in contours:
                # Only take the contour in the centre of the image
                # *** Update to be more general
                x0 = this_image.shape[0] // 2
                y0 = this_image.shape[1] // 2
                in_contour = cv2.pointPolygonTest(c, (x0, y0), False)

                if in_contour == 1 and not found:
                    feats.append(fit_ellipse(c, this_image))
                    found = True

            if not found:
                feats.append(0)
        feats = np.hstack(feats)

        return feats
