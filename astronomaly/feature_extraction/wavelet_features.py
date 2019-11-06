import pywt
import numpy as np
from astronomaly.base.base_pipeline import PipelineStage


def flatten_swt2_coefficients(wavelet_coeffs):
    """
    A standardised way of flattening the swt2d coefficients
    They are stored as n_levels -> (cA, (cH, cV, cD)) where each of the sets 
    of coeffcients has a list of [npixels, npixels]

    Parameters
    ----------
    wavelet_coeffs : list
        Exactly as output by pywt

    Returns
    -------
    np.ndarray
        Flattened coefficients
    labels
        The labels of the coefficients

    """

    pixel_count = np.prod(wavelet_coeffs[0][0].shape)
    total_len = len(wavelet_coeffs) * 4 * pixel_count

    output_array = np.zeros(total_len)

    for lev in range(len(wavelet_coeffs)):
        approx_coeffs = wavelet_coeffs[lev][0]
        output_array[4 * lev * pixel_count:(4 * lev + 1) * pixel_count] = \
            approx_coeffs.reshape(pixel_count)
        for det in range(3):
            detailed_coeffs = wavelet_coeffs[lev][1][det]
            start = (4 * lev + det + 1) * pixel_count
            output_array[start:start + pixel_count] = detailed_coeffs.reshape(
                pixel_count)

    return output_array


def generate_labels(wavelet_coeffs):
    pixel_count = np.prod(wavelet_coeffs[0][0].shape)
    total_len = len(wavelet_coeffs) * 4 * pixel_count

    labels = np.zeros(total_len).astype('str')
    cfs = ['H', 'V', 'D']

    for lev in range(len(wavelet_coeffs)):
        labels[4 * lev * pixel_count:(4 * lev + 1) * pixel_count] = \
            np.array(['cA%d_%d' % (lev, i) for i in range(pixel_count)], 
                     dtype='str')
        for det in range(3):
            start = (4 * lev + det + 1) * pixel_count
            labels[start: start + pixel_count] = \
                ['c%s%d_%d' % (cfs[det], lev, i) for i in range(pixel_count)]
    return labels


def reshape_swt2_coefficients(flat_coeffs, nlev, image_shape):
    """
    Inverse function to restore a flattened array to pywt structure.

    Parameters
    ----------
    flat_coeffs : np.ndarray
        Flattened array of coefficients
    nlev : int
        Number of levels wavelet decomposition was performed with
    image_shape : tuple
        Shape of original images

    Returns
    -------
    list
        pywt compatible coefficient structure
    """

    pixel_count = np.prod(image_shape)

    output = []
    for lev in range(nlev):
        output_lev = []
        start = 4 * lev * pixel_count
        unshaped_coeffs = flat_coeffs[start: start + pixel_count]
        approx_coeffs = unshaped_coeffs.reshape(image_shape)
        output_lev.append(approx_coeffs)
        det_coeffs = []
        for det in range(3):
            start = (4 * lev + det + 1) * pixel_count
            unshaped_coeffs = flat_coeffs[start: start + pixel_count]
            det_coeffs.append(unshaped_coeffs.reshape(image_shape))
        output_lev.append(det_coeffs)
        output.append(output_lev)
    return output


def wavelet_decomposition(img, level=2, wavelet_family='sym2'):
    """
    Perform wavelet decomposition on single image

    Parameters
    ----------
    img : np.ndarray
        Image
    level : int, optional
        Level of depth for the wavelet transform
    wavelet_family : string or pywt.Wavelet object
        Which wavelet family to use

    Returns
    -------
    np.ndarray
        Flattened array of coefficients
    labels
        The labels of the coefficients
    """
    coeffs = pywt.swt2(img, wavelet_family, level=level)
    return coeffs


class WaveletFeatures(PipelineStage):
    def __init__(self, level=2, wavelet_family='sym2', **kwargs):
        """
        Performs a stationary wavelet transform

        Parameters
        ----------
        level : int, optional
            Level of depth for the wavelet transform
        wavelet_family : string or pywt.Wavelet object
            Which wavelet family to use
        """
        super().__init__(level=level, wavelet_family=wavelet_family, **kwargs)

        self.level = level
        self.wavelet_family = wavelet_family

    def _execute_function(self, image):
        """
        Does the work in actually extracting the wavelets

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        pd.DataFrame
            Contains the extracted wavelet features

        """

        # Here I'm explicitly assuming any multi-d images store the colours 
        # in the last dim
        if len(image.shape) == 2:
            # Greyscale-like image

            coeffs = wavelet_decomposition(image, level=self.level, 
                                           wavelet_family=self.wavelet_family)
            flattened_coeffs = flatten_swt2_coefficients(coeffs)
            if len(self.labels) == 0:
                self.labels = generate_labels(coeffs)
            return flattened_coeffs
        else:
            wavs_all_bands = []
            all_labels = []
            for band in range(len(image.shape[2])):
                coeffs = wavelet_decomposition(image, level=self.level, 
                                            wavelet_family=self.wavelet_family)  # noqa E128
                flattened_coeffs = flatten_swt2_coefficients(coeffs)
                wavs_all_bands += list(flattened_coeffs)
                if len(self.labels) == 0:
                    labels = generate_labels(coeffs)
                    all_labels += ['%s_band_%d' % (labels[i], band)
                                   for i in range(labels)]
            if len(self.labels) == 0:
                self.labels = all_labels
            return wavs_all_bands
