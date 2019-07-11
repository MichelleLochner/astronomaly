import pywt
import numpy as np
import xarray

def flatten_swt2_coefficients(wavelet_coeffs):
    """
    A standardised way of flattening the swt2d coefficients
    They are stored as n_levels -> (cA, (cH, cV, cD)) where each of the sets of coeffcients has a list of
    [npixels, npixels]

    Parameters
    ----------
    wavelet_coeffs : list
        Exactly as output by pywt

    Returns
    -------
    np.ndarray
        Flattened coefficients

    """

    pixel_count = np.prod(wavelet_coeffs[0][0].shape)
    total_len = len(wavelet_coeffs) * 4 * pixel_count

    output_array = np.zeros(total_len)
    for lev in range(len(wavelet_coeffs)):
        approx_coeffs = wavelet_coeffs[lev][0]
        output_array[4 * lev * pixel_count:(4 * lev + 1) * pixel_count] = approx_coeffs.reshape(pixel_count)
        for det in range(3):
            detailed_coeffs = wavelet_coeffs[lev][1][det]
            output_array[(4 * lev + det + 1) * pixel_count:(4 * lev + det + 2) * pixel_count] = detailed_coeffs.reshape(
                pixel_count)
    return output_array


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
        approx_coeffs = flat_coeffs[4 * lev * pixel_count:(4 * lev + 1) * pixel_count].reshape(image_shape)
        output_lev.append(approx_coeffs)
        det_coeffs = []
        for det in range(3):
            det_coeffs.append(flat_coeffs[(4 * lev + det + 1) * pixel_count:(4 * lev + det + 2) * pixel_count].reshape(image_shape))
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
    """
    coeffs = pywt.swt2(img, wavelet_family, level=level)
    flattened_coeffs = flatten_swt2_coefficients(coeffs)
    return flattened_coeffs


def extract_features_wavelets(pipeline_dict, level=2, wavelet_family='sym2', input_key='cutouts',
                              output_key='features_wavelets'):
    """
    Performs a 2d stationary wavelet transform on the image cutouts in the pipeline_dict.

    Parameters
    ----------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    column_name : string, optional
        The name of the column in the dataframe containing the cutouts (defaults to the same name image_preprocessing.py uses)
    level : int, optional
        Level of depth for the wavelet transform
    wavelet_family : string or pywt.Wavelet object
        Which wavelet family to use
    input_key : str, optional
        The input key of pipeline_dict to run the function on.
    output_key : str, optional
        The output key of pipeline_dict

    Returns
    -------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores

    """
    print('Extracting wavelets...')
    cutouts = pipeline_dict[input_key]
    output = []

    for i in range(len(cutouts)):
        output.append(wavelet_decomposition(cutouts[i], level=level, wavelet_family=wavelet_family))

    pipeline_dict[output_key] = xarray.DataArray(
        output, coords={'id': pipeline_dict['metadata'].id}, dims=['id', 'features'], name=output_key)

    print('Done!')
    return pipeline_dict