import numpy as np
from scipy import ndimage
import xarray

def psd_2d(img, nbins):
    """
    Computes the power spectral density for an input image. Translation and rotation invariate features.

    Parameters
    ----------
    img : np.ndarray
        Input image
    nbins : int
        Number of frequency bins to use. Frequency will range from 1 pixel to the largest axis of the input image,
        measured in pixels.

    Returns
    -------
    np.ndarray
        Power spectral density at each frequency
    """
    the_fft = np.fft.fftshift(np.fft.fft2(img))
    psd = np.abs(the_fft) ** 2

    # Now radially bin the power spectral density
    X, Y = np.meshgrid(np.arange(the_fft.shape[1]), np.arange(the_fft.shape[0]))
    r = np.hypot(X - the_fft.shape[1] // 2, Y - the_fft.shape[0] // 2)
    max_freq = np.min((the_fft.shape[0]//2, the_fft.shape[1]//2))
    rbin = (nbins * r / max_freq).astype(np.int)

    radial_sum = ndimage.sum(psd, labels=rbin, index=np.arange(1, nbins+1))

    return radial_sum


def extract_features_psd2d(pipeline_dict, nbins='auto', input_key='cutouts', output_key='features_psd2d'):
    """
    Performs a 2d FFT and constructs the power spectral density function for each image cutout.

    Parameters
    ----------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    nbins : int, string, optional
        Number of frequency bins to use. Frequency will range from 1 pixel to the largest axis of the input image,
        measured in pixels. If 'auto', nbins will be set to half the size of the smallest axis (maximum possible
        frequency according to Nyquist)
    input_key : str, optional
        The input key of pipeline_dict to run the function on.
    output_key : str, optional
        The output key of pipeline_dict

    Returns
    -------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores

    """
    print('Extracting 2D PSD...')
    cutouts = pipeline_dict[input_key]

    if nbins == 'auto':
        shp = cutouts[0].shape
        nbins = int(min(shp)//2)

    output = []
    for i in range(len(cutouts)):
        output.append(psd_2d(cutouts[i], nbins=nbins))

    pipeline_dict[output_key] = xarray.DataArray(
        output, coords={'id': pipeline_dict['metadata'].id}, dims=['id', 'features'], name=output_key)

    print('Done!')
    return pipeline_dict
