import numpy as np
from scipy import ndimage
from astronomaly.base.base_pipeline import PipelineStage


def psd_2d(img, nbins):
    """
    Computes the power spectral density for an input image. Translation and 
    rotation invariate features.

    Parameters
    ----------
    img : np.ndarray
        Input image
    nbins : int
        Number of frequency bins to use. Frequency will range from 1 pixel to 
        the largest axis of the input image, measured in pixels.

    Returns
    -------
    np.ndarray
        Power spectral density at each frequency
    """
    the_fft = np.fft.fftshift(np.fft.fft2(img))
    psd = np.abs(the_fft) ** 2

    # Now radially bin the power spectral density
    X, Y = np.meshgrid(np.arange(the_fft.shape[1]), 
                       np.arange(the_fft.shape[0]))
    r = np.hypot(X - the_fft.shape[1] // 2, Y - the_fft.shape[0] // 2)
    max_freq = np.min((the_fft.shape[0] // 2, the_fft.shape[1] // 2))
    rbin = (nbins * r / max_freq).astype(np.int)

    radial_sum = ndimage.sum(psd, labels=rbin, index=np.arange(1, nbins + 1))

    return radial_sum


class PSD_Features(PipelineStage):
    def __init__(self, nbins='auto', **kwargs):
        """
        Computes the power spectral density for an input image. Translation and 
        rotation invariate features.

        Parameters
        ----------
        nbins : int, str
            Number of frequency bins to use. Frequency will range from 1 pixel
            to the largest axis of the input image, measured in pixels. If set 
            to 'auto' will use the Nyquist theorem to automatically calculate 
            the appropriate number of bins at runtime.
        """
        super().__init__(nbins=nbins, **kwargs)

        self.nbins = nbins

    def _set_labels(self):

        if self.nbands == 1:
            self.labels = ['psd_%d' % i for i in range(self.nbins)]
        else:
            self.labels = []
            for band in range(self.nbands):
                self.labels += \
                    ['psd_%d_band_%d' % (i, band) for i in range(self.nbins)]

    def _execute_function(self, image):
        """
            Does the work in actually extracting the PSD

            Parameters
            ----------
            image : np.ndarray
                Input image

            Returns
            -------
            pd.DataFrame
                Contains the extracted PSD features

            """
        if self.nbins == 'auto':
            # Here I'm explicitly assuming any multi-d images store the 
            # colours in the last dim
            shp = image.shape[:2] 
            self.nbins = int(min(shp) // 2)

        if len(image.shape) != 2:
            self.nbands = image.shape[2]
        else:
            self.nbands = 1

        if len(self.labels) == 0:
            # Only call this once we know the dimensions of the input data. 
            # *****Needs to be more robust!
            self._set_labels() 

        if self.nbands == 1:
            # Greyscale-like image
            psd_feats = psd_2d(image, self.nbins)

            return psd_feats

        else:
            psd_all_bands = []
            labels = []
            for band in range(len(image.shape[2])):
                psd_feats = psd_2d(image[:, :, band], self.nbins)
                psd_all_bands += list(psd_feats)
                labels += \
                    ['psd_%d_band_%d' % (i, band) for i in range(self.nbins)]

            return psd_all_bands
