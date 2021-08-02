import numpy as np
from scipy import ndimage
from astronomaly.base.base_pipeline import PipelineStage
from astronomaly.preprocessing.image_preprocessing import image_transform_scale


def calculate_flux_histogram(img, nbins, norm=True):
    """
    Histograms the flux values of the pixels into a given number of bins.
    

    Parameters
    ----------
    img : np.ndarray
        Input image
    nbins : int
        Number of bins to use. 
    norm : boolean
        If true, normalises the image first so that histogram will be of values
        from zero to one.

    Returns
    -------
    bins
        x-axis bins for the histogram
    values
        Histogram values in each bin from zero to one
    """
    if norm:
        img = image_transform_scale(img)
    vals, bins = np.histogram(img, bins=nbins, density=True)

    return vals


class FluxHistogramFeatures(PipelineStage):
    def __init__(self, nbins=25, norm=True, **kwargs):
        """
        Simple histogram of flux values.

        Parameters
        ----------
        nbins : int
            Number of bins to use.
        norm : bool
            If true, normalises the image first so that histogram will range 
            from zero to one.
        """

        super().__init__(nbins=nbins, norm=norm, **kwargs)

        self.nbins = nbins
        self.norm = norm

    def _set_labels(self):
        """
        Because the number of features may not be known till runtime, we can
        only create the labels of these features at runtime.
        """

        if self.nbands == 1:
            self.labels = ['hist_%d' % i for i in range(self.nbins)]
        else:
            self.labels = []
            for band in range(self.nbands):
                self.labels += \
                    ['hist_%d_band_%d' % (i, band) for i in range(self.nbins)]

    def _execute_function(self, image):
        """
        Does the work in actually extracting the histogram

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        array
            Contains the extracted flux histogram features
        """

        if len(image.shape) != 2:
            self.nbands = image.shape[2]
        else:
            self.nbands = 1

        if len(self.labels) == 0:
            # Only call this once we know the dimensions of the input data. 
            self._set_labels() 

        if self.nbands == 1:
            # Greyscale-like image
            hist_feats = calculate_flux_histogram(image, nbins=self.nbins)

            return hist_feats

        else:
            hist_all_bands = []
            labels = []
            for band in range(image.shape[2]):
                hist_feats = calculate_flux_histogram(image[:, :, band], 
                                                      nbins=self.nbins,
                                                      norm=self.norm)
                hist_all_bands += list(hist_feats)
                labels += \
                    ['hist_%d_band_%d' % (i, band) for i in range(self.nbins)]

            return hist_all_bands
