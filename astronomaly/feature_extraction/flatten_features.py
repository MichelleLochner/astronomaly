from astronomaly.base.base_pipeline import PipelineStage
import numpy as np


class Flatten_Features(PipelineStage):
    def __init__(self, **kwargs):
        """
        A very simple feature extraction that ravels an input image to reduce
        it to a 1d vector. This can be useful for simple test datasets like
        MNIST or to flatten images that are already aligned in some way to then
        use PCA on.
        """
        super().__init__(**kwargs)
        self.labels = None

    def _set_labels(self, image):
        """
        Because the number of features may not be known till runtime, we can
        only create the labels of these features at runtime.
        """
        n = np.prod(image.shape)
        self.labels = np.array(np.arange(n), dtype='str')

    def _execute_function(self, image):
        """
        Does the work in flattening the image

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        Array
            Flattened image
        """
        feats = image[:, :, 0].ravel()

        if self.labels is None:
            self._set_labels(image[:, :, 0])

        return feats
