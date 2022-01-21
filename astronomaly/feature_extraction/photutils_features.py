from astronomaly.base.base_pipeline import PipelineStage
import numpy as np
from photutils import morphology


class PhotutilsFeatures(PipelineStage):
    def __init__(self, columns, **kwargs):
        """
        Uses the photutils package to extract requested properties from the
        image. The list of available photutil properties is here: 
        https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html#photutils.segmentation.SourceCatalog
        Properties that are returned as arrays will automatically be flattened
        and each element will be treated as an independent feature.
        """
        super().__init__(columns=columns, **kwargs)
        self.columns = columns
        self.labels = None

    def _set_labels(self, labels):
        """
        Because the number of features may not be known till runtime, we can
        only create the labels of these features at runtime.
        """
        self.labels = np.array(labels, dtype='str')

    def _execute_function(self, image):
        """
        Does the work in extracting the requested properties using photutils.

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        Array
            Features
        """
        if np.prod(image.shape) > 2:
            image = image[0]
        feats = []
        labels = []
        cat = morphology.data_properties(image)
        for c in self.columns:
            prop = getattr(cat, c)
            prop = np.array(prop)
            prop = prop.flatten()
            if len(prop) == 1:
                feats.append(prop[0])
                labels.append(c)
            else:
                feats += prop.tolist()
                for i in range(len(prop)):
                    labels.append(c + str(i))

        if self.labels is None:
            self._set_labels(labels)

        return np.array(feats)
