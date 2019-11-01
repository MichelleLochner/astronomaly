import pandas as pd
import xarray
from astronomaly.base.base_pipeline import PipelineStage

def read_from_file(pipeline_dict, file_location, output_key='features_from_file', drop_na=True):
    print('Reading in features from file ',file_location,'...')
    df = pd.read_csv(file_location, delim_whitespace=True)
    if drop_na:
        df = df.dropna()
    ids = df[df.columns[0]]
    xr=xarray.DataArray(df[df.columns[1:]], 
        coords={'id': ids, 'features':df.columns[1:]}, dims=['id', 'features'])

    pipeline_dict[output_key] = xr
    print('Done!')

    return pipeline_dict

class RawFeatures(PipelineStage):
    def __init__(self, filename='', **kwargs):
        """
        Lightweight function that simply returns the original data as features.
        """
        super().__init__(filename=filename, **kwargs)

        self.filename = filename

    def _execute_function(self, data):
        """
        Does the work in actually extracting the PSD

        Parameters
        ----------
        idx : str
            Index of the sample the feature extractor is run on. This is to correctly store the index with the output
            features.
        image : np.ndarray
            Input image

        Returns
        -------
        pd.DataFrame
            Contains the extracted PSD features

        """
        if self.nbins == 'auto':
            shp = image.shape[:2] # Here I'm explicitly assuming any multi-d images store the colours in the last dim
            self.nbins = int(min(shp) // 2)

        if len(image.shape) == 2:
            # Greyscale-like image

            psd_feats = psd_2d(image, self.nbins)

            return pd.DataFrame(data=[psd_feats], index=[idx], columns=['psd_%d' %i for i in range(self.nbins)])
        else:
            psd_all_bands = []
            labels = []
            for band in range(len(image.shape[2])):
                psd_feats = psd_2d(image[:,:,band], self.nbins)
                psd_all_bands += list(psd_feats)
                labels += ['psd_%d_band_%d' %(i,band) for i in range(self.nbins)]
            return pd.DataFrame(data=psd_all_bands, index=[idx], columns=labels)