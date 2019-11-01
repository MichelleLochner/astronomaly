import pandas as pd 
import os
from astronomaly.base.base_dataset import Dataset


class LightCurveDataset(Dataset):
    def __init__(self, directory='', list_of_files=[], lower_mag=1, upper_mag=25, **kwargs):
        """
         Read in a set of images either from a directory or from a list of file paths (absolute)

        Parameters
        ----------
        directory : string, optional
            Path to files to read from
        list_of_files : list, optional
            List of files to read in (absolute path)
        fits_index : integer, optional
            If these are fits files, specifies which HDU object in the list to work with
        window_size : int, tuple or list, optional
            The size of the cutout in pixels. If an integer is provided, the cutouts will be square. Otherwise a list of
            [window_size_x, window_size_y] is expected.
        window_shift : int, tuple or list, optional
            The size of the window shift in pixels. If the shift is less than the window size, a sliding window is used to
             create cutouts. This can be particularly useful for (for example) creating a training set for an autoencoder.
             If an integer is provided, the shift will be the same in both directions. Otherwise a list of
            [window_shift_x, window_shift_y] is expected.
        transform_function : function or list, optional
            The transformation function or list of functions that will be applied to each cutout. The function should take
            an input 2d array (the cutout) and return an output 2d array. If a list is provided, each function is applied
            in the order of the list.
        """

        super().__init__(directory=directory, list_of_files=list_of_files, lower_mag=lower_mag, upper_mag=upper_mag,
                         **kwargs)

        self.data_type = 'light_curve'

        self.metadata = pd.DataFrame(data=[])

        ids = [f.split(os.sep)[-1] for f in self.files]
        self.metadata = pd.DataFrame({'filepath': self.files}, index=ids)

        self.lower_mag = lower_mag
        self.upper_mag = upper_mag

    @staticmethod
    def read_lc_from_file(flpath):
        light_curve = pd.read_csv(flpath, delim_whitespace=True)
        return light_curve

    def get_display_data(self, idx):
        # print(id)
        ### Need to extend this to deal with other bands
        time_col = 'MJD'
        mag_col = 'g_mag'
        err_col = 'g_mag_err'

        out_dict = {}

        metadata = self.metadata
        flpath = metadata[idx]['filepath'].iloc[0]
        try:
            light_curve = self.read_lc_from_file(flpath)
            light_curve = light_curve[(self.lower_mag < light_curve[mag_col]) & (light_curve[mag_col] < self.upper_mag)]
            light_curve['err_lower'] = light_curve[mag_col] - light_curve[err_col]
            light_curve['err_upper'] = light_curve[mag_col] + light_curve[err_col]

            out_dict['data'] = light_curve[[time_col, mag_col]].values.tolist()
            out_dict['errors'] = light_curve[[time_col, 'err_lower', 'err_upper']].values.tolist()

        except (pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError) as e:
            print('Error parsing file', flpath)
            print('Error message:')
            print(e)
            out_dict = {'data': [], 'errors': []}

        return out_dict


