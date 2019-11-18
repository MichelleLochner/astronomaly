import pandas as pd 
import os
from astronomaly.base.base_dataset import Dataset


class LightCurveDataset(Dataset):
    def __init__(self, lower_mag=1, upper_mag=25, **kwargs):
        """
        Reads in light curve data from file(s).

        Parameters
        ----------
        filename : str
            If a single file (of any time) is to be read from, the path can be
            given using this kwarg. 
        directory : str
            A directory can be given instead of an explicit list of files. The
            child class will load all appropriate files in this directory.
        list_of_files : list
            Instead of the above, a list of files to be loaded can be
            explicitly given.
        output_dir : str
            The directory to save the log file and all outputs to. Defaults to
            './' 
        lower_mag : float, optional
            Applies a cut to the data, excludes everything above this, by 
            default 1
        upper_mag : int, optional
            Applies a cut to the data, excludes everything below this, by 
            default 25
        """

        super().__init__(lower_mag=lower_mag, upper_mag=upper_mag, **kwargs)

        self.data_type = 'light_curve'

        self.metadata = pd.DataFrame(data=[])

        ids = [f.split(os.sep)[-1] for f in self.files]
        self.metadata = pd.DataFrame({'filepath': self.files}, index=ids)

        self.lower_mag = lower_mag
        self.upper_mag = upper_mag

    @staticmethod
    def read_lc_from_file(flpath):
        """
        Reads the light curve from file returning a dataframe
        """
        light_curve = pd.read_csv(flpath, delim_whitespace=True)
        return light_curve

    def get_display_data(self, idx):
        """
        Returns a single instance of the dataset in a form that is ready to be
        displayed by the web front end.

        Parameters
        ----------
        idx : str
            Index (should be a string to avoid ambiguity)

        Returns
        -------
        dict
            json-compatible dictionary of the light curve data
        """
        # print(id)
        # ***** Need to extend this to deal with other bands
        time_col = 'MJD'
        mag_col = 'g_mag'
        err_col = 'g_mag_err'

        out_dict = {}

        metadata = self.metadata
        flpath = metadata[idx]['filepath'].iloc[0]
        try:
            light_curve = self.read_lc_from_file(flpath)
            light_curve = light_curve[
                (self.lower_mag < light_curve[mag_col]) & 
                (light_curve[mag_col] < self.upper_mag)]
            light_curve['err_lower'] = light_curve[mag_col] - \
                light_curve[err_col]
            light_curve['err_upper'] = light_curve[mag_col] + \
                light_curve[err_col]

            out_dict['data'] = light_curve[[time_col, mag_col]].values.tolist()
            lc_errs = light_curve[[time_col, 'err_lower', 'err_upper']]
            out_dict['errors'] = lc_errs.values.tolist()

        except (pd.errors.ParserError, pd.errors.EmptyDataError, 
                FileNotFoundError) as e:
            print('Error parsing file', flpath)
            print('Error message:')
            print(e)
            out_dict = {'data': [], 'errors': []}

        return out_dict
