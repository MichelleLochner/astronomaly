import pandas as pd
import numpy as np
from astronomaly.base.base_dataset import Dataset


class LightCurveDataset(Dataset):
    def __init__(self, data_dict, header_nrows=1, delim_whitespace=False,
                 **kwargs):
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
        data_dict: Dictionary
                It a dictionary with index of the column names corresponding to
                the following specific keys:
                ('id','time','mag','mag_err','flux','flux_err','filters')
                e.g {'time':1,'mag':2}, where 1 and 2 are column index
                correpoding to 'time' and 'mag' in the input data.
                If the data does not have unique ids, the user can neglect the
                'id' key, and the ids will be the file path by default.
                The user can also provide a list of indices for the 'mag' and
                'flux' columns.
                This is the case where the brightness is recorded in more than
                one column. e.g {'time':1,'mag':[2,3]} 2 and 3 corresponds to
                columns with brightness records
        header_nrows: int
                The number of rows the header covers in the dataset, by
                default 1
        delim_whitespace: bool
                Should be True if the data is not separated by a comma, by
                default False"""

        super().__init__(data_dict=data_dict, header_nrows=header_nrows,
                         delim_whitespace=delim_whitespace, **kwargs)

        self.data_type = 'light_curve'
        self.metadata = pd.DataFrame(data=[])
        self.data_dict = data_dict
        self.header_nrows = header_nrows
        self.delim_whitespace = delim_whitespace

    #         ================================================================
    #                         Reading the light curve data
    #         ================================================================

        # The case where there is one file
        data = pd.read_csv(self.files[0], skiprows=self.header_nrows,
                           delim_whitespace=self.delim_whitespace, header=None)

        # The case for multiple files of light curve data
        file_len = [len(data)]
        if len(self.files) > 1:

            file_paths = [self.files[0]]
            for fl in range(1, len(self.files)):
                data = pd.concat([data, pd.read_csv(self.files[fl],
                                 skiprows=self.header_nrows,
                                 delim_whitespace=self.delim_whitespace,
                                 header=None)])

                file_paths.append(self.files[fl])
                file_len.append(len(data))

            IDs = [file_paths[0] for i in range(file_len[0])]
            for fl in range(1, len(file_len)):

                for f in range(file_len[fl] - file_len[fl-1]):

                    IDs.append(file_paths[fl])

    #         =================================================================
    #            Renaming the columns into standard columns for astronomaly
    #         =================================================================
        time = data.iloc[:, self.data_dict['time']]
        standard_data = {'time': time}

        if 'id' in data_dict.keys():
            idx = data.iloc[:, self.data_dict['id']]
            ids = np.unique(idx)
            ids = np.array(ids, dtype='str')
            self.index = ids[:100]  # Testing for 100 objects
            self.metadata = pd.DataFrame({'ID': ids}, index=ids)
            standard_data.update({'ID': np.array(idx, dtype='str')})

        else:
            idx = self.files
            self.index = idx
            self.metadata = pd.DataFrame({'ID': idx}, index=idx)
            standard_data.update({'ID': IDs})

        # Possible brightness columns
        brightness_cols = ['mag', 'flux']

        # WE NEED TO CONVERT FLUX TO MAG FOR FEETS FEATURE EXTRACTOR
        # Looping through the brightness columns
        for col in range(len(brightness_cols)):
            data_col = brightness_cols[col]
            if data_col in self.data_dict.keys():

                # ============Multiple brightness columns======================
                try:

                    for i in range(len(self.data_dict[data_col])):

                        # The case where there are no error columns
                        standard_data.update({data_col+str(i+1):
                                             data.iloc[:, self.data_dict[
                                                          data_col][i]]})

                        # The case where there are brightness error columns
                        if data_col+'_err' in self.data_dict.keys():

                            # Updating the standard dictionary to include the
                            # brightness_errors
                            standard_data.update({data_col+'_error'+str(i+1):
                                                  data.iloc[:, self.data_dict[
                                                   data_col+'_err'][i]]})

                #  =================Single brightness Column===================
                #  ============================================================
                except TypeError:

                    # The case for single brightness column and no errors
                    standard_data.update({data_col:
                                          data.iloc[:, self.data_dict[
                                                       data_col]]})

                    if data_col+'_err' in self.data_dict.keys():

                        standard_data.update({data_col+'_error':
                                              data.iloc[:, self.data_dict[
                                                           data_col+'_err']]})

                # ============The case where there are filters in the data=====
                if 'filters' in self.data_dict.keys():

                    standard_data.update({'filters': data.iloc[
                                        :, self.data_dict['filters']]})

        lc = pd.DataFrame.from_dict(standard_data)

        if 'flux' in lc.columns:
            # Discard all the negative flux values
            # since they are due noice or are for
            # faint observations
            lc = lc[lc['flux'].values > 0]

            for i in range(len(np.unique(lc['filters']))):

                f_zero = [24.63, 25.11, 24.80, 24.36, 22.83, 23]  # look_up y
                # Filters
                filt_val = (lc['filters'] == i)
                f_obs = lc.flux[filt_val].values
                f_obs_err = lc.flux_error[filt_val].values
                # converting
                flux_convs = -2.5*np.log10(f_obs/f_zero[i])
                err_convs = -2.5*np.log10(f_obs_err/f_zero[i])
                lc.loc[filt_val, 'flux'] = flux_convs
                lc.loc[filt_val, 'flux_error'] = err_convs

            conv_lc_cols = {'flux': 'mag', 'flux_error': 'mag_error'}
            self.light_curves_data = lc.rename(columns=conv_lc_cols,
                                               inplace=False)

        else:
            self.light_curves_data = lc

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

        # WE NEED TO EXPAND THIS TO BE MORE GENERAL

        # All the standard columns are included here
        data_col = ['mag']
        err_col = ['mag_error']
        out_dict = {}

        # Reading in the light curve data
        light_curve = self.light_curves_data[
                      self.light_curves_data['ID'] == idx]

        # Data and error index

        lc_cols = light_curve.columns.values.tolist()
        mag_indx = [cl for cl in data_col if cl in lc_cols]
        err_indx = [cl for cl in err_col if cl in lc_cols]

        # Returns true if we have error columns
        # if err_col[0] in light_curve.columns.values.tolist() or err_col[1] in
        #  light_curve.columns.values.tolist():

        light_curve['err_lower'] = light_curve[mag_indx].values - \
            light_curve[err_indx].values

        light_curve['err_upper'] = light_curve[mag_indx].values + \
            light_curve[err_indx].values

        lc_errs = light_curve[['time', 'err_lower', 'err_upper']]

        # inserting the time column to data and adding 'data'
        # and 'errors' to out_dict
        mag_indx.insert(0, 'time')
        out_dict['data'] = light_curve[mag_indx].values.tolist()
        out_dict['errors'] = lc_errs.values.tolist()

        return out_dict

    def get_sample(self, idx):

        # Choosing light curve values for a specific ID
        light_curve_sample = self.light_curves_data[
                             self.light_curves_data['ID'] == idx]

        return light_curve_sample
