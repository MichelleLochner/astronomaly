import pandas as pd
import numpy as np
from astronomaly.base.base_dataset import Dataset

# ignores the false positve pandas warning
# for the following kind of code
# df['key'] == item, for an existing key in a df
pd.options.mode.chained_assignment = None


def split_lc(lc_data, max_gap):
    '''Splits the light curves into smaller chunks based on their gaps

    Parameters
    ----------
        lc_data: Dataframe with the light curves
        max_gap: Maximum gap between observations'''

    unq_ids = np.unique(lc_data.ID)
    unq_ids = unq_ids
    splitted_dict = {}

    for ids in unq_ids:

        lc = lc_data[lc_data['ID'] == ids]
        if 'filters' in lc.columns:

            unq_filters = np.unique(lc.filters)

            for filtr in unq_filters:

                lc1 = lc[lc['filters'] == filtr]

                time = lc1.time
                time_diff = [time.iloc[i] - time.iloc[i-1]
                             for i in range(1, len(time))]
                time_diff.insert(0, 0)
                lc1['time_diff'] = time_diff
                gap_idx = np.where(lc1.time_diff > max_gap)[0]

                # Separating the lc as by the gap index
                try:

                    lc0 = lc1.iloc[:gap_idx[0]]
                    lc0['ID'] = [ids+'_0' for i in range(len(lc0.time))]

                    splitted_dict.update({'lc'+ids+'_'+str(filtr)+str(0): lc0})

                    for k in range(1, len(gap_idx)):

                        lcn = lc1.iloc[gap_idx[k-1]:gap_idx[k]]
                        lcn['ID'] = [ids+'_'+str(k)
                                     for i in range(len(lcn.time))]

                        splitted_dict.update({'lc'+ids+'_'+str(filtr)+str(k):
                                             lcn})

                    lc2 = lc1.iloc[gap_idx[k]:]
                    lc2['ID'] = [ids+'_'+str(k+1)
                                 for i in range(len(lc2.time))]

                    splitted_dict.update({'lc'+ids+'_'+str(filtr)+str(k+1):
                                         lc2})

                except (IndexError, UnboundLocalError):
                    pass

    final_data = pd.concat(splitted_dict.values(), ignore_index=False)
    return final_data


def convert_flux_to_mag(lcs, f_zero):
    '''Converts flux to mags for a given light curve data

    Parameters
    ----------
        lcs: DataFrame with the light curve values
        zeropoint: Zeropoint magnitude
    '''

    # Discard all the negative flux values
    # since they are due noice or are for
    # faint observations
    lc = lcs[lcs['flux'].values > 0]
    lc = lc[lc['flux_error'].values > 0]

    # Flux and flux error
    f_obs = lc.flux.values
    f_obs_err = lc.flux_error.values
    constants = (2.5/np.log(10))
    # converting
    flux_convs = f_zero - 2.5*np.log10(f_obs)
    err_convs = constants*(f_obs_err/f_obs)
    # Adding the new mag and mag_error column
    lc['mag'] = flux_convs
    lc['mag_error'] = err_convs
    lc = lc[lc['mag_error'].values < 2]

    return lc


class LightCurveDataset(Dataset):
    def __init__(self, data_dict, f_zero=22, header_nrows=1,
                 delim_whitespace=False, max_gap=50, **kwargs):
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
        f_zero : float/int
                The  zero flux magnitude values, by default 22
        max_gap: int
                Maximum gap between consecute observations, default 50
        delim_whitespace: bool
                Should be True if the data is not separated by a comma, by
                default False"""

        super().__init__(data_dict=data_dict, header_nrows=header_nrows,
                         delim_whitespace=delim_whitespace, f_zero=f_zero,
                         max_gap=max_gap, **kwargs)

        self.data_type = 'light_curve'
        self.metadata = pd.DataFrame(data=[])
        self.data_dict = data_dict
        self.header_nrows = header_nrows
        self.delim_whitespace = delim_whitespace
        self.f_zero = f_zero
        self.max_gap = max_gap

    #         ================================================================
    #                         Reading the light curve data
    #         ================================================================

        # The case where there is one file
        data = pd.read_csv(self.files[0], skiprows=self.header_nrows,
                           delim_whitespace=self.delim_whitespace, header=None)

        # Spliting the light curve data using the gaps

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
#             self.index = ids[:5]  # Testing for 100 objects
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

            # Convert flux to mag
            lc = convert_flux_to_mag(lc, self.f_zero)
            # Split the light curve into chunks
            lc = split_lc(lc, self.max_gap)
            self.light_curves_data = lc

        else:
            self.light_curves_data = lc

        ids = np.unique(lc.ID)
        self.index = ids
        self.metadata = pd.DataFrame({'ID': ids}, index=ids)

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
