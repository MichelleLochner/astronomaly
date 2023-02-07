import pandas as pd
import numpy as np
from astronomaly.base.base_dataset import Dataset
import os

# ignores the false positve pandas warning
# for the following kind of code
# df['key'] == item, for an existing key in a df
pd.options.mode.chained_assignment = None


def split_lc(lc_data, max_gap):
    """
    Splits the light curves into smaller chunks based on their gaps. This is 
    useful for long light curves that span many observing seasons so have 
    large gaps that can sometimes interfere with feature extraction.

    Parameters
    ----------
    lc_data : pd.Dataframe 
        Light curves
    max_gap : int
        Maximum gap between observations

    Returns
    -------
    pd.DataFrame
        Split light curves
    """

    unq_ids = np.unique(lc_data.ID)
    unq_ids = unq_ids
    splitted_dict = {}

    id_n = 0
    for idx in unq_ids:
        id_n += 1
        ids = (str)(idx)  # Used for renaming things
        progress = id_n / len(unq_ids)
        progress = progress * 100

        lc = lc_data[lc_data['ID'] == ids]

        if 'filters' in lc.columns:
            unq_filters = np.unique(lc.filters)

            for filtr in unq_filters:

                lc1 = lc[lc['filters'] == filtr]

                tm = lc1.time
                time_diff = [tm.iloc[i] - tm.iloc[i - 1]
                             for i in range(1, len(tm))]
                time_diff.insert(0, 0)
                lc1['time_diff'] = time_diff
                gap_idx = np.where(lc1.time_diff > max_gap)[0]

                # Separating the lc as by the gap index
                try:
                    lc0 = lc1.iloc[:gap_idx[0]]
                    lc0['ID'] = [str(ids) + '_0' for i in range(len(lc0.time))]
                    # Create a new index for the first of split light curves
                    key = 'lc' + ids + '_' + str(filtr) + str(0)
                    splitted_dict.update({key: lc0})

                    for k in range(1, len(gap_idx)):
                        lcn = lc1.iloc[gap_idx[k - 1]:gap_idx[k]]
                        lcn['ID'] = [str(ids) + '_' + str(k)
                                     for i in range(len(lcn.time))]
                        key = 'lc' + ids + '_' + str(filtr) + str(k)
                        splitted_dict.update({key: lcn})

                    lc2 = lc1.iloc[gap_idx[k]:]
                    lc2['ID'] = [ids + '_' + str(k + 1)
                                 for i in range(len(lc2.time))]

                    key = 'lc' + ids + '_' + str(filtr) + str(k + 1)
                    splitted_dict.update({key: lc2})

                except (IndexError, UnboundLocalError):
                    pass

    final_data = pd.concat(splitted_dict.values(), ignore_index=False)
    return final_data


def convert_flux_to_mag(lcs, mag_ref):
    """
    Converts flux to mags for a given light curve data.

    Parameters
    ----------
    lcs: pd.DataFrame 
        Light curve
    mag_ref: float
        Reference magnitude
    """

    # Discard all the negative flux values
    # since they are due to noise or are for
    # faint observations
    # Replacing the negative flux values with their respective errors
    neg_flux_indx = np.where(lcs['flux'].values < 0)
    lcs.loc[lcs['flux'] < 0, ['flux']] = lcs['flux_error'].iloc[neg_flux_indx]
    lc = lcs

    # Flux and flux error
    f_obs = lc.flux.values
    f_obs_err = lc.flux_error.values
    constant = (-2.5 / np.log(10))
    # converting
    flux_convs = mag_ref - 2.5 * np.log10(f_obs)
    err_convs = np.abs(constant * (f_obs_err / f_obs))
    # Adding the new mag and mag_error column
    lc['mag'] = flux_convs
    lc['mag_error'] = err_convs

    return lc


class LightCurveDataset(Dataset):
    def __init__(self, data_dict, header_nrows=1,
                 delim_whitespace=False, max_gap=50, plot_errors=True,
                 convert_flux=False, mag_ref=22,
                 split_lightcurves=False,
                 filter_colors=['#9467bd', '#1f77b4', '#2ca02c', '#d62728',
                                '#ff7f0e', '#8c564b'],
                 filter_labels=[],
                 which_filters=[],
                 plot_column='flux',
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
            Dictionary with index of the column names corresponding to
            the following specific keys:
            ('id','time','mag','mag_err','flux','flux_err','filters',
            'labels')
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
        convert_flux : bool
            If true converts flux to magnitudes
        mag_ref : float/int
            The  reference magnitude for conversion, by default 22. Used to
            convert flux to magnitude if required
        split_lightcurves : bool
            If true, splits up light curves that have large gaps due to
            multiple observing seasons
        max_gap: int
            Maximum gap between consecutive observations, default 50
        delim_whitespace: bool
            Should be True if the data is not separated by a comma, by
            default False
        plot_errors: bool
            If errors are available for the data, this boolean allows them
            to be plotted
        filter_colors: list
            Allows the user to define their own colours (using hex codes)
            for the different filter bands. Will revert to default
            behaviour of the JavaScript chart if the list of colors
            provided is shorter than the number of unique filters.
        filter_labels: list
            For multiband data, labels will be passed to the frontend
            allowing easy identification of different bands in the light
            curve. Assumes the filters are identified by an integer in the
            data such that the first filter (e.g. filter 0) will correspond
            to the first label provided. For example, to plot PLAsTiCC
            data, provide filter_labels=['u','g','r','i','z','y']
        which_filters: list
            Allows the user to select specific filters (thereby dropping
            others). The list of filters to be included must be numeric and
            integer. For example, to select the griz bands only, set
            which_filters = [1, 2, 3, 4]
        plot_column: string
            Indicates which column to plot. Usually data will have either a
            flux or a mag column. The code will automatically detect which is
            available but if both are available, it will use this kwarg to
            select which to use. The corresponding errors are also used (if
            requested)
        """

        super().__init__(data_dict=data_dict, header_nrows=header_nrows,
                         delim_whitespace=delim_whitespace, mag_ref=mag_ref,
                         max_gap=max_gap, plot_errors=plot_errors,
                         filter_labels=filter_labels,
                         which_filters=which_filters,
                         convert_flux=convert_flux,
                         split_lightcurves=split_lightcurves,
                         filter_colors=filter_colors, 
                         plot_column=plot_column, **kwargs)

        self.data_type = 'light_curve'
        self.metadata = pd.DataFrame(data=[])
        self.data_dict = data_dict
        self.header_nrows = header_nrows
        self.delim_whitespace = delim_whitespace
        self.max_gap = max_gap
        self.plot_errors = plot_errors
        self.filter_labels = filter_labels
        self.filter_colors = filter_colors
        self.convert_flux = convert_flux
        self.plot_column = plot_column

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
                this_data = pd.read_csv(
                    self.files[fl],
                    skiprows=self.header_nrows,
                    delim_whitespace=self.delim_whitespace,
                    header=None)
                data = pd.concat([data, this_data])

                file_paths.append(self.files[fl])
                file_len.append(len(this_data))

            IDs = []
            for fl in range(0, len(file_len)):
                for f in range(file_len[fl]):
                    IDs.append(file_paths[fl].split(os.path.sep)[-1])

#         =================================================================
#            Renaming the columns into standard columns for astronomaly
#         =================================================================
        time = data.iloc[:, self.data_dict['time']]
        standard_data = {'time': time}

        if 'id' in data_dict.keys():
            idx = data.iloc[:, self.data_dict['id']]
            ids = np.unique(idx)
            ids = np.array(ids, dtype='str')
            standard_data.update({'ID': np.array(idx, dtype='str')})

        else:
            idx = IDs
            self.index = idx
            self.metadata = pd.DataFrame({'ID': idx}, index=idx)
            standard_data.update({'ID': IDs})

        if 'labels' in data_dict.keys():
            labels = data.iloc[:, self.data_dict['labels']]
            standard_data.update({'labels': labels})

        # Possible brightness columns
        brightness_cols = ['mag', 'flux']

        # Looping through the brightness columns
        for col in range(len(brightness_cols)):
            data_col = brightness_cols[col]
            if data_col in self.data_dict.keys():

                # ============Multiple brightness columns======================
                try:
                    # We'll have to duplicate these if there are multiple
                    # flux/mag columns
                    id_values = standard_data['ID']
                    time_values = standard_data['time']
                    if 'labels' in standard_data.keys():
                        label_values = standard_data['labels']
                    else:
                        label_values = []

                    for i in range(len(self.data_dict[data_col])):
                        curr_band = data.iloc[:, self.data_dict[data_col][i]]

                        # We have to skip the first time to avoid double
                        # counting
                        if i != 0:
                            standard_data['ID'] = np.concatenate(
                                (standard_data['ID'], id_values))
                            standard_data['time'] = np.concatenate(
                                (standard_data['time'], time_values))
                            if 'labels' in standard_data.keys():
                                standard_data['labels'] = np.concatenate(
                                    (standard_data['labels'], label_values))
                        filter_values = np.array([i] * len(curr_band))
                        if 'filters' not in standard_data.keys():
                            standard_data['filters'] = filter_values
                        else:
                            standard_data['filters'] = np.concatenate(
                                (standard_data['filters'], filter_values))
                        if data_col not in standard_data.keys():
                            standard_data[data_col] = curr_band
                        else:
                            standard_data[data_col] = np.concatenate(
                                (standard_data[data_col], curr_band))

                        # The case where there are brightness error columns
                        if data_col + '_err' in self.data_dict.keys():
                            # Updating the standard dictionary to include the
                            # brightness_errors
                            err_col_name = data_col + '_err'
                            err_col = self.data_dict[err_col_name][i]
                            band_err = data.iloc[:, err_col]
                            if err_col_name not in standard_data.keys():
                                standard_data[err_col_name] = band_err
                            else:
                                standard_data[err_col_name] = np.concatenate(
                                    (standard_data[err_col_name], band_err))

                #  =================Single brightness Column===================
                #  ============================================================
                except TypeError:
                    # The case for single brightness column and no errors
                    val = data.iloc[:, self.data_dict[data_col]]
                    standard_data.update({data_col: val})

                    if data_col + '_err' in self.data_dict.keys():
                        key = data_col + '_error'
                        val = data.iloc[:, self.data_dict[data_col + '_err']]
                        standard_data.update({key: val})

                # ============The case where there are filters in the data=====
                if 'filters' in self.data_dict.keys():
                    val = data.iloc[:, self.data_dict['filters']]
                    standard_data.update({'filters': val})

        lc = pd.DataFrame.from_dict(standard_data)
        if len(which_filters) > 0 and 'filters' in lc.columns:
            # Drop filters if requested
            lc = lc.loc[np.in1d(lc['filters'], which_filters)]

        if 'flux' in lc.columns:
            # Convert flux to mag
            if convert_flux is True:
                lc = convert_flux_to_mag(lc, mag_ref)

        if split_lightcurves:
            # Split the light curve into chunks
            lc = split_lc(lc, self.max_gap)

        self.light_curves_data = lc

        ids = np.unique(lc.ID)
        self.index = ids

        # Add the classes to the metadata
        if 'labels' in lc.columns:
            lc1 = lc.copy()
            lc1 = lc.drop_duplicates(subset='ID')
            labels = [lc1[lc1['ID'] == i]['labels'].values[0] for i in ids]
            self.metadata = pd.DataFrame({'label': labels, 'ID': ids},
                                         index=ids)

        # Metadata without the class
        else:
            self.metadata = pd.DataFrame({'ID': ids}, index=ids)
        print('%d light curves loaded successfully' % len(self.index))

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

        # Reading in the light curve data
        light_curve_original = self.light_curves_data[
            self.light_curves_data['ID'] == idx]
        lc_cols = light_curve_original.columns.values.tolist()

        # Make a decision about what to plot based on what columns are 
        # available and what column is requested 
        if 'flux' in lc_cols and 'mag' in lc_cols:
            data_col = [self.plot_column]
            err_col = [self.plot_column + '_error']
        elif 'mag' in lc_cols:
            data_col = ['mag']
            err_col = ['mag_error']
        else:
            data_col = ['flux']
            err_col = ['flux_error']

        out_dict = {'plot_data_type': data_col, 
                    'data': [], 'errors': [], 'filter_labels': [],
                    'filter_colors': []}

        if err_col[0] in lc_cols and self.plot_errors:
            plot_errors = True
        else:
            plot_errors = False

        if 'filters' in lc_cols:
            multiband = True
            unique_filters = np.unique(light_curve_original['filters'])
        else:
            multiband = False
            unique_filters = [0]

        k = 0
        for filt in unique_filters:
            if multiband:
                msk = light_curve_original['filters'] == filt
                light_curve = light_curve_original[msk]
            else:
                light_curve = light_curve_original

            mag_indx = [cl for cl in data_col if cl in lc_cols]
            err_indx = [cl for cl in err_col if cl in lc_cols]

            if plot_errors:
                light_curve['err_lower'] = light_curve[mag_indx].values - \
                    light_curve[err_indx].values

                light_curve['err_upper'] = light_curve[mag_indx].values + \
                    light_curve[err_indx].values

                lc_errs = light_curve[['time', 'err_lower', 'err_upper']]

                err = lc_errs.values.tolist()

            # inserting the time column to data and adding 'data'
            # and 'errors' to out_dict
            mag_indx.insert(0, 'time')
            dat = light_curve[mag_indx].values.tolist()

            out_dict['data'].append(dat)

            if plot_errors:
                out_dict['errors'].append(err)
            else:
                out_dict['errors'].append([])

            if len(self.filter_labels) >= len(unique_filters):
                out_dict['filter_labels'].append(self.filter_labels[k])
            else:
                out_dict['filter_labels'].append((str)(filt))

            if len(self.filter_colors) >= len(unique_filters):
                out_dict['filter_colors'].append(self.filter_colors[k])
            else:
                out_dict['filter_colors'].append('')

            k += 1

        return out_dict

    def get_sample(self, idx):

        # Choosing light curve values for a specific ID
        light_curve_sample = self.light_curves_data[
            self.light_curves_data['ID'] == idx]

        return light_curve_sample
