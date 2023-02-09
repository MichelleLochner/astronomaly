import numpy as np
import feets
from astronomaly.base.base_pipeline import PipelineStage
import warnings
from astronomaly.base import logging_tools


class Feets_Features(PipelineStage):
    def __init__(self, exclude_features, 
                 compute_on_mags=False, 
                 ignore_warnings=False, 
                 filter_labels=['u', 'g', 'r', 'i', 'z', 'y'],
                 **kwargs):
        """
        Applies the 'feets' general time series feature extraction package

        Parameters
        ----------
        exclude_features : list
            List of features to be excluded when calculating the features (as
            strings)
        compute_on_mags : bool
            If true, will convert flux to magnitude
        ignore_warnings : bool
            The feets feature extraction package raises many, many warnings
            especially when run on large datasets. This flag will disable all
            warning printouts from feets. It is HIGHLY recommended
            to first check the warnings before disabling them.
        filter_labels : list
            Optional list of strings corresponding to the name of each filter.
            We explicitly assume each observations' filter (if available) has
            a numerical value, translated to a string using this list
        output_dir : str
            The directory to save the log file and all outputs to. Defaults to
            './' 
        force_rerun : bool
            If True will force the function to run over all data, even if it 
            has been called before.
        """

        super().__init__(exclude_features=exclude_features,
                         compute_on_mags=compute_on_mags,
                         ignore_warnings=ignore_warnings, 
                         filter_labels=filter_labels, **kwargs)

        self.exclude_features = exclude_features
        self.labels = None
        self.compute_on_mags = compute_on_mags
        self.ignore_warnings = ignore_warnings
        self.filter_labels = filter_labels

    def _set_labels(self, feature_labels):
        """
        Because the number of features may not be known till runtime, we can
        only create the labels of these features at runtime.
        """
        # All available features
        self.labels = feature_labels

    def _execute_function(self, lc_data):
        """
        Takes light curve data for a single object and computes the features
        based on the available columns.

        Parameters
        ----------
        lc_data: pandas DataFrame
            Light curve of a single object

        Returns
        -------
        array
            An array of the calculated features or an array of nan values
            incase there is an error during the feature extraction process
        """

        with warnings.catch_warnings():
            if self.ignore_warnings:
                # Feets produces a lot of warnings that can't easily be
                # redirected, this switches them off
                warnings.simplefilter('ignore')
            if self.compute_on_mags is True and 'mag' not in lc_data.columns:
                msg = """compute_on_mags selected but no magnitude column
                found - switching to flux"""
                logging_tools.log(msg, level='WARNING')

            if self.compute_on_mags is True and 'mag' in lc_data.columns:
                standard_lc_columns = ['time', 'mag', 'mag_error']

            else:
                standard_lc_columns = ['time', 'flux', 'flux_error']

            current_lc_columns = [cl for cl in standard_lc_columns
                                  if cl in lc_data.columns]

            # list to store column names supported by  feets
            available_columns = ['time']

            # Renaming the columns for feets
            for cl in current_lc_columns:
                if cl == 'mag' or cl == 'flux':
                    available_columns.append('magnitude')

                if cl == 'mag_error' or cl == 'flux_error':
                    available_columns.append('error')

            # Creates the feature extractor
            fs = feets.FeatureSpace(data=available_columns,
                                    exclude=self.exclude_features)

            # Getting the length of features to be calculated
            len_labels = len(fs.features_)
            # print(fs.features_)
            # The case where we have filters
            if 'filters' in lc_data.columns:
                ft_values = []
                ft_labels = []
                for i in np.unique(lc_data.filters):
                    passbands = self.filter_labels
                    filter_lc = lc_data[lc_data['filters'] == i]

                    lc_columns = []
                    for col in current_lc_columns:
                        lc_columns.append(filter_lc[col])

                    # Accounts for light curves that do not have some filters
                    if len(filter_lc.ID) != 0:
                        # Checking the number of points in the light curve
                        if len(filter_lc.ID) >= 5:
                            features, values = fs.extract(*lc_columns)

                            new_labels = [f + '_' + passbands[i]
                                          for f in features]

                            for j in range(len(features)):
                                ft_labels.append(new_labels[j])
                                ft_values.append(values[j])

                        else:
                            for ft in fs.features_:
                                ft_labels.append(ft + '_' + passbands[i])
                                ft_values.append(np.nan)

                    else:
                        for vl in fs.features_:
                            ft_values.append(np.nan)
                            ft_labels.append(vl + '_' + passbands[i])

                # Updating the labels
                if self.labels is None:
                    self._set_labels(list(ft_labels))

                # print(self.labels)
                return ft_values

            # The case with no filters
            else:
                if len(lc_data.ID) >= 5:
                    # print('passed')
                    lc_columns = []
                    for col in current_lc_columns:
                        lc_columns.append(lc_data[col])

                    ft_labels, ft_values = fs.extract(*lc_columns)

                    # # Updating the labels
                    if self.labels is None:
                        self._set_labels(list(ft_labels))
                    return ft_values

                # Feature extraction fails so returns an array of nan values
                else:
                    return np.array([np.nan for i in range(len_labels)])
