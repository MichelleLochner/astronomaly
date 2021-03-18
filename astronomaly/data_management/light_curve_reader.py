import pandas as pd
import os
import numpy as np
from astronomaly.base.base_dataset import Dataset


class LightCurveDataset(Dataset):
    def __init__(self,data_dict,header_nrows=1,delim_whitespace =False,**kwargs):
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
                It a dictionary with index of the column names corresponding to the
                following specific keys: ('id','time','mag','mag_err','flux','flux_err','filters')
                e.g {'time':1,'mag':2}, where 1 and 2 are column index correpoding to
                'time' and 'mag' in the input data .
                The user can also provide a list of indices for the 'mag' and 'flux' columns. This is the
                case where the brightness is recorded in more than one column.
                e.g {'time':1,'mag':[2,3]} 2 and 3 corresponds to columns with brightness records
        header_nrows: int
                The number of rows the header covers in the dataset, by
                default 1
        delim_whitespace: bool
                Should be True if the data is not separated by a comma, by
                default False"""

        super().__init__(data_dict=data_dict,header_nrows=header_nrows,delim_whitespace=delim_whitespace,**kwargs)
       
        self.data_type = 'light_curve'
        self.metadata = pd.DataFrame(data=[])
        self.data_dict = data_dict
        self.header_nrows = header_nrows
        self.delim_whitespace = delim_whitespace

    #         =======================================================================================
                                    # Reading the light curve data
    #         =======================================================================================

        # The case where there is one file
        data = pd.read_csv(self.files[0],skiprows=self.header_nrows,
                           delim_whitespace=self.delim_whitespace, header=None)
            
        # The case for multiple files of light curve data
        if len(self.files) > 1:

            for fl in range(1,len(self.files)):

                data=pd.concat([data, pd.read_csv(self.files[fl],skiprows=self.header_nrows,
                                delim_whitespace=self.delim_whitespace,header=None)])
                           
    #         ========================================================================================
                            # Renaming the columns into standard columns for astronomaly 
    #         ========================================================================================

        idx = data.iloc[:,self.data_dict['id']]
        time = data.iloc[:,self.data_dict['time']]
        standard_data = {'ID':idx,'time':time}
                
        # Possible brightness columns
        brightness_cols = ['mag','flux']
        
        # Looping through the brightness columns
        for cols in range(len(brightness_cols)):
            
            if brightness_cols[cols] in self.data_dict.keys():

                # ============Multiple brightness columns=========================
                try:

                    for i in range(len(self.data_dict[brightness_cols[cols]])):

                        # The case where there are no error columns
                        standard_data.update({brightness_cols[cols]+str(i+1):
                                              data.iloc[:,self.data_dict[brightness_cols[cols]][i]]})

                        # The case where there are brightness error columns
                        if brightness_cols[cols]+'_err' in self.data_dict.keys():

                            # Updating the standard dictionary to include the brightness_errors
                            standard_data.update({brightness_cols[cols]+'_error'+str(i+1):
                                                  data.iloc[:,self.data_dict[brightness_cols[cols]+'_err'][i]]})

                #=================Single brightness Column============================
                #==============================================================
                except TypeError:

                    # The case for single brightness column and no errors
                    standard_data.update({brightness_cols[cols]:
                                          data.iloc[:,self.data_dict[brightness_cols[cols]]]})

                    if brightness_cols[cols]+'_err' in self.data_dict.keys():

                        standard_data.update({brightness_cols[cols]+'_error':
                                              data.iloc[:,self.data_dict[brightness_cols[cols]+'_err']]})

                # ============The case where there are filters in the data=================
                if 'filters' in self.data_dict.keys():

                    standard_data.update({'filters':data.iloc[:,self.data_dict['filters']]})
        
        ids = np.unique(standard_data['ID'])
        self.metadata = pd.DataFrame({'ID': ids}, index=ids)
        self.light_curves_data = pd.DataFrame.from_dict(standard_data)
        self.index = ids

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

        # All the standard columns are included here
        data_col = ['mag']
        err_col = ['mag_error']
        out_dict = {}

        # Reading in the light curve data
        light_curve = self.light_curves_data[self.light_curves_data['ID']==idx]

        # Data and error index
        mag_indx = [cl for cl in data_col if cl in light_curve.columns.values.tolist()]
        err_indx = [cl for cl in err_col if cl in light_curve.columns.values.tolist()]

        # Returns true if we have error columns
        # if err_col[0] in light_curve.columns.values.tolist() or err_col[1] in light_curve.columns.values.tolist():

        light_curve['err_lower'] = light_curve[mag_indx].values - light_curve[err_indx].values
        light_curve['err_upper'] = light_curve[mag_indx].values + light_curve[err_indx].values
        lc_errs = light_curve[['time', 'err_lower', 'err_upper']]

        # inserting the time column to data and adding 'data'
        # and 'errors' to out_dict
        mag_indx.insert(0,'time')
        out_dict['data'] = light_curve[mag_indx].values.tolist()
        out_dict['errors'] = lc_errs.values.tolist()

        return out_dict

    def get_sample(self,idx):

        # Choosing light curve values for a specific ID
        light_curve_sample = self.light_curves_data[self.light_curves_data['ID']==idx]

        return light_curve_sample