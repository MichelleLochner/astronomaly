import pandas as pd 
import os
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
                following specific keys: ('time','mag','mag_err','flux','flux_err','filters')
                
                e.g {'time':1,'mag':2}, were 1 and 2 are column index correpoding to 
                'time' and 'mag' in the input data 
        
        header_nrows: int
                The number of rows the header covers in the dataset, by 
                default 1
                
         delim_whitespace: bool
                Should be True if the data is not separated by a comma, by
                default False
                

        """

        super().__init__(data_dict,header_nrows=1,delim_whitespace =False,**kwargs)



        self.data_type = 'light_curve'
        self.metadata = pd.DataFrame(data=[])
        

        ##### need to understand this line of code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ids = [f.split(os.sep)[-1] for f in self.files]
        # self.metadata = pd.DataFrame({'filepath': self.files}, index=ids)

        

        self.data_dict = data_dict
        self.header_nrows = header_nrows
        self.delim_whitespace = delim_whitespace
     

        
        

#     @staticmethod
    def read_lc_from_file(self,flpath):

        
        '''Function to read the lc from the data
        
        Input:
        flpath: the location of the file
        
    
                
        Output:
       standardized pandas dataframe with lc data'''
        
        
        # Reading-in the data
        data = pd.read_csv(flpath,skiprows=self.header_nrows,delim_whitespace=self.delim_whitespace,header=None)
        
        
        # ==================Magnitudes==================================
        # ==============================================================
        ID = data.iloc[:,self.data_dict['id']]
        if 'mag' in self.data_dict.keys(): 
            
            
            # ============MUtliple Mag columns=========================
            
            # The case of multiple brightness columns        
            if type(self.data_dict['mag']) == list:
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; mag1 = data.iloc[:,self.data_dict['mag'][0]];
                mag2 = data.iloc[:,self.data_dict['mag'][1]]
                
                # Case where there are brightness error columns
                if 'mag_err' in self.data_dict.keys():                
                    
                    mag_error = data.iloc[:,self.data_dict['mag_err']]
                    # Creating a new dictionary for the columns above separate data
                    standard_data = {'ID':ID,'time':time,'mag1':mag1,'mag2':mag2,'mag_error':mag_error}
                    
                # Case were there are no error columns
                else:
                    
                    standard_data = {'ID':ID,'time':time,'mag1':mag1,'mag2':mag2}
                            
            
            
                    
            # ============Column with Mag_filters and errors==========================
            
            # Including filters in dataframe
            elif 'filters'in self.data_dict.keys() and 'mag_err' in self.data_dict.keys():
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; mag = data.iloc[:,self.data_dict['mag']];
                mag_error = data.iloc[:,self.data_dict['mag_err']]

                filters = data.iloc[:,self.data_dict['filters']]
                standard_data = {'ID':ID,'time':time,'mag':mag,'mag_error':mag_error,'filters':filters}
                
            elif 'filters' in self.data_dict.keys():
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; mag = data.iloc[:,self.data_dict['mag']];
        

                filters = data.iloc[:,self.data_dict['filters']]
                standard_data = {'ID':ID,'time':time,'mag':mag,'filters':filters}
                
                
                
                
            #=================Single Mag Column with and with errors============================
            
            # Case of single brightness columns    
            else:    
                
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; mag = data.iloc[:,self.data_dict['mag']]; 
                
                if 'mag_err' in self.data_dict.keys():
                    
                    
                    mag_error = data.iloc[:,self.data_dict['mag_err']]
                    # Creating a new dictionary for the columns above separate data
                    standard_data = {'ID':ID,'time':time,'mag':mag,'mag_error':mag_error}
                    
                else:
                    
                    standard_data = {'ID':ID,'time':time,'mag':mag}
                    
                    
    #-----------------------------------------------------------------------------------------------------------------                
                
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````````````      
    #============================================Fluxes===============================================================
    #=================================================================================================================
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````````````
            
        else:
            
            
            
                    # ============MUtliple Mag columns=========================
            
            # The case of multiple brightness columns        
            if type(self.data_dict['flux']) == list:
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; flux1 = data.iloc[:,self.data_dict['flux'][0]];
                flux2 = data.iloc[:,self.data_dict['flux'][1]]
                
                # Case where there are brightness error columns
                if 'flux_err' in self.data_dict.keys():                
                    
                    flux_error = self.data.iloc[:,self.data_dict['flux_err']]
                    # Creating a new dictionary for the columns above separate data
                    standard_data = {'ID':ID,'time':time,'flux1':flux1,'flux2':flux2,'flux_error':flux_error}
                    
                # Case were there are no error columns
                else:
                    
                    standard_data = {'ID':ID,'time':time,'flux1':flux1,'flux2':flux2}
                            
            
            
                    
            # ============Column with Mag_filters and errors==========================
            
            # Including filters in dataframe
            elif 'filters'in self.data_dict.keys() and 'flux_err' in self.data_dict.keys():
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; flux = data.iloc[:,self.data_dict['flux']];
                flux_error = data.iloc[:,self.data_dict['flux_err']]

                filters = data.iloc[:,self.data_dict['filters']]
                standard_data = {'ID':ID,'time':time,'flux':flux,'flux_error':flux_error,'filters':filters}
                
            elif 'filters' in self.data_dict.keys():
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; flux = data.iloc[:,self.data_dict['flux']];
        

                filters = data.iloc[:,self.data_dict['filters']]
                standard_data = {'ID':ID,'time':time,'flux':flux,'filters':filters}
                
                
                
                
            #=================Single Mag Column with and with errors============================
            
            # Case of single brightness columns    
            else:    
                
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; flux = data.iloc[:,self.data_dict['flux']]; 
                
                if 'flux_err' in self.data_dict.keys():
                    
                    
                    flux_error = data.iloc[:,self.data_dict['flux_err']]
                    # Creating a new dictionary for the columns above separate data
                    standard_data = {'ID':ID,'time':time,'flux':flux,'flux_error':flux_error}
                    
                else:
                    
                    standard_data = {'ID':ID,'time':time,'flux':flux}
            
            
        
        return pd.DataFrame.from_dict(standard_data)
    
    
    

    def get_display_data(self, idx,flpath):
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
#         time_col = 'time'

        # All the standard columns are included here
        data_col = ['time','mag','flux','mag1','mag2','flux1','flux2','filters']

        err_col = ['mag_error','flux_error']

        out_dict = {}

        metadata = self.metadata

        ##### need to understand this line of code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # flpath = metadata[idx]['filepath'].iloc[0]

        try:

            
            # Reading in the light curve data
            light_curve = self.read_lc_from_file(flpath)
            light_curve = light_curve[light_curve['ID']==idx]
            
            # Data and error index 
            data_indx = [cl for cl in data_col if cl in light_curve.columns.values.tolist()] 
            err_indx = [cl for cl in err_col if cl in light_curve.columns.values.tolist()]

            
            # Getting both the data and error columns as per index
            out_dict['data'] = light_curve[data_indx].values.tolist()
            
            
            # Returns true if we have error columns
            if err_col[0] in light_curve.columns.values.tolist() or err_col[1] in light_curve.columns.values.tolist():
                lc_errs = light_curve[err_indx]
                out_dict['errors'] = lc_errs.values.tolist()
                
             

                

        except (pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError) as e:
            print('Error parsing file', flpath)
            print('Error message:')
            print(e)
            out_dict = {'data': [], 'errors': []}

        return out_dict
    

    def get_sample(self,idx,flpath):
        
        
        # All the standard columns for feature extraction 
        data_col = ['time','mag','flux','mag1','mag2','flux1','flux2','mag_error','flux_error']


    
        metadata = self.metadata

        ##### need to understand this line of code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # flpath = metadata[idx]['filepath'].iloc[0]

        # empty pandas dataframe to update as per data_col
        out_data = pd.DataFrame({})
        try:

            
            # Reading in the light curve data
            light_curve = self.read_lc_from_file(flpath)
            
            # Choosing light curve values for a specific ID
            light_curve = light_curve[light_curve['ID']==idx]
            
            
            sample_data = []
            # Getting the columns that correspond to lc columns 
            for cl in data_col:
                
                if cl in light_curve.columns.values.tolist():

                    out_data[cl] = light_curve[cl].values.tolist()
                    sample_data.append(out_data[cl])
            
                
            return sample_data
                
             

                

        except (pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError) as e:
            print('Error parsing file', flpath)
            print('Error message:')
            print(e)
            out_dict = {'data': [], 'errors': []}

    