import pandas as pd 
import os
import numpy as np
# from astronomaly.base.base_dataset import Dataset


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
                'time' and 'mag' in the input data .
                
                The user can also provide a list of indices for the 'mag' and 'flux' columns. This is the
                case were the brightness is recorded in more than one column.
                
                e.g {'time':1,'mag':[2,3]} 2 and 3 corresponds to columns with brightness records
        
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
     
        
#         ========================================================================================
                                   
                                    # Reading the light curve data 
        
#         ========================================================================================

        
        
        # Reading-in the data
        
        
        
        # ===================Case for multiple files of light curve data================================
        try:

            data=pd.concat([pd.read_csv(self.files[0],skiprows=self.header_nrows,
                               delim_whitespace=self.delim_whitespace,header=None),
                               
                               pd.read_csv(self.files[1],skiprows=self.header_nrows,
                                           delim_whitespace=self.delim_whitespace,header=None)])
            
            

            for fl in range(2,len(self.files)):

                data=pd.concat([data, pd.read_csv(self.files[fl],skiprows=self.header_nrows,
                               delim_whitespace=self.delim_whitespace,header=None)])
                
        
        # ===================Case for single file of light curve data==================================
        except IndexError:
            
            
            data = pd.read_csv(self.files[0],skiprows=self.header_nrows,
                               delim_whitespace=self.delim_whitespace,header=None)

            

        

        
        # ==================Magnitudes==================================
        # ==============================================================
        Id = data.iloc[:,self.data_dict['id']]
        time = data.iloc[:,self.data_dict['time']]
        
        standard_data = {'ID':Id,'time':time}
        
        if 'mag' in self.data_dict.keys(): 
            
            
            # ============MUtliple Mag columns=========================
            
            # The case of multiple brightness columns        
            try:
                
                
        
                for i in range(len(self.data_dict['mag'])):
                    
                    
                    # Separatting the columns as per input dictionary
    #             

                    # Case where there are brightness error columns
                    if 'mag_err' in self.data_dict.keys():                

                        # Creating a new dictionary for the columns above separate data
                        standard_data.update({'mag'+str(i+1):data.iloc[:,self.data_dict['mag'][i]],
                                              'mag_error'+str(i+1):data.iloc[:,self.data_dict['mag_err'][i]]})

                    # Case were there are no error columns
                    else:

                        standard_data.update({'mag'+str(i+1):data.iloc[:,self.data_dict['mag'][i]]})
                    
                    
            
                
            #=================Single Mag Column with and with errors============================
            
            # Case of single brightness columns    
            except TypeError:    
                
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; mag = data.iloc[:,self.data_dict['mag']]; 
                
                if 'mag_err' in self.data_dict.keys():
                    
                    
                    mag_error = data.iloc[:,self.data_dict['mag_err']]
                    # Creating a new dictionary for the columns above separate data
                    standard_data.update({'mag':mag,'mag_error':mag_error})
                    
                else:
                    
                    standard_data.update({'mag':mag})
                            
            
            
                    
            # ============Column with Mag_filters and errors==========================
            
            # Including filters in dataframe
            if 'filters'in self.data_dict.keys() and 'mag_err' in self.data_dict.keys():
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; mag = data.iloc[:,self.data_dict['mag']];
                mag_error = data.iloc[:,self.data_dict['mag_err']]

                filters = data.iloc[:,self.data_dict['filters']]
                standard_data.update({'mag':mag,'mag_error':mag_error,'filters':filters})
                
            elif 'filters' in self.data_dict.keys():
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; mag = data.iloc[:,self.data_dict['mag']];
        

                filters = data.iloc[:,self.data_dict['filters']]
                standard_data.update({'mag':mag,'filters':filters})
                
                
                
                    
                    
    #-----------------------------------------------------------------------------------------------------------------                
                
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````````````      
    #============================================Fluxes===============================================================
    #=================================================================================================================
    #`````````````````````````````````````````````````````````````````````````````````````````````````````````````````
            
        else:
            
            
            
                    # ============MUtliple Mag columns=========================
            
            # The case of multiple brightness columns        
            try:
                
                
        
                for i in range(len(self.data_dict['flux'])):
                    
                    
                    # Separatting the columns as per input dictionary
    #             

                    # Case where there are brightness error columns
                    if 'flux_err' in self.data_dict.keys():                

                        # Creating a new dictionary for the columns above separate data
                        standard_data.update({'flux'+str(i+1):data.iloc[:,self.data_dict['flux'][i]],
                                              'flux_error'+str(i+1):data.iloc[:,self.data_dict['flux_err'][i]]})
                    # Case were there are no error columns
                    else:

                        standard_data.update({'flux'+str(i+1):data.iloc[:,self.data_dict['flux'][i]]})
                    
                    
                    
            except TypeError:    
                
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; flux = data.iloc[:,self.data_dict['flux']]; 

                if 'flux_err' in self.data_dict.keys():


                    flux_error = data.iloc[:,self.data_dict['flux_err']]
                    # Creating a new dictionary for the columns above separate data
                    standard_data.update({'flux':flux,'flux_error':flux_error})

                else:

                    standard_data.update({'flux':flux})


            
                    
            # ============Column with Mag_filters and errors==========================
            
            # Including filters in dataframe
            if 'filters'in self.data_dict.keys() and 'flux_err' in self.data_dict.keys():
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; flux = data.iloc[:,self.data_dict['flux']];
                flux_error = data.iloc[:,self.data_dict['flux_err']]

                filters = data.iloc[:,self.data_dict['filters']]
                standard_data.update({'flux':flux,'flux_error':flux_error,'filters':filters})
                
            elif 'filters' in self.data_dict.keys():
                
                # Separatting the columns as per input dictionary
                time = data.iloc[:,self.data_dict['time']]; flux = data.iloc[:,self.data_dict['flux']];
        

                filters = data.iloc[:,self.data_dict['filters']]
                standard_data.update({'flux':flux,'filters':filters})
                
                
                
                
            #=================Single Mag Column with and with errors============================
            
            # Case of single brightness columns    
          
            
        ids = np.unique(standard_data['ID'])   
        self.metadata = pd.DataFrame({'ID': ids}, index=ids)
        self.light_curves_data = pd.DataFrame.from_dict(standard_data)
    
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
        data_col = ['mag','flux','mag1','mag2','flux1','flux2','filters']

        err_col = ['mag_error','flux_error']

        out_dict = {}


            
        # Reading in the light curve data

        light_curve = self.light_curves_data[self.light_curves_data['ID']==idx]

        # Data and error index 
        mag_indx = [cl for cl in data_col if cl in light_curve.columns.values.tolist()] 
        err_indx = [cl for cl in err_col if cl in light_curve.columns.values.tolist()]




        # Returns true if we have error columns
        if err_col[0] in light_curve.columns.values.tolist() or err_col[1] in light_curve.columns.values.tolist():


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
           
            
            
            
