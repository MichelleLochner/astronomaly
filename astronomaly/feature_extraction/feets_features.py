import numpy as np
import feets
from astronomaly.base.base_pipeline import PipelineStage

class Feets_Features(PipelineStage):
    
    '''Computes the features using feets package

    Parameters:
        exclude_features: Features to be excluded when calculating the features
            
    Output:
        A 1D array with the extracted feature'''
    
    
    def __init__(self,exclude_features, **kwargs):
        
        self.exclude_features = exclude_features
        self.labels = None
        
        super().__init__(exclude_features=exclude_features, **kwargs)
         
    def _set_labels(self,feature_labels):
        
        # All available features
        self.labels = feature_labels     
        
    def _excute_function(self,lc_data):
        
        '''Takes light curve data for a single object and computes the features based on 
        the available columns.
        
        Input: 
            lc_data: Light curves of a single object
            
        Output:
            An array of the calculated features or an array of nan values incase there is
            an error during the feature extraction process'''
        
        # Sorting the columns for the feature extractor
        standard_lc_columns = ['time','mag','mag_error'] # This needs to be extended to be more general
        current_lc_columns = [cl for cl in standard_lc_columns if cl in lc_data.columns]
        available_columns = ['time']
        
        # Renaming the columns for feets
        for cl in current_lc_columns:
   
            if cl == 'mag':

                available_columns.append('magnitude')

            if cl == 'mag_error':

                available_columns.append('error')
                
        # Getting the length of features to be calculated
        if len(current_lc_columns) == 3:
            
            len_labels = 63 - len(self.exclude_features)
            
        if len(current_lc_columns) == 2:
            
            len_labels = 57 - len(self.exclude_features)

        # Computing the features            
        if len(lc_data) >= 20:
             
            fs = feets.FeatureSpace(data=available_columns,
                                    exclude=self.exclude_features)
            
            # Getting the light curve columns for the extractor
            lc_columns = []
            for cols in current_lc_columns:
                
                lc_columns.append(lc_data[cols])
                
            features, values = fs.extract(*lc_columns)
                
            # Updating the labels
            if self._set_labels(features) == None:
           
                self._set_labels(list(features))
                
            # The calculated features
            return values
        
        # Returns an array of nan values
        else:
                
                return np.array([np.nan for i in range(len_labels)])       