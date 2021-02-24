import numpy as np
import feets
from astronomaly.base.base_pipeline import PipelineStage

class Feets_Features(PipelineStage):

    '''Computes the features using the feets package

    Parameters:
        available_data: This is a list of data available in the input light curve.
            It can strickly be any of the following:
            magnitude, time, error, magnitude2, time2 and error2, where 2 indicates observations
            in a different band
            
    Output:
        A 1D array with the calculated feature that correspond to the available data'''
    
    def __init__(self,available_data, **kwargs):
        
        self.available_data = available_data

        super().__init__(available_data=available_data, **kwargs)
        
    def _set_labels(self):
        
        # All available features
        self.labels = feets.available_features()
               
    def _excute_function(self,lc_data):
        
        try:

            # Computing the features and their labels      
            fs = feets.FeatureSpace(data=self.available_data,exclude=["Period_fit"])
            lc_columns = [lc_data.time,lc_data.mag,lc_data.mag_error]
            features, values = fs.extract(*lc_columns)
        
            # Updating the labels
            self.labels = features
            self._set_labels()
                
            # The calculated features
            return values
                        
        except (ZeroDivisionError,UnboundLocalError,IndexError,ValueError,AttributeError) as e:
                
                print('The light curve has < 20 point')
                print(e)         