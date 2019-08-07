import pandas as pd
import xarray

def read_from_file(pipeline_dict, file_location, output_key='features_from_file', drop_na=True):
    print('Reading in features from file ',file_location,'...')
    df = pd.read_csv(file_location, delim_whitespace=True)
    if drop_na:
        df = df.dropna()
    ids = df[df.columns[0]]
    xr=xarray.DataArray(df[df.columns[1:]], 
        coords={'id': ids, 'features':df.columns[1:]}, dims=['id', 'features'])

    pipeline_dict[output_key] = xr
    print('Done!')

    return pipeline_dict