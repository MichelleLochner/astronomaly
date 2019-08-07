import pandas as pd 
import os

def read_light_curves(directory='', list_of_files=[], num_files=-1):
    print('Reading light curves...')

    ### Need to extend this to deal with other bands
        # time_col = 'MJD'
        # mag_col = 'g_mag'
        # err_col = 'g_mag_err'

    if len(list_of_files) != 0 and len(directory)==0:
        # Assume the list of files are absolute paths
        fls = list_of_files
    elif len(list_of_files) != 0 and len(directory)!=0:
        # Assume the list of files are relative paths to directory
        fls = list_of_files
        fls = [os.path.join(directory, f) for f in fls]
    elif len(directory) != 0:
        #Assume directory contains all the files we need
        fls = os.listdir(directory)
        fls.sort()
        fls = [os.path.join(directory, f) for f in fls]
    else:
        fls = []

    ids = [f.split(os.sep)[-1] for f in fls]
    metadata = pd.DataFrame({'id':ids, 'filepath':fls})
        
        # if num_files != -1:
        #     fls = fls[1:num_files]
        
        # ids = []
        # for f in fls:
        #     flname = f.split(os.sep)[-1]
        #     if os.path.exists(f):
        #         try:
        #             light_curve = pd.read_csv(f, delim_whitespace=True)
        #             light_curve = light_curve[(0<light_curve[mag_col])&(light_curve[mag_col]<26)]
        #             if len(light_curve)>0:
        #                 out_dict[flname] = light_curve
        #                 ids.append(flname)
        #         except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        #             print('Error parsing file', f)
        #             print('Error message:')
        #             print(e)


        # if len(out_dict) == 0:
        #     print("No light curves found")

    # pipeline_dict = {'light_curves': out_dict, 'metadata':metadata, 'ml_scores':ml_scores}

    pipeline_dict = {'metadata':metadata}
    print('Done!')
    return pipeline_dict
