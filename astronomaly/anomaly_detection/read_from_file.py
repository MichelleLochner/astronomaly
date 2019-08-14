import pandas as pd

def read_anomaly_score(pipeline_dict, filename, file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv(filename, dtype={'id':'str'})
        
        if 'ml_scores' in pipeline_dict.keys():
            ml_df = pipeline_dict['ml_scores']
            pipeline_dict['ml_scores']= ml_df.merge(df, on='id')
        else:
            pipeline_dict['ml_scores'] = df

    return pipeline_dict