from astronomaly.data_management import light_curve_reader
from astronomaly.feature_extraction import load_features
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.clustering import tsne
from astronomaly.postprocessing import scaling

def run_pipeline(light_curve_dir, features_file='', features = 'from_file', dim_reduct = 'pca', 
                scaled = 'scaled', anomaly_algo = 'iforest', clustering='tsne'):

    pipeline_dict = light_curve_reader.read_light_curves(light_curve_dir)
    if len(features_file) != 0:
        pipeline_dict = load_features.read_from_file(pipeline_dict, features_file)
        input_key='features_'+features

    if scaled == 'scaled':
        pipeline_dict = scaling.scale_features(pipeline_dict, input_key)
        input_key = input_key + '_scaled'

    if anomaly_algo=='iforest':
        pipeline_dict = isolation_forest.run_isolation_forest(pipeline_dict, 
        input_key=input_key, output_column_name='iforest_score')
        pipeline_dict = human_loop_learning.convert_anomaly_score(pipeline_dict, 'iforest_score',
                                                              output_column='anomaly_score')
    
    
    if clustering == 'tsne':
        pipeline_dict = tsne.make_tsne(pipeline_dict, input_key, sort_by_column='anomaly_score',
        perplexity=50)

    return pipeline_dict
