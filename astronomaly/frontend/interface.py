import numpy as np
from astronomaly.anomaly_detection import human_loop_learning
import os
import importlib
import sys


class Controller:
    def __init__(self, pipeline_file):
        self.dataset = None
        self.features = None
        self.anomaly_scores = None
        self.clustering = None
        self.module_name = None
        self.active_learning = None

        self.set_pipeline_script(pipeline_file)

    def run_pipeline(self):
        pipeline_script = importlib.import_module(self.module_name)
        print('Running pipeline from', self.module_name+'.py')
        pipeline_dict = pipeline_script.run_pipeline()

        ######## Add some try catches here

        self.dataset = pipeline_dict['dataset']
        self.features = pipeline_dict['features']
        self.anomaly_scores = pipeline_dict['anomaly_scores']
        if 'cluster' in list(pipeline_dict.keys()):
            self.clustering = pipeline_dict['cluster']
        if 'active_learning' in list(pipeline_dict.keys()):
            self.active_learning = pipeline_dict['active_learning']

    def get_data_type(self):
        return self.dataset.data_type

    def set_pipeline_script(self, pipeline_file):
        module_name = pipeline_file.split(os.path.sep)[-1]
        pth = pipeline_file.replace(module_name, '')
        module_name = module_name.split('.')[0]

        self.module_name = module_name
        sys.path.append(pth) # Allows importing the module from anywhere

    def get_display_data(self, idx):
        try:
            return self.dataset.get_display_data(idx)
        except KeyError:
            return {}

    def get_features(self, idx):
        try:
            out_dict = dict(zip(self.features.columns.astype('str'), self.features.loc[idx].values))
            return out_dict
        except KeyError:
            return {}

    def set_human_label(self, idx, label):
        ml_df = self.anomaly_scores
        if 'human_label' not in ml_df.columns:
            ml_df['human_label'] = [-1] * len(ml_df)
        ml_df.loc[idx, 'human_label'] = label

    def run_active_learning(self):
        ##################################################
        pipeline_active_learning = self.active_learning
        features_with_labels = pipeline_active_learning.combine_data_frames(self.features, self.anomaly_scores)
        # print(features_with_labels)
        scores = pipeline_active_learning.run(features_with_labels)
        self.anomaly_scores['final_score'] = scores

    def get_cluster_data(self, color_by_column=''):
        clst = self.clustering
        if clst is not None:
            if len(color_by_column) == 0:
                cols = [0.5] * len(clst)
                clst['color'] = cols
            else:
                clst['color'] = self.anomaly_scores.loc[self.anomaly_scores.index, color_by_column]
            out = []
            for idx in clst.index:
                dat = clst.loc[idx].values
                out.append({'id': (str)(idx), 'x': '{:f}'.format(dat[0]), 'y': '{:f}'.format(dat[1]),
                            'opacity': '0.5', 'color': '{:f}'.format(clst.loc[idx, 'color'])})
            return out
        else:
            return None

    def get_original_id_from_index(self, ind):
        this_ind = list(self.anomaly_scores.index)[ind]
        return this_ind

    def get_metadata(self, idx, exclude_keywords=[], include_keywords=[]):
        idx = str(idx)
        meta_df = self.dataset.metadata
        ml_df = self.anomaly_scores

        try:
            out_dict = {}
            if len(include_keywords) != 0:
                cols = include_keywords
            else:
                cols = meta_df.columns

            for col in cols:
                if col not in exclude_keywords:
                    out_dict[col] = (str)(meta_df.loc[idx, col])
            for col in ml_df.columns:
                if col not in exclude_keywords:
                    out_dict[col] = (str)(ml_df.loc[idx, col])
            return out_dict
        except KeyError:
            return {}

    def randomise_ml_scores(self):
        inds = np.random.permutation(self.anomaly_scores.index)
        self.anomaly_scores = self.anomaly_scores.loc[inds]

    def sort_ml_scores(self, column_to_sort_by='score'):
        anomaly_scores = self.anomaly_scores
        if column_to_sort_by in anomaly_scores.columns:
            if column_to_sort_by == "iforest_score":
                ascending = True
            else:
                ascending = False
            anomaly_scores.sort_values(column_to_sort_by, inplace=True, ascending=ascending)
        else:
            print("Requested column not in ml_scores dataframe")

# Various functions that are called by the REST API in run_server.py

# which_data = 'image'
# which_dataset = 'deep2'
#
# # pipeline_dict = pipeline.run_pipeline(image_dir='/home/michelle/BigData/Anomaly/GOODS_S/', features='psd', dim_reduct='pca')
# # Image downloaded from here: https://archive.stsci.edu/pub/hlsp/goods/v2/
# if which_dataset == 'deep2':
#     #image_dir = '/home/michelle/BigData/Anomaly/Meerkat_deep2/'
#     #image_dir = '/home/michelle/BigData/Anomaly/Meerkat_clusters/'
#     # image_dir = '/home/michelle/BigData/Anomaly/Meerkat_abell/'
#     image_dir = '/home/michelle/BigData/Anomaly/Meerkat_not_abell/'
#     # image_dir = '/home/michelle/BigData/Anomaly/Meerkat_galaxy/'
#     cutouts_file = ''
#     anomaly_file = ''
#     output_dir = image_dir + 'output'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
# elif which_dataset == 'hsc':
#     cutouts_file = '/home/michelle/BigData/Anomaly/hsc_data/imarr_i20.0_100k.npy'
#     anomaly_file = '/home/michelle/BigData/Anomaly/hsc_data/anomaly_score.csv'
#     image_dir = ''
#
#
#
#
# dwf_dir = '/home/michelle/BigData/Anomaly/dwf_data/2015_01_CDF-S/'
# light_curve_dir = os.path.join(dwf_dir, 'light_curves/', 'files/')
# # features_file = os.path.join(dwf_dir,'2015_01_CDF-S_150114_feats_with_gaia_FULL.ascii')
# features_file = os.path.join(dwf_dir,'2015_01_CDF-S_150115_feats_with_gaia_FULL.ascii')
# # features_file = os.path.join(dwf_dir,'2015_01_CDF-S_150117_feats_with_gaia_FULL.ascii')
#
# if which_data == 'image':
#     features = 'psd2d'
#     dim_reduct = 'pca'
#     scaled = 'scaled'
#     anomaly_algo = 'iforest'
#     clustering = 'tsne'
#     pipeline_dict = image_pipeline.run_pipeline(image_dir=image_dir, cutouts_file=cutouts_file,
#                                     features=features, dim_reduct=dim_reduct,
#                                     scaled = scaled,
#                                     anomaly_algo=anomaly_algo, clustering=clustering,
#                                     nproc=6, anomaly_file=anomaly_file,
#                                     output_dir=output_dir)
# else:
#     features = 'from_file'
#     dim_reduct = ''
#     scaled = 'scaled'
#     anomaly_algo = 'iforest'
#     clustering = 'tsne'
#     pipeline_dict = light_curve_pipeline.run_pipeline(light_curve_dir=light_curve_dir,
#         features_file=features_file)



# window_size_x= dataset.window_size_x
# window_size_y = dataset.window_size_y

# def sort_ml_scores(column_to_sort_by='score'):
#     anomaly_scores = pipeline_dict['anomaly_scores']
#     print("sorting by"+column_to_sort_by)
#     if column_to_sort_by in anomaly_scores.columns:
#         if column_to_sort_by == "iforest_score":
#             ascending = True
#         else:
#             ascending = False
#         anomaly_scores.sort_values(column_to_sort_by, inplace=True, ascending=ascending)
#     else:
#         print("Requested column not in ml_scores dataframe")

# def randomise_ml_scores():
#     anomaly_scores = pipeline_dict['anomaly_scores']
#     inds = np.random.permutation(anomaly_scores.index)
#     pipeline_dict['anomaly_scores'] = anomaly_scores.loc[inds]




# def get_image(id, load_cutout=False):
#     return pipeline_dict['dataset'].get_display_image(id)

# def read_lc_from_file(flpath, lower_mag=1, upper_mag=25):
#
#     light_curve = pd.read_csv(flpath, delim_whitespace=True)
#     return light_curve
#
# def get_light_curve(id):
#     # print(id)
#     ### Need to extend this to deal with other bands
#     time_col = 'MJD'
#     mag_col = 'g_mag'
#     err_col = 'g_mag_err'
#
#     out_dict = {}
#
#     metadata = pipeline_dict['metadata']
#     flpath = metadata[metadata.id==id]['filepath'].iloc[0]
#     try:
#         light_curve = read_lc_from_file(flpath)
#         light_curve = light_curve[(1<light_curve[mag_col])&(light_curve[mag_col]<25)]
#         light_curve['err_lower'] = light_curve[mag_col] - light_curve[err_col]
#         light_curve['err_upper'] = light_curve[mag_col] + light_curve[err_col]
#
#         out_dict['data'] = light_curve[[time_col, mag_col]].values.tolist()
#         out_dict['errors'] = light_curve[[time_col, 'err_lower','err_upper']].values.tolist()
#
#     except (pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError) as e:
#         print('Error parsing file', flpath)
#         print('Error message:')
#         print(e)
#         out_dict = {'data':[], 'errors':[]}
#
#     return out_dict

# def get_features(id):
#     # print(features_key)
#     dat = pipeline_dict['features']
#     # print(dat)
#     out_dict = dict(zip(dat.columns.astype('str'),dat.loc[id].values))
#     return out_dict

# def set_human_label(idx, label):
#     ml_df = pipeline_dict['anomaly_scores']
#     if 'human_label' not in ml_df.columns:
#         ml_df['human_label'] = [-1]*len(ml_df)
#     ml_df.loc[idx, 'human_label'] = label
    #ml_df.to_parquet('/home/michelle/BigData/Anomaly/Meerkat_clusters/output/labeled.parquet')

# def run_active_learning():
#     ml_df = pipeline_dict['anomaly_scores']
#     features = pipeline_dict['features']
#     pipeline_active_learning = human_loop_learning.NeighbourScore(alpha=1)
#     pipeline_active_learning._execute_function(features, ml_df)


# def get_tsne_data(color_by_column=''):
    # ts = pipeline_dict['tsne']
    # # out_str = '['
    # # for i in range(len(ts.id))[:5]:
    # #     out_str+="{{x:{:f}, y:{:f} }},".format(dat[i,0], dat[i,1])
    # # out_str = out_str[:-1]
    # # out_str += ']'
    # # print(out_str)
    # # return out_str
    # if len(color_by_column)==0:
    #     cols = [0.5]*len(ts)
    #     ts['colour'] =cols
    # else:
    #     ## This is pretty ugly, maybe storing as pd dataframes is better after all
    #     #temp_df = pd.concat((ts, pipeline_dict['anomaly_scores']))
    #     #cols = temp_df[color_by_column].values
    #     ts['color'] = pipeline_dict['anomaly_scores'][color_by_column]
    # #print(ts)
    # out = []
    # for idx in ts.index:
    #     dat = ts.loc[idx].values
    #     out.append({'id':(str)(idx), 'x':'{:f}'.format(dat[0]), 'y':'{:f}'.format(dat[1]),
    #     'opacity':'0.5', 'color':'{:f}'.format(ts.loc[idx,'color'])})
    #     # out.append({'x':dat[i,0], 'y':dat[i,1]})
    # return out

# def get_original_id_from_index(ind):
#     anomaly_scores = pipeline_dict['anomaly_scores']
#     this_ind =list(anomaly_scores.index)[ind]
#     return this_ind

# def get_metadata(idx, exclude_keywords=[], include_keywords=[]):
#     idx = str(idx)
#     meta_df = pipeline_dict['dataset'].metadata
#     ml_df = pipeline_dict['anomaly_scores']
#
#     out_dict = {}
#     if len(include_keywords) != 0:
#         cols = include_keywords
#     else:
#         cols = meta_df.columns
#
#     for col in cols:
#         if col not in exclude_keywords:
#             out_dict[col] = (str)(meta_df.loc[idx,col])
#     for col in ml_df.columns:
#         if col not in exclude_keywords:
#             out_dict[col] = (str)(ml_df.loc[idx,col])
#     return out_dict



