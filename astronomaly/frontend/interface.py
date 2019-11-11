import numpy as np
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
        print('Running pipeline from', self.module_name + '.py')
        pipeline_dict = pipeline_script.run_pipeline()

        # ***** Add some try catches here

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
        sys.path.append(pth)  # Allows importing the module from anywhere

    def get_display_data(self, idx):
        try:
            return self.dataset.get_display_data(idx)
        except KeyError:
            return {}

    def get_features(self, idx):
        try:
            out_dict = dict(zip(self.features.columns.astype('str'), 
                                self.features.loc[idx].values))
            return out_dict
        except KeyError:
            return {}

    def set_human_label(self, idx, label):
        ml_df = self.anomaly_scores
        if 'human_label' not in ml_df.columns:
            ml_df['human_label'] = [-1] * len(ml_df)
        ml_df.loc[idx, 'human_label'] = label

    def run_active_learning(self):
        # ****************
        pipeline_active_learning = self.active_learning
        features_with_labels = \
            pipeline_active_learning.combine_data_frames(self.features, 
                                                         self.anomaly_scores)
        print(features_with_labels)
        scores = pipeline_active_learning.run(features_with_labels)
        self.anomaly_scores['final_score'] = scores

    def get_cluster_data(self, color_by_column=''):
        clst = self.clustering
        if clst is not None:
            if len(color_by_column) == 0:
                cols = [0.5] * len(clst)
                clst['color'] = cols
            else:
                clst['color'] = \
                    self.anomaly_scores.loc[self.anomaly_scores.index, 
                                            color_by_column]
            out = []
            for idx in clst.index:
                dat = clst.loc[idx].values
                out.append({'id': (str)(idx), 
                            'x': '{:f}'.format(dat[0]), 
                            'y': '{:f}'.format(dat[1]),
                            'opacity': '0.5', 
                            'color': '{:f}'.format(clst.loc[idx, 'color'])})
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
            anomaly_scores.sort_values(column_to_sort_by, inplace=True, 
                                       ascending=ascending)
        else:
            print("Requested column not in ml_scores dataframe")
