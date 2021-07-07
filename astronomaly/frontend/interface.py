import numpy as np
import os
import importlib
import sys


class Controller:
    def __init__(self, pipeline_file):
        """
        This is the main controller for the interface between the Python
        backend and the JavaScript frontend. The Controller is passed a python
        file, which must contain a "run_pipeline" function and return a
        dictionary. The Controller consists of various functions which get
        called by the front end asking for things like data to plot, metadata,
        anomaly scores etc.

        Parameters
        ----------
        pipeline_file : str
            The script to run Astronomaly (see the "scripts" folder for 
            examples)
        """

        self.dataset = None
        self.features = None
        self.anomaly_scores = None
        self.visualisation = None
        self.module_name = None
        self.active_learning = None
        self.current_index = 0  # Index in the anomalies list

        self.set_pipeline_script(pipeline_file)

    def run_pipeline(self):
        """
        Runs (or reruns) the pipeline. Reimports the pipeline script so changes
        are reflected.
        """
        pipeline_script = importlib.import_module(self.module_name)
        print('Running pipeline from', self.module_name + '.py')
        pipeline_dict = pipeline_script.run_pipeline()

        # ***** Add some try catches here

        self.dataset = pipeline_dict['dataset']
        self.features = pipeline_dict['features']
        self.anomaly_scores = pipeline_dict['anomaly_scores']
        if 'visualisation' in list(pipeline_dict.keys()):
            self.visualisation = pipeline_dict['visualisation']
        if 'active_learning' in list(pipeline_dict.keys()):
            self.active_learning = pipeline_dict['active_learning']

    def get_data_type(self):
        return self.dataset.data_type

    def set_pipeline_script(self, pipeline_file):
        """
        Allows the changing of the input pipeline file.

        Parameters
        ----------
        pipeline_file : str
            New pipeline file
        """

        module_name = pipeline_file.split(os.path.sep)[-1]
        pth = pipeline_file.replace(module_name, '')
        module_name = module_name.split('.')[0]

        self.module_name = module_name
        sys.path.append(pth)  # Allows importing the module from anywhere

    def get_display_data(self, idx):
        """
        Simply calls the underlying Dataset's function to return display data.
        """

        try:
            return self.dataset.get_display_data(idx)
        except KeyError:
            return None

    def get_features(self, idx):
        """
        Returns the features of instance given by index idx.
        """
        try:
            out_dict = dict(zip(self.features.columns.astype('str'), 
                                self.features.loc[idx].values))
            for key in list(out_dict.keys()):
                try:
                    formatted_val = '%.3g' % out_dict[key]
                    out_dict[key] = formatted_val
                except TypeError:  # Probably a string already
                    pass
            return out_dict
        except KeyError:
            return {}

    def set_human_label(self, idx, label):
        """
        Sets the human-assigned score to an instance. Creates the column
        "human_label" if necessary in the anomaly_scores dataframe.

        Parameters
        ----------
        idx : str
            Index of instance
        label : int
            Human-assigned label
        """
        
        ml_df = self.anomaly_scores
        if 'human_label' not in ml_df.columns:
            ml_df['human_label'] = [-1] * len(ml_df)
        ml_df.loc[idx, 'human_label'] = label
        ml_df = ml_df.astype({'human_label': 'int'})

        self.active_learning.save(
            ml_df, os.path.join(self.active_learning.output_dir, 
                                'ml_scores.csv'), file_format='csv')
                                    
    def run_active_learning(self):
        """
        Runs the selected active learning algorithm.
        """
        has_no_labels = 'human_label' not in self.anomaly_scores.columns
        labels_unset = np.sum(self.anomaly_scores['human_label'] != -1) == 0
        if has_no_labels or labels_unset:
            print("Active learning requested but no training labels "
                  "have been applied.")
            return "failed"
        else:
            pipeline_active_learning = self.active_learning
            features_with_labels = \
                pipeline_active_learning.combine_data_frames(
                    self.features, self.anomaly_scores)
            # print(features_with_labels)
            scores = pipeline_active_learning.run(features_with_labels)
            self.anomaly_scores['trained_score'] = scores
            return "success"

    def delete_labels(self):
        """
        Allows the user to delete all the labels they've applied and start 
        again
        """
        print('Delete labels called')
        if 'human_label' in self.anomaly_scores.columns:
            self.anomaly_scores['human_label'] = -1
        print('All user-applied labels have been reset to -1 (i.e. deleted)')

    def get_visualisation_data(self, color_by_column=''):
        """
        Returns the data for the visualisation plot in the correct json format.

        Parameters
        ----------
        color_by_column : str, optional
            If given, the points on the plot will be coloured by this column so
            for instance, more anomalous objects are brighter.

        Returns
        -------
        dict
            Formatting visualisation plot data
        """
        clst = self.visualisation
        if clst is not None:
            if len(color_by_column) == 0:
                cols = [0.5] * len(clst)
                clst['color'] = cols
            else:
                clst['color'] = \
                    self.anomaly_scores.loc[clst.index, 
                                            color_by_column]
            out = []
            clst.sort_values('color')
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
        """
        The frontend iterates through an ordered list that can change depending
        on the algorithm selected. This function returns the actual index of an
        instance (which might be 'obj2487' or simply '1') when given an array
        index. 

        Parameters
        ----------
        ind : int
            The position in an array

        Returns
        -------
        str
            The actual object id
        """
        this_ind = list(self.anomaly_scores.index)[ind]
        return this_ind

    def get_metadata(self, idx, exclude_keywords=[], include_keywords=[]):
        """
        Returns the metadata for an instance in a format ready for display.

        Parameters
        ----------
        idx : str
            Index of the object
        exclude_keywords : list, optional
            Any keywords to exclude being displayed
        include_keywords : list, optional
            Any keywords that should be displayed

        Returns
        -------
        dict
            Display-ready metadata
        """
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
                    out_dict[col] = meta_df.loc[idx, col]
            for col in ml_df.columns:
                if col not in exclude_keywords:
                    out_dict[col] = ml_df.loc[idx, col]

            for key in (list)(out_dict.keys()):
                try:
                    formatted_val = '%.3g' % out_dict[key]
                    out_dict[key] = formatted_val
                except TypeError:  # Probably a string already
                    pass
            return out_dict
        except KeyError:
            return {}

    def randomise_ml_scores(self):
        """
        Returns the anomaly scores in a random order
        """
        inds = np.random.permutation(self.anomaly_scores.index)
        self.anomaly_scores = self.anomaly_scores.loc[inds]

    def sort_ml_scores(self, column_to_sort_by='score'):
        """
        Returns the anomaly scores sorted by a particular column.
        """
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

    def get_max_id(self):
        return len(self.anomaly_scores)

    def clean_up(self):
        self.dataset.clean_up()
