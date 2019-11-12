from astronomaly.base.base_dataset import Dataset
import numpy as np
import pandas as pd


class RawFeatures(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.features = []
        self.labels = []

        print('Loading features...')
        for f in self.files:
            ext = f.split('.')[-1]
            feats = []
            labels = []

            if ext == 'npy':
                if 'labels' in f:
                    labels = np.load(f)
                    labels = pd.DataFrame(data=labels, 
                                          columns=['label'], dtype='int')
                else:
                    feats = np.load(f)
                    feats = pd.DataFrame(data=feats)

            elif ext == 'csv':
                if 'labels' in f:
                    labels = pd.read_csv(f)
                else:
                    feats = pd.read_csv(f)

            elif ext == 'parquet':
                if 'labels' in f:
                    labels = pd.read_parquet(f)
                else:
                    feats = pd.read_parquet(f)

            if len(feats) != 0:
                if len(self.features) == 0:
                    self.features = feats
                else:
                    self.features = pd.concat((self.features, feats))

            if len(labels) != 0:
                if len(self.labels) == 0:
                    self.labels = labels
                else:
                    self.labels = pd.concat((self.labels, labels))

        # Force string index because it's safer
        self.features.index = self.features.index.astype('str')
        self.labels.index = self.labels.index.astype('str')

        print('Done!')

        self.data_type = 'raw_features'

        if len(labels) != 0:
            self.metadata = self.labels
        else:
            self.metadata = pd.DataFrame(data=[], 
                                         index=list(self.features.index))

    def get_sample(self, idx):
        return self.features.loc[idx].values

    def get_display_data(self, idx):
        cols = list(self.features.columns)
        feats = self.features.loc[idx].values

        out_dict = {'categories': cols}
        out_dict['data'] = [[i, feats[i]] for i in range(len(feats))]
        return out_dict
