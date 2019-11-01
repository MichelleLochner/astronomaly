from astronomaly.base.base_dataset import Dataset
import numpy as np
import pandas as pd


class RawFeatures(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.features = []

        for f in self.files:
            ext = f.split('.')[-1]

            if ext == 'npy':
                feats = np.load(f)
                feats = pd.DataFrame(data=feats)

            elif ext == 'csv':
                feats = pd.read_csv(f)

            elif ext == 'parquet':
                feats = pd.read_parquet(f)

            if len(self.features) == 0:
                self.features = feats
            else:
                self.features = pd.concat((self.features, feats))

        # Force string index because it's safer
        self.features.index = self.features.index.astype('str')

        self.data_type = 'raw_features'

        self.metadata = pd.DataFrame(data=[], index=list(self.features.index))

    def get_sample(self, idx):
        return self.features.loc[idx].values

    def get_display_data(self, idx):
        cols = list(self.features.columns)
        feats = self.features.loc[idx].values

        out_dict = {'categories':cols}
        out_dict['data'] = [[i, feats[i]] for i in range(len(feats))]
        return out_dict

