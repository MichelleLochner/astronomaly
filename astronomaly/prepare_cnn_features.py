import os
import glob
import pickle
from multiprocessing import Pool

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import IncrementalPCA


def clean_feature_csv(loc, image_format='png'):
    clean_loc = loc.replace('_features_', '_features_cleaned_').replace('.csv', '.parquet')
    if not os.path.isfile(clean_loc):
    
        features_df = pd.read_csv(loc)
        feature_cols = [col for col in features_df if col.startswith('feat')]

        for col in feature_cols:
            features_df[col] = features_df[col].apply(lambda x: float(x.replace('[', '').replace(']', '')))  # extract from list e.g. [0.1456] to 0.1456

        features_df['filename'] = features_df['image_loc'].str.replace('/raid/scratch/walml/galaxy_zoo/decals/png', '/media/walml/beta1/decals/png_native/dr5')

        del features_df['image_loc']
        features_df['objid'] = list(features_df['filename'].apply(lambda x: x.split('/')[-1].replace(f'.{image_format}', '')))

        # features_df = features_df[features_df['objid'].isin(safe_galaxies)]

        # must be indexed by the names of the png files for astronomaly to know which image has which features
        # cast to list to avoid pandas renaming index to image_loc - too clever for its own good

        features_df.to_parquet(clean_loc)

if __name__ == '__main__':

    sns.set_context('notebook')

    # # raw_csv_locs = glob.glob('/raid/scratch/walml/repos/zoobot/data/results/dr8_b0_full_features_*.csv')

    # # pool = Pool(processes=20)

    # # pbar = tqdm(total=len(raw_csv_locs))
    # # for _ in pool.imap_unordered(clean_feature_csv, raw_csv_locs):
    # #     pbar.update()

    # cleaned_locs = glob.glob('/raid/scratch/walml/repos/zoobot/data/results/dr8_b0_full_features_cleaned_*.parquet')
    # df = pd.concat([pd.read_parquet(loc) for loc in cleaned_locs]).sort_values('filename')
    # df.to_parquet('/raid/scratch/walml/repos/zoobot/data/results/dr8_b0_full_features_all.parquet', index=False)

    df = pd.read_parquet('/raid/scratch/walml/repos/zoobot/data/results/dr8_b0_full_features_all.parquet')
    
    print(len(df))
    # may allow all columns
    catalog = pd.read_parquet('/raid/scratch/walml/repos/download_DECaLS_images/working_dr8_master.parquet', columns=['png_loc', 'weighted_radius', 'ra', 'dec', 'dr8_id'])
    catalog['filename'] = catalog['png_loc']  # will use this for first merge w/ features, then use dr8_id or galaxy_id going forwards
    df = pd.merge(df, catalog, on='filename', how='inner').reset_index(drop=True)  # applies previous filters implicitly
    print(len(df))

    # this is the catalog whose index will match
    df.to_parquet('dr8_b0_full_features_and_safe_catalog.parquet', index=False)

    # TODO I think this has not been correctly filtered for bad images. Run the checks again, perhaps w/ decals downloader? Or check data release approach
    dr5_df = pd.read_parquet('dr5_b0_full_features_and_safe_catalog.parquet')
    print(dr5_df.head())
    dr5_df['estimated_radius'] = dr5_df['petro_th50']
    dr5_df['galaxy_id'] = dr5_df['iauname']

    dr8_df = pd.read_parquet('dr8_b0_full_features_and_safe_catalog.parquet')
    print(dr8_df.head())
    dr8_df['estimated_radius'] = dr8_df['weighted_radius']
    dr8_df['galaxy_id'] = dr8_df['dr8_id']

    df = pd.concat([dr5_df, dr8_df], axis=0).reset_index(drop=True)  # concat rowwise, some cols will have nans - but not png_loc or feature cols
    # important to reset index else index is not unique, would be like 0123...0123...


    feature_cols = [col for col in df.columns.values if col.startswith('feat')]
    not_feature_cols = [col for col in df.columns.values if not col.startswith('feat')]

    df[not_feature_cols].to_parquet('dr5_dr8_catalog_with_radius.parquet')
  
    # TODO rereun including ra/dec and check for duplicates/very close overlaps
    
    features = df[feature_cols].values

    # TODO add temporary shuffle and reset after fit_transform, as rows are not random as sky is not homogenously imaged
        
    for n_components in tqdm([2, 10, 30]):
        pca = IncrementalPCA(n_components=n_components, batch_size=20000)
        reduced_embed = pca.fit_transform(features)
        print(reduced_embed[:10])
        if np.isnan(reduced_embed).any():
            raise ValueError(f'embed is {np.isnan(reduced_embed).mean()} nan')
        embed_df = pd.DataFrame(data=reduced_embed, columns=['feat_{}_pca'.format(n) for n in range(n_components)])
        print(embed_df.head())
        embed_df_with_ids = pd.concat([df[['galaxy_id']], embed_df], axis=1)
        print(embed_df_with_ids.head())
        # includes both pca embedding and png_loc. replaces pickle files in older version
        embed_df_with_ids.to_parquet('dr5_8_b0_pca{}_and_safe_ids.parquet'.format(n_components), index=False)

        if n_components == 30:
            plt.plot(range(n_components), pca.explained_variance_ratio_)
            plt.xlabel('Nth Component')
            plt.ylabel('Explained Variance')
            plt.tight_layout()
            plt.savefig('explained_variance_all.pdf')
