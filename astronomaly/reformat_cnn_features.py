import glob
import os
import logging
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

def raw_loc_to_clean_loc(raw_loc):
    return raw_loc.replace('_full_features_', '_full_cleaned_').replace('.csv', '.parquet')


def clean_feature_csv(raw_locs, image_format = 'png'):
        
    for loc in tqdm(raw_locs):
        logging.info('Reformatting {}'.format(loc))
        assert '_full_features_' in loc
        assert '_full_cleaned_' not in loc
        clean_loc = raw_loc_to_clean_loc(loc)
        if not os.path.isfile(clean_loc):

            features_df = pd.read_csv(loc)
            feature_cols = [col for col in features_df if col.startswith('feat')]

            for col in feature_cols:
                features_df[col] = features_df[col].apply(lambda x: float(x.replace('[', '').replace(']', '')))  # extract from list e.g. [0.1456] to 0.1456

            # features_df['png_loc'] = features_df['image_loc'].str.replace('/raid/scratch/walml/galaxy_zoo/gz2/png', '/media/walml/beta1/galaxy_zoo/gz2/png')
            features_df['filename'] = features_df['image_loc']

        #     assert all([os.path.isfile(x) for x in features_df['filename']])
            del features_df['image_loc']
            features_df['id_str'] = list(features_df['filename'].apply(lambda x: os.path.basename(x).split('.')[-2]))

        #         features_df = features_df[features_df['objid'].isin(safe_galaxies)]

            features_df.to_parquet(clean_loc)


def concat(clean_locs):
    data = []
    for loc in clean_locs:
        data.append(pd.read_parquet(loc))
    df = pd.concat(data)
    logging.info('Total galaxies in reformatted and concat df: {}'.format(len(df)))
    return df


def main(raw_search_str, clean_search_str, reformatted_parquet_loc):
        
    logging.info('Raw files: {}'.format(raw_search_str))
    logging.info('Reformatted files: {}'.format(clean_search_str))
    logging.info('Destination parquet:: {}'.format(reformatted_parquet_loc))

    raw_locs = glob.glob(raw_search_str)
    assert raw_locs
    logging.info('Raw csvs to reformat: {}'.format(len(raw_locs)))

    pool = Pool(processes=20)

    pbar = tqdm(total=len(raw_locs))
    for _ in pool.imap_unordered(clean_feature_csv, raw_locs):
        pbar.update()

    clean_locs = glob.glob(clean_search_str)
    assert clean_locs
    df = concat(clean_locs)
    print(df.head())
    df.to_parquet(reformatted_parquet_loc, index=False)



if __name__ == '__main__':

    # raw_search_str = '/media/walml/beta1/cnn_features/gz2/*_full_features_*.csv'
    # clean_search_str = '/media/walml/beta1/cnn_features/gz2/*_cleaned_*.parquet'
    # reformatted_parquet_loc = '/media/walml/beta1/cnn_features/gz2/cnn_features_concat.parquet'

    logging.basicConfig(level=logging.INFO)

    raw_search_str = '/share/nas/walml/repos/zoobot/data/results/dr5_color_full_features_*.csv'
    clean_search_str = raw_loc_to_clean_loc(raw_search_str)
    assert raw_search_str != clean_search_str
    reformatted_parquet_loc = os.path.join(os.path.dirname(raw_search_str), 'dr5_color_cnn_features_concat.parquet')

    main(raw_search_str, clean_search_str, reformatted_parquet_loc)
