from sklearn.manifold import TSNE
import numpy as np
import xarray

def make_tsne(pipeline_dict, input_key, output_key='', perplexity=30, max_objs=2000,
             sort_by_column=''):

    print('Computing t-SNE...')

    if len(output_key) == 0:
        output_key = input_key+'_tsne'
        # print(output_key)
    
    feats = pipeline_dict[input_key]

    if len(feats) > max_objs:
        if sort_by_column:
            ml_df = pipeline_dict['ml_scores'] #### Needs error checking
            inds = ml_df.sort_values(sort_by_column, ascending=False).id.values[:max_objs]
        else:
            inds = np.random.choice(feats.id, max_objs, replace=False)
        feats = feats.loc[inds]


    
    ts = TSNE(perplexity=perplexity)
    ts.fit(feats)
    print('Done!')

    fitted_tsne = ts.embedding_

    
    pipeline_dict[output_key] = xarray.DataArray(
        fitted_tsne, coords={'id': inds}, dims=['id', 'tsne'], name=output_key)

    pipeline_dict[input_key+'_tsne_object'] = ts

    return pipeline_dict