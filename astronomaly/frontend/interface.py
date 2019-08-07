from astronomaly.scripts import image_pipeline, light_curve_pipeline
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from astronomaly.preprocessing import image_preprocessing
import io
import time
import pandas as pd
import os

# Various functions that are called by the REST API in run.py

which_data = 'light_curve'

# pipeline_dict = pipeline.run_pipeline(image_dir='/home/michelle/BigData/Anomaly/GOODS_S/', features='psd', dim_reduct='pca')
# Image downloaded from here: https://archive.stsci.edu/pub/hlsp/goods/v2/
image_dir = '/home/michelle/BigData/Anomaly/'





dwf_dir = '/home/michelle/BigData/Anomaly/dwf_data/2015_01_CDF-S/'
light_curve_dir = os.path.join(dwf_dir, 'light_curves/', 'files/')
# features_file = os.path.join(dwf_dir,'2015_01_CDF-S_150114_feats_with_gaia_FULL.ascii')
features_file = os.path.join(dwf_dir,'2015_01_CDF-S_150115_feats_with_gaia_FULL.ascii')
# features_file = os.path.join(dwf_dir,'2015_01_CDF-S_150117_feats_with_gaia_FULL.ascii')

if which_data == 'image':
    features = 'psd2d'
    dim_reduct = 'pca'
    scaled = 'scaled'
    anomaly_algo = 'iforest'
    clustering = 'tsne'
    pipeline_dict = image_pipeline.run_pipeline(image_dir=image_dir, 
                                    features=features, dim_reduct=dim_reduct,
                                    scaled = scaled, 
                                    anomaly_algo=anomaly_algo, clustering=clustering)
else:
    features = 'from_file'
    dim_reduct = ''
    scaled = 'scaled'
    anomaly_algo = 'iforest'
    clustering = 'tsne'
    pipeline_dict = light_curve_pipeline.run_pipeline(light_curve_dir=light_curve_dir,
        features_file=features_file)


window_size_x= window_size_y = 128 #Change this to read from the dict

def sort_ml_scores(column_to_sort_by):
    print("sorting by"+column_to_sort_by)
    if column_to_sort_by in pipeline_dict['ml_scores'].columns:
        if column_to_sort_by == "iforest_score":
            ascending = True
        else:
            ascending = False
        pipeline_dict['ml_scores'].sort_values(column_to_sort_by, inplace=True, ascending=ascending)
    else:
        print("Requested column not in ml_scores dataframe")

def randomise_ml_scores():
    inds = np.random.permutation(pipeline_dict['ml_scores'].index)
    pipeline_dict['ml_scores'] = pipeline_dict['ml_scores'].loc[inds]


def convert_array_to_image(arr):
    fig = plt.figure(figsize=(1,1),dpi=window_size_x*4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(arr, cmap='hot')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)
    return output

def get_image_cutout(id):
    img = pipeline_dict['images'][0].image
    x0 = pipeline_dict['metadata'].loc[id, 'x']
    y0 = pipeline_dict['metadata'].loc[id, 'y']

    factor = 1.5
    xmin = (int)(x0-window_size_x*factor)
    xmax = (int)(x0+window_size_x*factor)
    ymin = (int)(y0-window_size_y*factor)
    ymax = (int)(y0+window_size_y*factor)

    xstart = max(xmin, 0)
    xend = min(xmax, img.shape[1])
    ystart = max(ymin, 0)
    yend = min(ymax, img.shape[0])
    tot_size_x = int(2*window_size_x*factor)
    tot_size_y = int(2*window_size_y*factor)
    cutout = np.zeros([tot_size_y, tot_size_x])
    cutout[ystart-ymin:tot_size_y-(ymax-yend), xstart-xmin:tot_size_x-(xmax-xend)] = img[ystart:yend, xstart:xend]
    cutout = np.nan_to_num(cutout)


    ### Read this transform from params in dict
    cutout = image_preprocessing.image_transform_log(cutout)

    return convert_array_to_image(cutout)

def read_lc_from_file(flpath, lower_mag=1, upper_mag=25):
    
    light_curve = pd.read_csv(flpath, delim_whitespace=True)
    return light_curve

def get_light_curve(id):
    print(id)
    ### Need to extend this to deal with other bands
    time_col = 'MJD'
    mag_col = 'g_mag'
    err_col = 'g_mag_err'

    out_dict = {}

    metadata = pipeline_dict['metadata']
    flpath = metadata[metadata.id==id]['filepath'].iloc[0]
    try:
        light_curve = read_lc_from_file(flpath)
        light_curve = light_curve[(1<light_curve[mag_col])&(light_curve[mag_col]<25)]
        light_curve['err_lower'] = light_curve[mag_col] - light_curve[err_col]
        light_curve['err_upper'] = light_curve[mag_col] + light_curve[err_col]
        
        out_dict['data'] = light_curve[[time_col, mag_col]].values.tolist()
        out_dict['errors'] = light_curve[[time_col, 'err_lower','err_upper']].values.tolist()
    
    except (pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError) as e:
        print('Error parsing file', flpath)
        print('Error message:')
        print(e)
        out_dict = {'data':[], 'errors':[]}

    return out_dict

def get_features(id):
    dat = pipeline_dict['features_'+features].loc[id]
    print(dat)
    out_dict = dict(zip(dat.coords['features'].values.astype('str'),dat.values))     
    return out_dict

def get_tsne_data(input_key, color_by_column=''):
    if input_key=='auto':
        input_key = 'features_%s' %(features)
        if len(dim_reduct) != 0:
            input_key += '_%s' %dim_reduct
        if len(scaled) != 0:
            input_key += '_%s' %scaled
        if len(clustering) != 0:
            input_key += '_%s' %clustering
    ts = pipeline_dict[input_key]
    dat = ts.data
    # out_str = '['
    # for i in range(len(ts.id))[:5]:
    #     out_str+="{{x:{:f}, y:{:f} }},".format(dat[i,0], dat[i,1])
    # out_str = out_str[:-1]
    # out_str += ']'
    # print(out_str)
    # return out_str
    if len(color_by_column)==0:
        cols = [0.5]*len(ts.id)
    else:
        ## This is pretty ugly, maybe storing as pd dataframes is better after all
        temp_df = pd.merge(pd.Series(list(ts.id.values),name='id'), pipeline_dict['ml_scores'], on='id')
        cols = temp_df[color_by_column].values

    out = []
    for i in range(len(ts.id)):
        out.append({'id':(str)(ts.id.values[i]), 'x':'{:f}'.format(dat[i,0]), 'y':'{:f}'.format(dat[i,1]),
        'opacity':'0.5', 'color':'{:f}'.format(cols[i])})
        # out.append({'x':dat[i,0], 'y':dat[i,1]})
    return out

def get_original_id_from_index(ind):
    this_ind =list(pipeline_dict['ml_scores'].id)[ind]
    return this_ind

def get_metadata(id, exclude_keywords=[], include_keywords=[]):
    id = str(id)
    meta_df = pipeline_dict['metadata']
    ml_df = pipeline_dict['ml_scores']
    df = meta_df[meta_df.id==id].merge(ml_df[ml_df.id==id], on='id')

    out_dict = {}
    if len(include_keywords) != 0:
        cols = include_keywords
    else:
        cols = df.columns 
    
    for col in cols:
        if col not in exclude_keywords:
            out_dict[col] = (str)(df.loc[0,col])
    return out_dict




