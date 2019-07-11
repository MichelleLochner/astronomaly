from astronomaly.scripts import pipeline
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from astronomaly.preprocessing import image_preprocessing
import io

# Various functions that are called by the REST API in run.py

pipeline_dict = pipeline.run_pipeline(image_dir='/home/michelle/BigData/Anomaly/GOODS_S/', features='psd', dim_reduct='pca')
# Image downloaded from here: https://archive.stsci.edu/pub/hlsp/goods/v2/

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

def get_original_id_from_index(ind):
    this_ind =list(pipeline_dict['ml_scores'].id)[ind]
    return this_ind
