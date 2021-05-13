from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum, autoencoder
from astronomaly.feature_extraction import shape_features
from astronomaly.dimensionality_reduction import pca
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import tsne
from astronomaly.utils import utils
import os
import pandas as pd

data_dir = '/home/verlon/Documents/Data'

part = 'Part_40'

image_dir = os.path.join(data_dir,'Input',part,'Cutouts')
output_dir = os.path.join(data_dir,'Output',part, '')

catalogue = pd.read_csv(os.path.join(data_dir,'Input',part, part+'.csv'))[900000:1000000]

print(image_dir)
print(output_dir)


image_transform_function = [image_preprocessing.image_transform_scale,
                            image_preprocessing.image_transform_band_reorder,
                            image_preprocessing.image_transform_greyscale,
                            #image_preprocessing.image_band_addition,
                            image_preprocessing.image_transform_sigma_clipping,
                            # image_preprocessing.image_transform_inverse_sinh,
                            image_preprocessing.image_transform_scale,
                            #image_preprocessing.image_transform_cv2_resize,
                            ]

display_transform_function = [#image_preprocessing.image_transform_inverse_sinh,
                              image_preprocessing.image_transform_colour_correction,
                              image_preprocessing.image_transform_scale
                              ]

which_data = 'decals'
list_of_files = []
window_size = 32

band_prefixes = ['z-', 'r-', 'g-']
bands_rgb = {'r': 'z-', 'g': 'r-', 'b': 'g-'}
plot_cmap = 'hot'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

feature_method = 'ellipse'
dim_reduction = ''
extending_ellipse = True





def run_pipeline():
    """
    An example of the full astronomaly pipeline run on image data

    Parameters
    ----------
    image_dir : str
        Directory where images are located (can be a single fits file or 
        several)
    features : str, optional
        Which set of features to extract on the cutouts
    dim_reduct : str, optional
        Which dimensionality reduction algorithm to use (if any)
    anomaly_algo : str, optional
        Which anomaly detection algorithm to use

    Returns
    -------
    pipeline_dict : dictionary
        Dictionary containing all relevant data including cutouts, features 
        and anomaly scores

    """

    image_dataset = image_reader.ImageFitsDataset(
        directory=image_dir,
        list_of_files=list_of_files,
        window_size=window_size, output_dir=output_dir, plot_square=False,
        transform_function=image_transform_function,
        display_transform_function=display_transform_function,
        plot_cmap=plot_cmap,
        catalogue=catalogue,
        band_prefixes=band_prefixes,
        bands_rgb=bands_rgb
        )

    pipeline_ellipse = shape_features.EllipseFitFeatures(
        percentiles=[90, 80, 70, 60, 50,0],
        output_dir=output_dir, channel=0,
        extending_ellipse = extending_ellipse,
        force_rerun=True
        )
    
    features_original = pipeline_ellipse.run_on_dataset(image_dataset)

    features = features_original.copy()

    if extending_ellipse:
        filname = os.path.join(data_dir,'Open_Ellipses', part+'_9_ellipse_catalogue.csv')
        utils.create_ellipse_check_catalogue(image_dataset, features, filename=filname)

    features.drop(['Warning_Open_Ellipse','Recommended_Window_Size'], 1, inplace=True)
    #print(features)

    pipeline_scaler = scaling.FeatureScaler(force_rerun=True,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)

    features.to_csv(os.path.join(data_dir, 'Features', part+'_9_Features.csv'))
    print(len(features))