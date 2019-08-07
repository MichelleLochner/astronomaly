import numpy as np
import pandas as pd
from skimage.transform import resize
import xarray


def image_transform_log(img):
    """
    Normalise and then perform log transform on image

    Parameters
    ----------
    img : np.ndarray
        Input image (assumed float values)

    Returns
    -------
    np.ndarray
        Transformed image

    """
    offset = 0.01
    mini = img.min()
    maxi = img.max()

    if maxi==0 and mini==0:
        img = img + offset
    else:
        img = (img-mini)/(maxi-mini) + offset
    return np.log(img)


def image_transform_scale(img):
    """
    Small function to normalise an image between 0 and 1. Useful for deep learning.

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Scaled image

    """
    return (img-img.min())/(img.max()-img.min())


def image_transform_resize(img, new_shape):
    """
    Resize an image to new dimensions (e.g. to feed into a deep learning network).
    Parameters
    ----------
    img : np.ndarray
        Input image
    new_shape : tuple
        Expected new shape for image

    Returns
    -------
    np.ndarray
        Reshaped image

    """
    return resize(img, new_shape, preserve_range=True)


def generate_cutouts(pipeline_dict, window_size=128, window_shift=None, transform_function=None,
                    input_key='images', output_key='cutouts'):
    """
    Cuts up all images into cutouts of the same size. An optional transform
    function (or series of functions) can be supplied to provide local transformations to the cutouts. A log transform
    is highly recommended for high dynamic range astronomy images to highlight fainter objects.

    Parameters
    ----------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    window_size : int, tuple or list, optional
        The size of the cutout in pixels. If an integer is provided, the cutouts will be square. Otherwise a list of
        [window_size_x, window_size_y] is expected.
    window_shift : int, tuple or list, optional
        The size of the window shift in pixels. If the shift is less than the window size, a sliding window is used to
         create cutouts. This can be particularly useful for (for example) creating a training set for an autoencoder.
         If an integer is provided, the shift will be the same in both directions. Otherwise a list of
        [window_shift_x, window_shift_y] is expected.
    transform_function : function or list, optional
        The transformation function or list of functions that will be applied to each cutout. The function should take
        an input 2d array (the cutout) and return an output 2d array. If a list is provided, each function is applied
        in the order of the list.
    input_key : str, optional
        The input key of pipeline_dict to run the function on.
    output_key : str, optional
        The output key of pipeline_dict

    Returns
    -------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores

    """
    print('Generating cutouts...')
    cutouts = []
    x_vals = []
    y_vals = []
    ra = []
    dec = []
    peak_flux = []
    original_image_names = []

    try:
        window_size_x = window_size[0]
        window_size_y = window_size[1]
    except TypeError:
        window_size_x = window_size
        window_size_y = window_size

    #We may in future want to allow sliding windows
    if window_shift is not None:
        try:
            window_shift_x = window_shift[0]
            window_shift_y = window_shift[1]
        except TypeError:
            window_shift_x = window_shift
            window_shift_y = window_shift
    else:
        window_shift_x = window_size_x
        window_shift_y = window_size_y


    for astro_img in pipeline_dict[input_key]:
        img = astro_img.image

        # Remember, numpy array index of [row, column] corresponds to [y, x]
        for j in range(window_size_x // 2, img.shape[1] - (int)(1.5 * window_size_x), window_shift_x):
            for i in range(window_size_y // 2, img.shape[0] - (int)(1.5 * window_size_y), window_shift_y):

                cutout = img[i:i + window_size_y, j:j + window_size_x]
                if not np.any(np.isnan(cutout)):
                    y0 = i + window_size_y // 2
                    x0 = j + window_size_x // 2
                    x_vals.append(x0)
                    y_vals.append(y0)
                    peak_flux.append(cutout.max())

                    ra.append(astro_img.coords[x0, 0])
                    dec.append(astro_img.coords[y0, 1])

                    original_image_names.append(astro_img.name)

                    if transform_function is not None:
                        try:
                            len(transform_function)
                            new_cutout = cutout
                            for f in transform_function:
                                new_cutout = f(new_cutout)
                            cutout = new_cutout
                        except TypeError:
                            cutout = transform_function(cutout)

                    cutouts.append(cutout)

    df = pd.DataFrame(data={'id':np.array(np.arange(len(cutouts)),dtype='str'), 
                            'original_image':original_image_names,
                            'x':x_vals, 'y':y_vals, 'ra':ra, 'dec':dec, 'peak_flux':peak_flux})

    pipeline_dict[output_key] = xarray.DataArray(cutouts, coords = {'id':df.id}, dims=['id','dim_1','dim_2'])

    if output_key == 'cutouts':
        metadata_key = 'metadata'
    else:
        metadata_key = 'metadata_' + output_key
    pipeline_dict[metadata_key] = df

    print('Done!')

    return pipeline_dict
