import numpy as np
from skimage.transform import resize


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

    if maxi == 0 and mini == 0:
        img = img + offset
    else:
        img = (img - mini) / (maxi - mini) + offset

    return np.log(img)


def image_transform_root(img):
    """
    Normalise and then perform square root transform on image

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

    if maxi == 0 and mini == 0:
        img = img + offset
    else:
        img = (img - mini) / (maxi - mini) + offset

    return np.sqrt(img)

 
def image_transform_scale(img):
    """
    Small function to normalise an image between 0 and 1. Useful for deep 
    learning.

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Scaled image

    """
    return (img - img.min()) / (img.max() - img.min())


def image_transform_resize(img, new_shape):
    """
    Resize an image to new dimensions (e.g. to feed into a deep learning 
    network).

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
