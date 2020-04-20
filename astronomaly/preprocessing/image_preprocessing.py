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

    mini = img[img != 0].min()
    maxi = img.max()
    offset = (maxi - mini) / 100

    if maxi == 0 and mini == 0:
        img = img + 0.01
    else:
        img = (img - mini) / (maxi - mini) + offset

    return np.log(img)


def image_transform_inverse_sinh(img):
    """
    Performs inverse hyperbolic sine transform on image

    Parameters
    ----------
    img : np.ndarray
        Input image (assumed float values)

    Returns
    -------
    np.ndarray
        Transformed image

    """
    if img.max() == 0:
        return img
    theta = 100 / img.max()

    return np.arcsinh(theta * img) / theta


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

    img[img < 0] = 0
    mini = img[img != 0].min()
    maxi = img.max()
    offset = (maxi - mini) / 10

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
    if img.min() == img.max():
        return img
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
