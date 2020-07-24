import numpy as np
from skimage.transform import resize
import cv2
from astropy.stats import sigma_clipped_stats


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


def image_transform_crop(img, new_shape=[160, 160]):
    """
    Crops an image to new dimensions (assumes you want to keep the centre)

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
    delt_0 = (img.shape[0] - new_shape[0]) // 2
    delt_1 = (img.shape[1] - new_shape[1]) // 2
    return img[delt_0:img.shape[0] - delt_0, delt_1:img.shape[1] - delt_1]


def image_transform_gaussian_window(img, width=2.5):
    """
    Applies a Gaussian window of a given width to the image. This has the
    effect of downweighting possibly interfering objects near the edge of the
    image. 

    Parameters
    ----------
    img : np.ndarray
        Input image
    width : float, optional
        The standard deviation of the Gaussian. The Gaussian is evaluated on a
        grid from -5 to 5 so a width=1 corresponds to a unit Gaussian. The
        width of the Gaussian will appear to be around 1/5 of the image, which
        would be fairly aggressive downweighting of outlying sources.

    Returns
    -------
    np.ndarray
        Windowed image

    """

    xvals = np.linspace(-5, 5, img.shape[0])
    yvals = np.linspace(-5, 5, img.shape[1])
    X, Y = np.meshgrid(xvals, yvals)
    Z = 1 / np.sqrt(width) / 2 * np.exp(-(X**2 + Y**2) / 2 / width**2)

    if len(img.shape) == 2:  # Only a single channel image
        return img * Z
    else:
        new_img = np.zeros_like(img)
        for i in range(img.shape[-1]):
            new_img[:, :, i] = img[:, :, i] * Z
        return new_img


def image_transform_sigma_clipping(img, sigma=3, central=True):
    """
    Applies sigma clipping, fits contours and

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray

    """
    if len(img.shape) > 2:
        im = img[:, :, 0]
    else:
        im = img
    mean, median, std = sigma_clipped_stats(im, sigma=sigma)
    thresh = std + median
    img_bin = np.zeros(im.shape, dtype=np.uint8)

    img_bin[im <= thresh] = 0
    img_bin[im > thresh] = 1

    contours, hierarchy = cv2.findContours(img_bin, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)

    x0 = img.shape[0]//2
    y0 = img.shape[1]//2
    for c in contours:
        if cv2.pointPolygonTest(c, (x0, y0), False) == 1:
            break

    contour_mask = np.zeros_like(img)
    cv2.drawContours(contour_mask, [c], 0, (1, 1, 1), -1)

    new_img = np.zeros_like(img)
    new_img[contour_mask == 1] = img[contour_mask == 1]

    return new_img
   