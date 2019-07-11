from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os

known_file_types = ['fits']

def read_images(directory='', list_of_files=[], fits_index=0):
    """
    Read in a set of images either from a directory or from a list of file paths (absolute)

    Parameters
    ----------
    directory : string, optional
        Path to files to read from
    list_of_files : list, optional
        List of files to read in (absolute path)
    fits_index : integer, optional
        If these are fits files, specifies which HDU object in the list to work with

    Returns
    -------
    pipeline_dict : dictionary
        A dictionary which contains the images in the "images" key

    """
    images = []

    if len(directory) != 0:
        fls = os.listdir(directory)
        fls.sort()
        for f in fls:
            extension = f.split('.')[-1]
            if extension in known_file_types:
                astro_img = AstroImage(os.path.join(directory, f), file_type=extension, fits_index=fits_index)
                images.append(astro_img)

    if len(list_of_files) != 0:
        for f in list_of_files:
            extension = f.split('.')[-1]
            if extension in known_file_types:
                astro_img = AstroImage(f, file_type=extension, fits_index=fits_index)
                images.append(astro_img)

    if len(images) == 0:
        print("No images found")


    pipeline_dict = {'images': images}
    return pipeline_dict


class AstroImage:
    def __init__(self, filename, file_type='fits', fits_index=0):
        """
        Lightweight wrapper for an astronomy image from a fits file

        Parameters
        ----------
        filename : string
            Filename of fits file to be read
        fits_index : integer
            Which HDU object in the list to work with

        """
        print('Reading image data from %s...' %filename)
        self.filename = filename
        self.file_type = file_type
        self.fits_index = fits_index

        if file_type == 'fits':
            try:
                self.hdul = fits.open(filename)

            except FileNotFoundError:
                print("Error: File", filename, "not found")
                raise FileNotFoundError


        self.name = self._strip_filename()
        self.image = self.hdul[fits_index].data
        self.metadata = dict(self.hdul[self.fits_index].header)
        self.coords = self._convert_to_world_coords()
        print('Done!')

    def _strip_filename(self):

        """
        Tiny utility function to make a nice formatted version of the image name from the input filename string

        Returns
        -------
        string
            Formatted file name

        """
        s1 = self.filename.split(os.path.sep)[-1]
        return s1.split('.')[0]

    def _convert_to_world_coords(self):
        """
        Converts pixels to (ra,dec) in degrees. This is much faster to do once per image than on-demand for individual
        cutouts.

        Returns
        -------
        np.array
            Nx2 array with RA,DEC pairs for every pixel in image
        """

        w = WCS(self.hdul[self.fits_index].header, naxis=2)
        coords = w.wcs_pix2world(np.vstack((np.arange(self.image.shape[0])[::-1], np.arange(self.image.shape[1])[::-1])).T, 0)
        return coords


