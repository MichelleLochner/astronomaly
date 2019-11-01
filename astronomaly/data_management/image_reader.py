from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os
import pandas as pd
import xarray
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import io
from astronomaly.base.base_dataset import Dataset


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
        if len(self.image.shape)>2:
            self.image = np.squeeze(self.image)
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
        extension = s1.split('.')[-1]
        return s1.split('.'+extension)[0]

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

class ImageDataset(Dataset):
    def __init__(self, directory='', list_of_files=[], fits_index=0, window_size=128,
                 window_shift=None, transform_function=None, **kwargs):
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
        """

        super().__init__(directory=directory, list_of_files=list_of_files, fits_index=fits_index,
                       window_size=window_size, window_shift=window_shift, transform_function=transform_function, **kwargs)
        self.known_file_types = ['fits']
        self.data_type = 'image'


        images = {}
        for f in self.files:
            extension = f.split('.')[-1]
            if extension in self.known_file_types:
                astro_img = AstroImage(f, file_type=extension, fits_index=fits_index)
                images[astro_img.name] = astro_img

        if len(list(images.keys())) == 0:
            print("No images found")

        try:
            self.window_size_x = window_size[0]
            self.window_size_y = window_size[1]
        except TypeError:
            self.window_size_x = window_size
            self.window_size_y = window_size

        # We may in future want to allow sliding windows
        if window_shift is not None:
            try:
                self.window_shift_x = window_shift[0]
                self.window_shift_y = window_shift[1]
            except TypeError:
                self.window_shift_x = window_shift
                self.window_shift_y = window_shift
        else:
            self.window_shift_x = self.window_size_x
            self.window_shift_y = self.window_size_y

        self.images = images
        self.transform_function = transform_function

        self.metadata = pd.DataFrame(data=[])
        self.cutouts = pd.DataFrame(data=[])

        self.generate_cutouts()
        self.index = self.metadata.index.values

    def transform(self, cutout):
        if self.transform_function is not None:
            try:
                len(self.transform_function)
                new_cutout = cutout
                for f in self.transform_function:
                    new_cutout = f(new_cutout)
                cutout = new_cutout
            except TypeError:
                cutout = self.transform_function(cutout)
        return cutout

    def generate_cutouts(self):
        """
        Cuts up all images into cutouts of the same size. An optional transform
        function (or series of functions) can be supplied to provide local transformations to the cutouts. A log transform
        is highly recommended for high dynamic range astronomy images to highlight fainter objects.
        """
        print('Generating cutouts...')
        cutouts = []
        x_vals = []
        y_vals = []
        ra = []
        dec = []
        peak_flux = []
        original_image_names = []



        for image_name in list(self.images.keys()):
            astro_img = self.images[image_name]
            img = astro_img.image


            # Remember, numpy array index of [row, column] corresponds to [y, x]
            for j in range(self.window_size_x // 2, img.shape[1] - (int)(1.5 * self.window_size_x), self.window_shift_x):
                for i in range(self.window_size_y // 2, img.shape[0] - (int)(1.5 * self.window_size_y), self.window_shift_y):

                    cutout = img[i:i + self.window_size_y, j:j + self.window_size_x]
                    if not np.any(np.isnan(cutout)):
                        y0 = i + self.window_size_y // 2
                        x0 = j + self.window_size_x // 2
                        x_vals.append(x0)
                        y_vals.append(y0)
                        peak_flux.append(cutout.max())

                        ra.append(astro_img.coords[x0, 0])
                        dec.append(astro_img.coords[y0, 1])

                        original_image_names.append(astro_img.name)

                        cutout = self.transform(cutout)

                        cutouts.append(cutout)

        self.metadata = pd.DataFrame(data={'original_image': original_image_names,
                                'x': x_vals, 'y': y_vals, 'ra': ra, 'dec': dec, 'peak_flux': peak_flux},
                                     index=np.array(np.arange(len(cutouts)), dtype='str'))

        self.cutouts = xarray.DataArray(cutouts, coords={'index': self.metadata.index}, dims=['index', 'dim_1', 'dim_2'])

        print('Done!')

    def get_sample(self, idx):
        """
        Returns the data for a single sample in the dataset as indexed by idx.

        Parameters
        ----------
        idx : string
            Index of sample

        Returns
        -------
        nd.array
            Array of image cutout
        """
        return self.cutouts.loc[idx].values

    def convert_array_to_image(self, arr):
        with mpl.rc_context({'backend': 'Agg'}):
            fig = plt.figure(figsize=(1, 1), dpi=self.window_size_x * 4)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(arr, cmap='hot')
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            plt.close(fig)
        return output

    def get_display_data(self, idx):
        try:
            img_name = self.metadata.loc[idx, 'original_image']
        except KeyError:
            return None

        img = self.images[img_name].image
        x0 = self.metadata.loc[idx,'x']
        y0 = self.metadata.loc[idx,'y']

        factor = 1.5
        xmin = (int)(x0 - self.window_size_x * factor)
        xmax = (int)(x0 + self.window_size_x * factor)
        ymin = (int)(y0 - self.window_size_y * factor)
        ymax = (int)(y0 + self.window_size_y * factor)

        xstart = max(xmin, 0)
        xend = min(xmax, img.shape[1])
        ystart = max(ymin, 0)
        yend = min(ymax, img.shape[0])
        tot_size_x = int(2 * self.window_size_x * factor)
        tot_size_y = int(2 * self.window_size_y * factor)
        cutout = np.zeros([tot_size_y, tot_size_x])
        cutout[ystart - ymin:tot_size_y - (ymax - yend), xstart - xmin:tot_size_x - (xmax - xend)] = img[
                                                                                                     ystart:yend,
                                                                                                     xstart:xend]
        cutout = np.nan_to_num(cutout)

        ### Read this transform from params in dict
        cutout = self.transform(cutout)
        offset_x = (tot_size_x-self.window_size_x)//2
        offset_y = (tot_size_y-self.window_size_y)//2
        x1 = offset_x
        x2 = tot_size_x - offset_x
        y1 = offset_y
        y2 = tot_size_y - offset_y

        # mx = cutout.max()
        # cutout[y1:y2,x1] = mx
        # cutout[y1:y2,x2] = mx
        # cutout[y1,x1:x2] = mx
        # cutout[y2,x1:x2] = mx

        return self.convert_array_to_image(cutout)



# def read_images(directory='', list_of_files=[], fits_index=0):
#     """
#     Read in a set of images either from a directory or from a list of file paths (absolute)
#
#     Parameters
#     ----------
#     directory : string, optional
#         Path to files to read from
#     list_of_files : list, optional
#         List of files to read in (absolute path)
#     fits_index : integer, optional
#         If these are fits files, specifies which HDU object in the list to work with
#
#     Returns
#     -------
#     pipeline_dict : dictionary
#         A dictionary which contains the images in the "images" key
#
#     """
#     images = []
#
#     if len(list_of_files) != 0 and len(directory)==0:
#         # Assume the list of files are absolute paths
#         fls = list_of_files
#     elif len(list_of_files) != 0 and len(directory)!=0:
#         # Assume the list of files are relative paths to directory
#         fls = list_of_files
#         fls = [os.path.join(directory, f) for f in fls]
#     elif len(directory) != 0:
#         #Assume directory contains all the files we need
#         fls = os.listdir(directory)
#         fls.sort()
#         fls = [os.path.join(directory, f) for f in fls]
#     else:
#         fls = []
#
#     for f in fls:
#         extension = f.split('.')[-1]
#         if extension in known_file_types:
#             astro_img = AstroImage(f, file_type=extension, fits_index=fits_index)
#             images.append(astro_img)
#
#     if len(images) == 0:
#         print("No images found")
#
#
#     pipeline_dict = {'images': images}
#     return pipeline_dict

def read_cutouts_from_file(filename, file_type='npy', output_key='cutouts'):
    if file_type == 'npy':
        cutouts = np.load(filename, mmap_mode='r')
        # cutouts = cutouts[:10000] 

    df = pd.DataFrame(data={'id':np.array(np.arange(len(cutouts)),dtype='str')})
    pipeline_dict = {'metadata':df}
    pipeline_dict[output_key] = xarray.DataArray(cutouts, coords = {'id':df.id}, dims=['id','dim_1','dim_2'])
    return pipeline_dict

# def read_cutouts_from_directory(directory, file_type='jpg', output_key='cutouts'):
#     ids = []
#     cutouts = []
#     if file_type == 'jpg':
#
#
#     df = pd.DataFrame(data={'id':np.array(np.arange(len(cutouts)),dtype='str')})
#     pipeline_dict = {'metadata':df}
#     pipeline_dict[output_key] = xarray.DataArray(cutouts, coords = {'id':df.id}, dims=['id','dim_1','dim_2'])
#     return pipeline_dict




