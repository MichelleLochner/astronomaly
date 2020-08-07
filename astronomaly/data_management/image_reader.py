from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os
import pandas as pd
import xarray
import matplotlib as mpl
import io
from skimage.transform import resize
import cv2
from astronomaly.base.base_dataset import Dataset
from astronomaly.base import logging_tools
mpl.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: E402, E501
import matplotlib.pyplot as plt  # noqa: E402


def convert_array_to_image(arr, plot_cmap='hot'):
    """
    Function to convert an array to a png image ready to be served on a web
    page.

    Parameters
    ----------
    arr : np.ndarray
        Input image

    Returns
    -------
    png image object
        Object ready to be passed directly to the frontend
    """
    with mpl.rc_context({'backend': 'Agg'}):
        fig = plt.figure(figsize=(1, 1), dpi=4 * arr.shape[1])
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(arr, cmap=plot_cmap)
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        plt.close(fig)
    return output


def apply_transform(cutout, transform_function):
    """
    Applies the transform function(s) given at initialisation to the image.

    Parameters
    ----------
    cutout : np.ndarray
        Cutout of image

    Returns
    -------
    np.ndarray
        Transformed cutout
    """
    if transform_function is not None:
        try:
            len(transform_function)
            new_cutout = cutout
            for f in transform_function:
                new_cutout = f(new_cutout)
            cutout = new_cutout
        except TypeError:  # Simple way to test if there's only one function
            cutout = transform_function(cutout)
    return cutout


class AstroImage:
    def __init__(self, filenames, file_type='fits', fits_index=0, name=''):
        """
        Lightweight wrapper for an astronomy image from a fits file

        Parameters
        ----------
        filenames : list of files
            Filename of fits file to be read. Can be length one if there's only
            one file or multiple if there are multiband images
        fits_index : integer
            Which HDU object in the list to work with

        """
        print('Reading image data from %s...' % filenames[0])
        self.filenames = filenames
        self.file_type = file_type
        self.metadata = {}
        self.coords = None
        self.fits_index = fits_index

        images = []
        try:
            for f in filenames:
                hdul = fits.open(f)
                image = self._get_image_data(hdul)
                images.append(image)
                metadata = dict(hdul[self.fits_index].header)
                self.metadata[f] = metadata
                coords = self._convert_to_world_coords(hdul, image)
                if self.coords is None:
                    self.coords = coords
                else:
                    if not (self.coords == coords).all():
                        print(
                            'Coordinates of different bands are not the same')
                        raise ValueError 

        except FileNotFoundError:
            print("Error: File", f, "not found")
            raise FileNotFoundError

        if len(images) > 1:
            # Should now be a 3d array with multiple channels
            self.image = np.dstack(images)
        else:
            self.image = images[0]  # Was just the one image

        if len(name) == 0:
            self.name = self._strip_filename()
        else:
            self.name = name

        print('Done!')

    def _get_image_data(self, hdul):
        """Returns the image data from a fits HDUlist object

        Parameters
        ----------
        hdul : fits.HDUlist
            HDUlist object returned by fits.open

        Returns
        -------
        np.array
            Image data
        """
        if self.fits_index is None:
            for i in range(len(hdul)):
                self.fits_index = i
                dat = hdul[self.fits_index].data
                if dat is not None:
                    image = dat
                    break
        else:
            image = hdul[self.fits_index].data

        if len(image.shape) > 2:
            image = np.squeeze(image)

        return image

    def _strip_filename(self):

        """
        Tiny utility function to make a nice formatted version of the image 
        name from the input filename string

        Returns
        -------
        string
            Formatted file name

        """
        s1 = self.filenames[0].split(os.path.sep)[-1]
        # extension = s1.split('.')[-1]
        return s1

    def _convert_to_world_coords(self, hdul, image):
        """
        Converts pixels to (ra,dec) in degrees. This is much faster to do once 
        per image than on-demand for individual cutouts.

        Parameters
        ----------
        hdul : fits.HDUlist
            HDUlist object returned by fits.open
        image : np.array
            The image data to determine the coordinates shape

        Returns
        -------
        np.array
            Nx2 array with RA,DEC pairs for every pixel in image
        """

        w = WCS(hdul[self.fits_index].header, naxis=2)
        coords = w.wcs_pix2world(np.vstack(
                                 (np.arange(image.shape[0])[::-1], 
                                  np.arange(image.shape[1])[::-1])).T, 0)
        return coords


class ImageDataset(Dataset):
    def __init__(self, fits_index=None, window_size=128, window_shift=None, 
                 display_image_size=128, band_prefixes=[], bands_rgb={},
                 transform_function=None, display_transform_function=None,
                 plot_square=False, catalogue=None,
                 plot_cmap='hot', **kwargs):
        """
        Read in a set of images either from a directory or from a list of file
        paths (absolute). Inherits from Dataset class.

        Parameters
        ----------
        filename : str
            If a single file (of any time) is to be read from, the path can be
            given using this kwarg. 
        directory : str
            A directory can be given instead of an explicit list of files. The
            child class will load all appropriate files in this directory.
        list_of_files : list
            Instead of the above, a list of files to be loaded can be
            explicitly given.
        output_dir : str
            The directory to save the log file and all outputs to. Defaults to
            './' 
        fits_index : integer, optional
            If these are fits files, specifies which HDU object in the list to
            work with
        window_size : int, tuple or list, optional
            The size of the cutout in pixels. If an integer is provided, the 
            cutouts will be square. Otherwise a list of 
            [window_size_x, window_size_y] is expected.
        window_shift : int, tuple or list, optional
            The size of the window shift in pixels. If the shift is less than 
            the window size, a sliding window is used to create cutouts. This 
            can be particularly useful for (for example) creating a training 
            set for an autoencoder. If an integer is provided, the shift will 
            be the same in both directions. Otherwise a list of
            [window_shift_x, window_shift_y] is expected.
        display_image_size : The size of the image to be displayed on the
            web page. If the image is smaller than this, it will be
            interpolated up to the higher number of pixels. If larger, it will
            be downsampled.
        band_prefixes : list
            Allows you to specify a prefix for an image which corresponds to a
            band identifier. This has to be a prefix and the rest of the image
            name must be identical in order for Astronomaly to detect these
            images should be stacked together. 
        bands_rgb : Dictionary
            Maps the input bands (in separate folders) to rgb values to allow
            false colour image plotting. Note that here you can only select
            three bands to plot although you can use as many bands as you like
            in band_prefixes. The dictionary should have 'r', 'g' and 'b' as
            keys with the band prefixes as values.
        transform_function : function or list, optional
            The transformation function or list of functions that will be 
            applied to each cutout. The function should take an input 2d array 
            (the cutout) and return an output 2d array. If a list is provided, 
            each function is applied in the order of the list.
        catalogue : pandas.DataFrame or similar
            A catalogue of the positions of sources around which cutouts will
            be extracted. Note that a cutout of size "window_size" will be
            extracted around these positions and must be the same for all
            sources. 
        plot_square : bool, optional
            If True this will add a white border indicating the boundaries of
            the original cutout when the image is displayed in the webapp.
        plot_cmap : str, optional
            The colormap with which to plot the image
        """

        super().__init__(fits_index=fits_index, window_size=window_size, 
                         window_shift=window_shift, 
                         display_image_size=display_image_size,
                         band_prefixes=band_prefixes, bands_rgb=bands_rgb,
                         transform_function=transform_function, 
                         display_transform_function=display_transform_function,
                         plot_square=plot_square, catalogue=catalogue, 
                         plot_cmap=plot_cmap,
                         **kwargs)
        self.known_file_types = ['fits', 'fits.fz', 'fits.gz']
        self.data_type = 'image'

        images = {}

        if len(band_prefixes) != 0:
            # Get the matching images in different bands
            bands_files = {}

            for p in band_prefixes:
                for f in self.files:
                    if p in f:
                        start_ind = f.find(p)
                        end_ind = start_ind + len(p)
                        flname = f[end_ind:]
                        if flname not in bands_files.keys():
                            bands_files[flname] = [f]
                        else:
                            bands_files[flname] += [f]

            for k in bands_files.keys():
                extension = k.split('.')[-1]
                # print(k, extension)
                if extension == 'fz' or extension == 'gz':
                    extension = '.'.join(k.split('.')[-2:])
                if extension in self.known_file_types:
                    astro_img = AstroImage(bands_files[k], file_type=extension, 
                                           fits_index=fits_index, name=k)
                    images[k] = astro_img
                    # print(astro_img.image.shape)

            # Also convert the rgb dictionary into an index dictionary
            # corresponding
            if len(bands_rgb) == 0:
                self.bands_rgb = {'r': 0, 'g': 1, 'b': 2}
            else:
                self.bands_rgb = {}
                for k in bands_rgb.keys():
                    band = bands_rgb[k]
                    ind = band_prefixes.index(band)
                    self.bands_rgb[k] = ind
        else:
            for f in self.files:
                extension = f.split('.')[-1]
                if extension == 'fz' or extension == 'gz':
                    extension = '.'.join(f.split('.')[-2:])
                if extension in self.known_file_types:
                    astro_img = AstroImage([f], file_type=extension, 
                                           fits_index=fits_index)
                    images[astro_img.name] = astro_img

        if len(list(images.keys())) == 0:
            print("No images found")

        try:
            self.window_size_x = window_size[0]
            self.window_size_y = window_size[1]
        except TypeError:
            self.window_size_x = window_size
            self.window_size_y = window_size

        # Allows sliding windows
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
        if display_transform_function is None:
            self.display_transform_function = transform_function
        else:
            self.display_transform_function = display_transform_function

        self.plot_square = plot_square
        self.plot_cmap = plot_cmap
        self.catalogue = catalogue
        self.display_image_size = display_image_size
        self.band_prefixes = band_prefixes

        self.metadata = pd.DataFrame(data=[])
        self.cutouts = pd.DataFrame(data=[])

        if self.catalogue is None:
            self.generate_cutouts()

        else:
            self.get_cutouts_from_catalogue()
        self.index = self.metadata.index.values

    def generate_cutouts(self):
        """
        Cuts up all images into cutouts of the same size. An optional transform
        function (or series of functions) can be supplied to provide local 
        transformations to the cutouts. A log transform is highly recommended 
        for high dynamic range astronomy images to highlight fainter objects.
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

            # Remember, numpy array index of [row, column] 
            # corresponds to [y, x]
            for j in range(self.window_size_x // 2, 
                           img.shape[1] - (int)(1.5 * self.window_size_x), 
                           self.window_shift_x):
                for i in range(self.window_size_y // 2, 
                               img.shape[0] - (int)(1.5 * self.window_size_y),
                               self.window_shift_y):

                    cutout = img[i:i + self.window_size_y, 
                                 j:j + self.window_size_x]

                    if not np.any(np.isnan(cutout)):
                        y0 = i + self.window_size_y // 2
                        x0 = j + self.window_size_x // 2
                        x_vals.append(x0)
                        y_vals.append(y0)
                        peak_flux.append(cutout.max())

                        ra.append(astro_img.coords[x0, 0])
                        dec.append(astro_img.coords[y0, 1])

                        original_image_names.append(astro_img.name)

                        cutout = apply_transform(cutout, 
                                                 self.transform_function)

                        cutouts.append(cutout)

        self.metadata = pd.DataFrame(data={
                                     'original_image': original_image_names,
                                     'x': x_vals, 
                                     'y': y_vals, 
                                     'ra': ra, 
                                     'dec': dec, 
                                     'peak_flux': peak_flux},
                                     index=np.array(np.arange(len(cutouts)), 
                                     dtype='str'))

        if len(cutouts[0].shape) > 2:
            dims = ['index', 'dim_1', 'dim_2', 'dim_3']
        else:
            dims = ['index', 'dim_1', 'dim_2'] 

        self.cutouts = xarray.DataArray(cutouts, 
                                        coords={'index': self.metadata.index}, 
                                        dims=dims)
        print('Done!')

    def get_cutouts_from_catalogue(self):
        """
        Generates cutouts using a provided catalogue
        """
        print('Generating cutouts from catalogue...')
        if 'original_image' not in self.catalogue.columns:
            if len(self.images) > 1:
                logging_tools.log("""If multiple fits images are used the
                                  original_image column must be provided in
                                  the catalogue to identify which image the 
                                  source belongs to.""", 
                                  level='ERROR')

                raise ValueError("Incorrect input supplied")

            else:
                self.catalogue['original_image'] = \
                    [list(self.images.keys())[0]] * len(self.catalogue)

        if 'objid' not in self.catalogue.columns:
            self.catalogue['objid'] = np.arange(len(self.catalogue))

        if 'peak_flux' not in self.catalogue.columns:
            self.catalogue['peak_flux'] = [np.NaN] * len(self.catalogue)

        cutouts = []

        for i in range(len(self.catalogue)):
            img_name = self.catalogue['original_image'][i]
            if img_name in self.images.keys():
                img = self.images[img_name].image
                x0 = int(self.catalogue['x'][i])
                y0 = int(self.catalogue['y'][i])
                xmin = x0 - self.window_size_x // 2
                xmax = x0 + self.window_size_x // 2
                ymin = y0 - self.window_size_y // 2
                ymax = y0 + self.window_size_y // 2

                if ymin < 0 or xmin < 0 or ymax > img.shape[0] \
                        or xmax > img.shape[1]:
                    self.catalogue.drop(i, inplace=True)

                else:
                    cutout = img[ymin:ymax, xmin:xmax]

                    if (cutout.max() == cutout.min()):
                        self.catalogue.drop(i, inplace=True)

                    else:
                        if np.isnan(self.catalogue['peak_flux'][i]):
                            self.catalogue['peak_flux'][i] = cutout.max()
                        cutout = apply_transform(cutout, 
                                                 self.transform_function)
                        cutouts.append(cutout)
            else:
                self.catalogue.drop(i, inplace=True)

        cols = ['original_image', 'x', 'y']

        if 'ra' in self.catalogue.columns:
            cols.append('ra')
        if 'dec' in self.catalogue.columns:
            cols.append('dec')
        if 'peak_flux' in self.catalogue.columns:
            cols.append('peak_flux')

        met = {}
        for c in cols:
            met[c] = self.catalogue[c].values

        the_index = np.array(self.catalogue['objid'].values, dtype='str')
        self.metadata = pd.DataFrame(met, index=the_index)

        if len(cutouts[0].shape) > 2:
            dims = ['index', 'dim_1', 'dim_2', 'dim_3']
        else:
            dims = ['index', 'dim_1', 'dim_2'] 
        self.cutouts = xarray.DataArray(cutouts, 
                                        coords={'index': the_index}, 
                                        dims=dims)

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

    def get_display_data(self, idx):
        """
        Returns a single instance of the dataset in a form that is ready to be
        displayed by the web front end.

        Parameters
        ----------
        idx : str
            Index (should be a string to avoid ambiguity)

        Returns
        -------
        png image object
            Object ready to be passed directly to the frontend
        """

        try:
            img_name = self.metadata.loc[idx, 'original_image']
        except KeyError:
            return None

        img = self.images[img_name].image
        x0 = self.metadata.loc[idx, 'x']
        y0 = self.metadata.loc[idx, 'y']

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

        if len(img.shape) > 2:
            shp = [tot_size_y, tot_size_x, img.shape[-1]]
        else:
            shp = [tot_size_y, tot_size_x]
        cutout = np.zeros(shp)
        # cutout[ystart - ymin:tot_size_y - (ymax - yend), 
        #        xstart - xmin:tot_size_x - (xmax - xend)] = img[ystart:yend, 
        #                                                        xstart:xend]
        cutout[ystart - ymin:yend - ymin, 
               xstart - xmin:xend - xmin] = img[ystart:yend, xstart:xend]
        cutout = np.nan_to_num(cutout)

        cutout = apply_transform(cutout, self.display_transform_function)

        if len(cutout.shape) > 2 and cutout.shape[-1] >= 3:
            new_cutout = np.zeros([cutout.shape[0], cutout.shape[1], 3])
            new_cutout[:, :, 0] = cutout[:, :, self.bands_rgb['r']]
            new_cutout[:, :, 1] = cutout[:, :, self.bands_rgb['g']]
            new_cutout[:, :, 2] = cutout[:, :, self.bands_rgb['b']]
            cutout = new_cutout

        if self.plot_square:
            offset_x = (tot_size_x - self.window_size_x) // 2
            offset_y = (tot_size_y - self.window_size_y) // 2
            x1 = offset_x
            x2 = tot_size_x - offset_x
            y1 = offset_y
            y2 = tot_size_y - offset_y

            mx = cutout.max()
            cutout[y1:y2, x1] = mx
            cutout[y1:y2, x2] = mx
            cutout[y1, x1:x2] = mx
            cutout[y2, x1:x2] = mx

        min_edge = min(cutout.shape[:2])
        max_edge = max(cutout.shape[:2])
        if max_edge != self.display_image_size:
            new_max = self.display_image_size
            new_min = int(min_edge * new_max / max_edge)
            if cutout.shape[0] <= cutout.shape[1]:
                new_shape = [new_min, new_max]
            else:
                new_shape = [new_max, new_min]
            if len(cutout.shape) > 2:
                new_shape.append(cutout.shape[-1])
            cutout = resize(cutout, new_shape, anti_aliasing=False)

        return convert_array_to_image(cutout, plot_cmap=self.plot_cmap)


class ImageThumbnailsDataset(Dataset):
    def __init__(self, display_image_size=128, transform_function=None, 
                 display_transform_function=None,
                 catalogue=None, additional_metadata=None, **kwargs):
        """
        Read in a set of images that have already been cut into thumbnails. 
        This would be uncommon with astronomical data but is needed to read a 
        dataset like galaxy zoo. Inherits from Dataset class.

        Parameters
        ----------
        filename : str
            If a single file (of any time) is to be read from, the path can be
            given using this kwarg. 
        directory : str
            A directory can be given instead of an explicit list of files. The
            child class will load all appropriate files in this directory.
        list_of_files : list
            Instead of the above, a list of files to be loaded can be
            explicitly given.
        output_dir : str
            The directory to save the log file and all outputs to. Defaults to
        display_image_size : The size of the image to be displayed on the
            web page. If the image is smaller than this, it will be
            interpolated up to the higher number of pixels. If larger, it will
            be downsampled.
        transform_function : function or list, optional
            The transformation function or list of functions that will be 
            applied to each cutout. The function should take an input 2d array 
            (the cutout) and return an output 2d array. If a list is provided, 
            each function is applied in the order of the list.
        catalogue : pandas.DataFrame or similar
            A catalogue of the positions of sources around which cutouts will
            be extracted. Note that a cutout of size "window_size" will be
            extracted around these positions and must be the same for all
            sources. 
        """

        super().__init__(transform_function=transform_function, 
                         display_image_size=128, catalogue=catalogue,
                         **kwargs)

        self.data_type = 'image'
        self.known_file_types = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']
        self.transform_function = transform_function
        if display_transform_function is None:
            self.display_transform_function = self.transform_function
        else:
            self.display_transform_function = display_transform_function
        self.display_image_size = display_image_size

        if catalogue is not None:
            if 'objid' in catalogue.columns:
                catalogue.set_index('objid')
            self.metadata = catalogue
        else:
            inds = []
            file_paths = []
            for f in self.files:
                extension = f.split('.')[-1]
                if extension in self.known_file_types:
                    inds.append(
                        f.split(os.path.sep)[-1][:-(len(extension) + 1)])
                    file_paths.append(f)
            self.metadata = pd.DataFrame(index=inds, 
                                         data={'filename': file_paths})

        self.index = self.metadata.index.values

        if additional_metadata is not None:
            self.metadata = self.metadata.join(additional_metadata)

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

        filename = self.metadata.loc[idx, 'filename']
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return apply_transform(img, self.transform_function)

    def get_display_data(self, idx):
        """
        Returns a single instance of the dataset in a form that is ready to be
        displayed by the web front end.

        Parameters
        ----------
        idx : str
            Index (should be a string to avoid ambiguity)

        Returns
        -------
        png image object
            Object ready to be passed directly to the frontend
        """

        filename = self.metadata.loc[idx, 'filename']
        cutout = cv2.imread(filename)
        cutout = cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)
        cutout = apply_transform(cutout, self.display_transform_function)

        min_edge = min(cutout.shape[:2])
        max_edge = max(cutout.shape[:2])
        if max_edge != self.display_image_size:
            new_max = self.display_image_size
            new_min = int(min_edge * new_max / max_edge)
            if cutout.shape[0] <= cutout.shape[1]:
                new_shape = [new_min, new_max]
            else:
                new_shape = [new_max, new_min]
            if len(cutout.shape) > 2:
                new_shape.append(cutout.shape[-1])
            cutout = resize(cutout, new_shape, anti_aliasing=False)

        return convert_array_to_image(cutout)
