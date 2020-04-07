from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os
import pandas as pd
import xarray
import matplotlib as mpl
import io
from astronomaly.base.base_dataset import Dataset
from astronomaly.base import logging_tools
mpl.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: E402, E501
import matplotlib.pyplot as plt  # noqa: E402


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
        print('Reading image data from %s...' % filename)
        self.filename = filename
        self.file_type = file_type
        self.fits_index = fits_index

        try:
            self.hdul = fits.open(filename)

        except FileNotFoundError:
            print("Error: File", filename, "not found")
            raise FileNotFoundError

        self.name = self._strip_filename()
        if self.fits_index is None:
            for i in range(len(self.hdul)):
                self.fits_index = i
                dat = self.hdul[self.fits_index].data
                if dat is not None:
                    self.image = dat
                    break
        else:
            self.image = self.hdul[self.fits_index].data
        if len(self.image.shape) > 2:
            self.image = np.squeeze(self.image)
        self.metadata = dict(self.hdul[self.fits_index].header)
        self.coords = self._convert_to_world_coords()
        print('Done!')

    def _strip_filename(self):

        """
        Tiny utility function to make a nice formatted version of the image 
        name from the input filename string

        Returns
        -------
        string
            Formatted file name

        """
        s1 = self.filename.split(os.path.sep)[-1]
        extension = s1.split('.')[-1]
        return s1.split('.' + extension)[0]

    def _convert_to_world_coords(self):
        """
        Converts pixels to (ra,dec) in degrees. This is much faster to do once 
        per image than on-demand for individual cutouts.

        Returns
        -------
        np.array
            Nx2 array with RA,DEC pairs for every pixel in image
        """

        w = WCS(self.hdul[self.fits_index].header, naxis=2)
        coords = w.wcs_pix2world(np.vstack(
                                 (np.arange(self.image.shape[0])[::-1], 
                                  np.arange(self.image.shape[1])[::-1])).T, 0)
        return coords


class ImageDataset(Dataset):
    def __init__(self, fits_index=None, window_size=128, window_shift=None, 
                 transform_function=None, plot_square=False, catalogue=None,
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
                         transform_function=transform_function, 
                         plot_square=plot_square, plot_cmap=plot_cmap, 
                         **kwargs)
        self.known_file_types = ['fits', 'fits.fz']
        self.data_type = 'image'

        images = {}
        for f in self.files:
            extension = f.split('.')[-1]
            if extension == 'fz':
                extension = '.'.join(f.split('.')[-2:])
            if extension in self.known_file_types:
                astro_img = AstroImage(f, file_type=extension, 
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
        self.plot_square = plot_square
        self.plot_cmap = plot_cmap
        self.catalogue = catalogue

        self.metadata = pd.DataFrame(data=[])
        self.cutouts = pd.DataFrame(data=[])

        if self.catalogue is None:
            self.generate_cutouts()

        else:
            self.get_cutouts_from_catalogue()
        self.index = self.metadata.index.values

    def transform(self, cutout):
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

                        cutout = self.transform(cutout)

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

        self.cutouts = xarray.DataArray(cutouts, 
                                        coords={'index': self.metadata.index}, 
                                        dims=['index', 'dim_1', 'dim_2'])

        print('Done!')

    def get_cutouts_from_catalogue(self):
        """
        Generates cutouts using a provided catalogue
        """
        print('Generating cutouts from catalogue...')
        if 'original_image' not in self.catalogue.columns:
            if len(self.images) > 1:
                logging_tools.log('If multiple fits images are used the \
                                  original_image column must be provided in \
                                  the catalogue to identify which image the \
                                  source belongs to.', 
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
            x0 = int(self.catalogue['x'][i])
            y0 = int(self.catalogue['y'][i])
            xmin = x0 - self.window_size_x
            xmax = x0 + self.window_size_x
            ymin = y0 - self.window_size_y
            ymax = y0 + self.window_size_y

            img = self.images[self.catalogue['original_image'][i]].image

            if ymin < 0 or xmin < 0 or ymax > img.shape[0] \
                    or xmax > img.shape[1]:
                self.catalogue.drop(i, inplace=True)

            else:
                cutout = img[ymin:ymax, xmin:xmax]

                if np.isnan(self.catalogue['peak_flux'][i]):
                    self.catalogue['peak_flux'][i] = cutout.max()

                cutout = self.transform(cutout)
                cutouts.append(cutout)

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
        print(met, self.metadata)

        self.cutouts = xarray.DataArray(cutouts, 
                                        coords={'index': the_index}, 
                                        dims=['index', 'dim_1', 'dim_2'])

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
            fig = plt.figure(figsize=(1, 1), dpi=self.window_size_x * 4)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(arr, cmap=self.plot_cmap)
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            plt.close(fig)
        return output

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
        print('GET DISPLAY DATA')
        print(idx, type(idx))
        print(idx in self.metadata.index)
        print(self.metadata.loc[idx, 'x'])
        try:
            img_name = self.metadata.loc[idx, 'original_image']
        except KeyError:
            print('KEY ERROR')
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

        cutout = np.zeros([tot_size_y, tot_size_x])
        cutout[ystart - ymin:tot_size_y - (ymax - yend), 
               xstart - xmin:tot_size_x - (xmax - xend)] = img[ystart:yend, 
                                                               xstart:xend]
        cutout = np.nan_to_num(cutout)

        cutout = self.transform(cutout)

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

        return self.convert_array_to_image(cutout)
