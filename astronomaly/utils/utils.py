import matplotlib.pyplot as plt
import astropy
import os
import pandas as pd
import numpy as np
import xlsxwriter
from PIL import Image


def convert_pybdsf_catalogue(catalogue_file, image_file,
                             remove_point_sources=False,
                             merge_islands=False,
                             read_csv_kwargs={},
                             colnames={}):
    """
    Converts a pybdsf fits file to a pandas dataframe to be given
    directly to an ImageDataset object.

    Parameters
    ----------
    catalogue_files : string
        Pybdsf catalogue in fits table format 
    image_file:
        The image corresponding to this catalogue (to extract pixel information
        and naming information)
    remove_point_sources: bool, optional
        If true will remove all sources with an S_Code of 'S'
    merge_islands: bool, optional
        If true, will locate all sources belonging to a particular island and
        merge them, maintaining only the brightest source
    read_csv_kwargs: dict, optional
        Will pass these directly to panda's read_csv function to allow reading
        in of a variety of file structures (e.g. different delimiters)
    colnames: dict, optional
        Allows you to choose the column names for "source_identifier" (which
        column to use to identify the source), "Isl_id", "Peak_flux" and 
        "S_Code" (if
        remove_point_sources is true)
    """

    if 'Peak_flux' not in colnames:
        colnames['Peak_flux'] = 'Peak_flux'
    if 'S_Code' not in colnames:
        colnames['S_Code'] = 'S_Code'
    if 'source_identifier' not in colnames:
        colnames['source_identifier'] = 'Source_id'
    if 'Isl_id' not in colnames:
        colnames['Isl_id'] = 'Isl_id'

    if 'csv' in catalogue_file:
        catalogue = pd.read_csv(catalogue_file, **read_csv_kwargs)
        cols = list(catalogue.columns)
        for i in range(len(cols)):
            cols[i] = cols[i].strip()
            cols[i] = cols[i].strip('#')
        catalogue.columns = cols
    else:
        dat = astropy.table.Table(astropy.io.fits.getdata(catalogue_file))
        catalogue = dat.to_pandas()

    if remove_point_sources:
        catalogue = catalogue[catalogue[colnames['S_Code']] != 'S']

    if merge_islands:
        inds = []
        for isl in np.unique(catalogue[colnames['Isl_id']]):
            msk = catalogue[colnames['Isl_id']] == isl
            selection = catalogue[msk][colnames['Peak_flux']]
            ind = catalogue[msk].index[selection.argmax()]
            inds.append(ind)
        catalogue = catalogue.loc[inds]

    hdul = astropy.io.fits.open(image_file)
    original_image = image_file.split(os.path.sep)[-1]

    w = astropy.wcs.WCS(hdul[0].header, naxis=2)

    x, y = w.wcs_world2pix(np.array(catalogue.RA), np.array(catalogue.DEC), 1)

    new_catalogue = pd.DataFrame()
    new_catalogue['objid'] = catalogue[colnames['source_identifier']]
    new_catalogue['original_image'] = [original_image] * len(new_catalogue)
    new_catalogue['peak_flux'] = catalogue[colnames['Peak_flux']]
    new_catalogue['x'] = x
    new_catalogue['y'] = y
    new_catalogue['ra'] = catalogue.RA
    new_catalogue['dec'] = catalogue.DEC

    new_catalogue.drop_duplicates(subset='objid', inplace=True)
    return new_catalogue


def create_catalogue_spreadsheet(image_dataset, scores,
                                 filename='anomaly_catalogue.xlsx',
                                 ignore_nearby_sources=True,
                                 source_radius=0.016):
    """
    Creates a catalogue of the most anomalous sources in the form of an excel
    spreadsheet that includes cutout images.

    Parameters
    ----------
    image_dataset : astronomaly.data_management.image_reader.ImageDataset
        The image dataset
    scores : pd.DataFrame
        The list of objects to convert to spreadsheet. NOTE: This must already
        be sorted in the order you want in the spreadsheet and limited to the
        number you want displayed.
    filename : str, optional
        Filename for spreadsheet, by default 'anomaly_catalogue.xlsx'
    ignore_nearby_sources : bool, optional
        If true, will search for nearby objects before adding to the
        spreadsheet and will only add if no source is found within
        source_radius degrees, by default True
    source_radius : float, optional
        Number of degrees to exclude nearby sources by in degrees, default 
        0.016 degrees
    """

    workbook = xlsxwriter.Workbook(filename, {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet()

    # Widen the first column to make the text clearer.
    worksheet.set_column('A:E', 25)
    worksheet.set_column('G:H', 25)
    worksheet.set_column('F:F', 30)

    cell_format = workbook.add_format({
        'bold': True, 'font_size': 14, 'center_across': True})
    worksheet.set_row(0, 50, cell_format)

    worksheet.write('A1', 'ObjID')
    worksheet.write('B1', 'Image Name')
    worksheet.write('C1', 'RA')
    worksheet.write('D1', 'DEC')
    worksheet.write('E1', 'Peak Flux')
    worksheet.write('F1', 'Cutout')
    worksheet.write('G1', 'Type')
    worksheet.write('H1', 'Comments')

    cell_format = workbook.add_format({'center_across': True})
    hgt = 180

    cat = image_dataset.metadata
    cat.index = cat.index.astype('str')

    row = 2
    for i in range(len(scores)):
        idx = scores.index[i]
        proceed = True

        if ignore_nearby_sources and i > 0:
            ra_prev = cat.loc[scores.index[:i], 'ra']
            dec_prev = cat.loc[scores.index[:i], 'dec']

            ra_diff = ra_prev - cat.loc[idx, 'ra']
            dec_diff = dec_prev - cat.loc[idx, 'dec']
            radius = np.sqrt(ra_diff ** 2 + dec_diff ** 2)
            if np.any(radius < source_radius):
                proceed = False

        if proceed:
            if cat.loc[idx, 'peak_flux'] == -1:
                # Will trigger it to set the flux
                image_dataset.get_sample(idx)
            worksheet.set_row(row - 1, hgt, cell_format)

            worksheet.write('A%d' % row, idx)

            worksheet.write('B%d' % row, cat.loc[idx, 'original_image'])
            worksheet.write('C%d' % row, cat.loc[idx, 'ra'])
            worksheet.write('D%d' % row, cat.loc[idx, 'dec'])
            worksheet.write('E%d' % row, cat.loc[idx, 'peak_flux'])
            fig = image_dataset.get_display_data(idx)

            image_options = {'image_data': fig, 'x_scale': 2, 'y_scale': 2}
            worksheet.insert_image('F%d' % row, 'img.png', image_options)

            row += 1
    workbook.close()


def get_visualisation_sample(features, anomalies, anomaly_column='score',
                             N_anomalies=20, N_total=2000):
    """
    Convenience function to downsample a set of data for a visualisation plot
    (such as t-SNE or UMAP). You can choose how many anomalies to highlight
    against a backdrop of randomly selected samples.
    Parameters
    ----------
    features : pd.DataFrame
        Input feature set
    anomalies : pd.DataFrame
        Contains the anomaly score to rank the objects by.
    anomaly_column : string, optional
        The column used to rank the anomalies by (always assumes higher is more
        anomalous), by default 'score'
    N_anomalies : int, optional
        Number of most anomalous objects to plot, by default 20
    N_total : int, optional
        Total number to plot (not recommended to be much more than 2000 for
        t-SNE), by default 2000
    """
    if N_total > len(features):
        N_total = len(features)
    if N_anomalies > len(features):
        N_anomalies = 0
    N_random = N_total - N_anomalies

    index = anomalies.sort_values(anomaly_column, ascending=False).index
    inds = index[:N_anomalies]
    other_inds = index[N_anomalies:]
    inds = list(inds) + list(np.random.choice(other_inds,
                             size=N_random, replace=False))
    return features.loc[inds]


def create_ellipse_check_catalogue(image_dataset, features,
                                   filename='ellipse_catalogue.csv'):
    """
    Creates a catalogue that contains sources which require a larger window
    or cutout size. Also contains the recommended windows size required.

    Parameters
    ----------
    image_dataset : astronomaly.data_management.image_reader.ImageDataset
        The image dataset
    features : pd.DataFrame
        Dataframe containing the extracted features about the sources. Used to
        obtain the ellipse warning column.
    filename : str, optional
        Filename for spreadsheet, by default 'ellipse_catalogue.csv'
    """

    dat = features.copy()

    met = image_dataset.metadata

    ellipse_warning = dat.loc[dat['Warning_Open_Ellipse'] == 1]

    data = pd.merge(ellipse_warning[[
                    'Warning_Open_Ellipse', 'Recommended_Window_Size']],
                    met, left_index=True, right_index=True)

    data.to_csv(filename)


class ImageCycler:
    def __init__(self, images, xlabels=None):
        """
        Convenience object to cycle through a list of images inside a jupyter 
        notebook.

        Parameters
        ----------
        images : list
            List of numpy arrays to display as images
        xlabels : list, optional
            List of custom labels for the images
        """

        self.current_ind = 0
        self.images = images
        self.xlabels = xlabels

    def onkeypress(self, event):
        """
        Matplotlib event handler for left and right arrows to cycle through 
        images.

        Parameters
        ----------
        event

        Returns
        -------

        """
        plt.gcf()
        if event.key == 'right' and self.current_ind < len(self.images):
            self.current_ind += 1

        elif event.key == 'left' and self.current_ind > 0:
            self.current_ind -= 1

        plt.clf()
        event.canvas.figure.gca().imshow(
            self.images[self.current_ind], origin='lower', cmap='hot')

        if self.xlabels is not None:
            plt.xlabel(self.xlabels[self.current_ind])
        plt.title(self.current_ind)
        event.canvas.draw()

    def cycle(self):
        """
        Creates the plots and binds the event handler
        """

        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        plt.imshow(self.images[self.current_ind], origin='lower', cmap='hot')
        plt.title(self.current_ind)


def get_file_paths(image_dir, catalogue_file, file_type='.fits'):
    """
    Finds and appends the pathways of the relevant files to the catalogue. 
    Required to access the files when passing a catalogue to the 
    ImageThumbnailsDataset.

    Parameters
    ----------
    image_dir : str
        Directory where images are located (can be a single fits file or 
        several)
    catalogue_file : pd.DataFrame
        Dataframe that contains the information pertaining to the data.
    file_type : str
        Sets the type of files used. Commonly used file types are .fits
        or .jpgs.

    Returns
    -------
    catalogue_file : pd.DataFrame
        Dataframe with the required file pathways attached.
    """

    filenames = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.endswith(file_type):
                filenames.append(os.path.join(root, f))

    filenames = sorted(filenames, key=lambda x: x.split('/')[-1])

    catalogue = catalogue_file.sort_values(['ra', 'dec'])

    catalogue['filename'] = filenames

    return catalogue


def convert_tractor_catalogue(catalogue_file, image_file, image_name=''):
    """
    Converts a tractor fits file to a pandas dataframe to be given
    directly to an ImageDataset object.

    Parameters
    ----------
    catalogue_files : string
        tractor catalogue in fits table format 
    image_file:
        The image corresponding to this catalogue (to extract pixel information
        and naming information)
    """

    catalogue = astropy.table.Table(astropy.io.fits.getdata(catalogue_file))

    dataframe = {}
    for name in catalogue.colnames:
        data = catalogue[name].tolist()
        dataframe[name] = data

    old_catalogue = pd.DataFrame(dataframe)
    hdul = astropy.io.fits.open(image_file)

    if len(image_name) == 0:
        original_image = image_file.split(os.path.sep)[-1]
    else:
        original_image = image_name

    new_catalogue = pd.DataFrame()
    new_catalogue['objid'] = old_catalogue['objid']
    new_catalogue['original_image'] = [original_image] * len(new_catalogue)
    new_catalogue['flux_g'] = old_catalogue['flux_g']
    new_catalogue['flux_r'] = old_catalogue['flux_r']
    new_catalogue['flux_z'] = old_catalogue['flux_z']
    new_catalogue['x'] = old_catalogue['bx'].astype('int')
    new_catalogue['y'] = old_catalogue['by'].astype('int')
    new_catalogue['ra'] = old_catalogue['ra']
    new_catalogue['dec'] = old_catalogue['dec']

    return new_catalogue


def create_png_output(image_dataset, number_of_images, data_dir):
    """
    Simple function that outputs a certain number of png files
    from the input fits files

    Parameters
    ----------
    image_dataset : astronomaly.data_management.image_reader.ImageDataset
        The image dataset
    number_of_images : integer
        Sets the number of images to be created by the function
    data_dir : directory
        Location of data directory. 
        Needed to create output folder for the images.

    Returns
    -------
    png : image object
        Images are created and saved in the output folder
    """

    out_dir = os.path.join(data_dir, 'Output', 'png')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(number_of_images):
        idx = image_dataset.index[i]
        name = image_dataset.metadata.original_image[i]

        sample = image_dataset.get_display_data(idx)

        pil_image = Image.open(sample)

        pil_image.save(os.path.join(
            out_dir, str(name.split('.fits')[0])+'.png'))


def remove_corrupt_file(met, ind, idx):
    """
    Function that removes the corrupt or missing file
    from the metadata and from the metadata index.

    Parameters
    ----------
    met : pd.DataFrame
        The metadata of the dataset
    ind : string
        The index of the metadata
    idx : string
        The index of the source file

    """

    ind = np.delete(ind, np.where(ind == idx))
    met = np.delete(met, np.where(met == idx))
