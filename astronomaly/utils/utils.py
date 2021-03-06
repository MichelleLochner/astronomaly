import matplotlib.pyplot as plt
import astropy
import os
import pandas as pd
import numpy as np
import xlsxwriter


def convert_pybdsf_catalogue(catalogue_file, image_file):
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
    """
    if 'csv' in catalogue_file:
        catalogue = pd.read_csv(catalogue_file, skiprows=5)
        cols = list(catalogue.columns)
        for i in range(len(cols)):
            cols[i] = cols[i].strip()
        catalogue.columns = cols
    else:
        dat = astropy.table.Table(astropy.io.fits.getdata(catalogue_file))
        catalogue = dat.to_pandas()

    hdul = astropy.io.fits.open(image_file)
    original_image = image_file.split(os.path.sep)[-1]

    w = astropy.wcs.WCS(hdul[0].header, naxis=2)

    x, y = w.wcs_world2pix(catalogue.RA, catalogue.DEC, 1)

    new_catalogue = pd.DataFrame()
    new_catalogue['objid'] = catalogue['Source_id']
    new_catalogue['original_image'] = [original_image] * len(new_catalogue)
    new_catalogue['peak_flux'] = catalogue['Peak_flux']
    new_catalogue['x'] = x
    new_catalogue['y'] = y
    new_catalogue['ra'] = catalogue.RA
    new_catalogue['dec'] = catalogue.DEC

    new_catalogue.drop_duplicates(subset='objid', inplace=True)
    new_catalogue.to_csv('deep2_offset_catalogue.tsv', sep='\t')
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

    workbook = xlsxwriter.Workbook(filename)
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
