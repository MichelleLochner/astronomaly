from random import random

import numpy as np
import pandas as pd

from bokeh.layouts import row, column, layout
from bokeh.models import Button, ColumnDataSource
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

from PIL import Image

# **two-dimensional array** of RGBA values **encoded as 32-bit integers**
# https://github.com/bokeh/bokeh/issues/1699
def encode_to_32bit(im_data):
    if im_data.ndim > 2: # could also be im_data.dtype == np.uint8
        if im_data.shape[2] == 3: # alpha channel not included
            im_data = np.dstack([im_data, np.ones(im_data.shape[:2], np.uint8) * 255])
        return np.squeeze(im_data.view(np.uint32))


def make_galaxy_figure(source):
    # can modify source by reference elsewhere still with callbacks, will update
    # create a plot and style its properties
    p = figure(x_range=(0, 424), y_range=(0, 424), toolbar_location=None, plot_width=250, plot_height=250)
    # p.border_fill_color = 'black'
    # p.background_fill_color = 'black'
    p.outline_line_color = None
    p.grid.grid_line_color = None
    _ = p.image_rgba('image', 'x', 'y', 'dw', 'dh', source=source)
    return p


def get_image_source(file_loc):
    im_data = np.array(Image.open(file_loc))
    im = encode_to_32bit(im_data)

    xsize = 424
    ysize = 424
    source = ColumnDataSource(
        data = {'image':[im], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
    )
    return source



df = pd.read_parquet('cnn_features_30.parquet', columns=['filename'])
filenames = list(df['filename'])

sources = [get_image_source(file_loc) for file_loc in filenames[:3]]
gal_figures = [make_galaxy_figure(source) for source in sources]

    
i = 0

# ds = r_image.data_source  # will be source
# update .data to modify

# create a callback that adds a number in a random location
def callback():
    global i

    file_locs = filenames[i*3:i*3+3]

    for source_n, source in enumerate(sources):
        file_loc = file_locs[source_n]
        source.data = dict(get_image_source(file_loc).data)
        
    i = i + 1

# add a button widget and configure with the call back
button = Button(label="Press Me")
button.on_click(callback)

# put the button and plot in a layout and add to the document


my_layout = layout([
    [button],
    [row(gal_figures)],
])
curdoc().add_root(my_layout)