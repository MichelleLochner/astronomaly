===========
Quick Start
===========

To run the frontend, run ``python astronomaly/frontend/run_server.py
<script_file>``. 
Astronomaly runs on the principle of "code as
config". The script you provide must tell the server what it should run.
Example scripts can be found in the ``scripts`` folder. They will generally all
follow the same format: reading in data into an Astronomaly ``Dataset`` object,
running feature extraction, dimensionality reduction, postprocessing and
anomaly detection as necessary. You may also make use of a visualisation method
like t-SNE and include human-in-the-loop learning. Your pipeline script must
return a dictionary with the following keywords:

* ``dataset`` - the Dataset object that supplies the required functions for the frontend to plot data
* ``features`` - DataFrame containing the final set of features used for anomaly detection
* ``anomaly_scores`` - DataFrame containing the machine learning anomaly
  scores (in a column called "score")
* ``visualisation`` - t-SNE or similar DataFrame
* | ``active_learning`` - A PipelineStage object (see below under "Contributing")
  | which can be called to interactively run active learning.

After the pipeline has been run by ``run_server.py``, the console will display a
url, usually http://127.0.0.1:5000/. Navigate there using any web browser to see the frontend, no
JavaScript development, running or compiling required.

Ensure you close Astronomaly properly with the cross on the top right corner so
the server is shut down correctly, otherwise you may find if you run
Astronomaly again it will detect the port as being unavailable.

Example scripts
---------------
There are several examples in the scripts folder. These make use of the
`example_data` folder in the top level directory and are all run with the same
command:

``python astronomaly/frontend/run_server.py astronomaly/scripts/<script_name>.py``.

The scripts are:

* ``galaxy_zoo_example.py`` - This will run a subset of the Galaxy Zoo data as described in the paper.

* ``raw_features_example.py`` - This will run the full simulations example from the paper, including automatic labelling and active learning

* ``goods_example.py`` - This is an example of how to run Astronomaly on a fits file with a catalogue. 

For the last example script, an example fits file from the GOODS-S survey will be
automatically downloaded (using `wget`) if no fits files are found. This is a
single band fits files, you can display multiband fits files but it's slightly
complicated to set up. Each fits file must be renamed with a band prefix (e.g.
``v-``, ``r-`` etc.) and the keywords `band_prefixes` and `bands_rgb` must be
provided to `ImageDataset`. More documentation on fits files will come soon.
Please contact us for help in working with multiband images.