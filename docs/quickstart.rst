===========
Quick Start
===========

To run the frontend, navigate to ``astronomaly/frontend`` and type ``python
run_server.py <script_file>``. Astronomaly runs on the principle of "code as
config". The script you provide must tell the server what it should run.
Example scripts can be found in the ``script`` folder. They will generally all
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

After the pipeline has been run by `run_server.py`, the console will display a
url, usually http://127.0.0.1:5000/. Navigate there using any web browser to see the frontend, no
JavaScript development, running or compiling required.

Ensure you close Astronomaly properly with the cross on the top right corner so
the server is shut down correctly, otherwise you may find if you run
Astronomaly again it will detect the port as being unavailable.