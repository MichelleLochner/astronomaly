<image src="Astronomaly_logo.png" width="200" align="left"/> 

# Astronomaly
A flexible framework for anomaly detection in astronomy.

<br>
<br>
<br>
<br>
Astronomaly is made up of two components: a python backend housed under "astronomaly", which can be used as a standalone library
or in conjunction with the JavaScript frontend, housed under "webapp".

## Warning

Astronomaly is being actively developed and may contain bugs or instabilities! I strongly recommend getting in touch with me if you're planning to use the code so I can help support you. Also if you're looking for light curve analysis (such as used in https://arxiv.org/abs/2008.04666), this is still being worked into Astronomaly and will be supported in future.

## Installation

### Setting up the environment

It's recommended to use anaconda to create an environment for Astronomaly. In the main repository directory, type `conda env create -f astronomaly_env.yml` which will automatically install all required packages. 

To install the python backend, use the standard `python setup.py install` or `python setup.py develop` if you plan to make
changes to the files. 



## Running

To run the frontend, navigate to `astronomaly/frontend` and type `python run_server.py <script_file>`. Astronomaly runs on the principle of "code as config". The script you provide must tell the server what it should run. Example scripts can be found in the `script` folder. They will generally all follow the same format: reading in data into an Astronomaly `Dataset` object, running feature extraction, dimensionality reduction, postprocessing and anomaly detection as necessary. You may also make use of a visualisation method like t-SNE and include human-in-the-loop learning. Your pipeline script must return a dictionary with the following keywords:<p>
`dataset` - the Dataset object that supplies the required functions for the frontend to plot data <p>
`features` - DataFrame containing the final set of features used for anomaly detection <p>
`anomaly_scores` - DataFrame containing the machine learning anomaly scores (in a column called "score") <p>
`visualisation` - t-SNE or similar DataFrame <p>
`active_learning` - A PipelineStage object (see below under "Contributing") which can be called to interactively run active learning.

After the pipeline has been run by `run_server.py`, the console will display a url, usually http://127.0.0.1:5000/. Navigate there to see the frontend, no JavaScript development, running or compiling required.

## Contributing to Astronomaly

Astronomaly runs in a fully object-oriented framework. The main advantage of this is a great deal of functionality can be inherited from the base classes. The base pipeline class takes care of the following:
- Automatic logging of all function calls
- Automatic checks of whether the same function with the same arguments has already been called for given data (can be overridden with `force_rerun=True`)
- Automatic saving and loading of output files (overridden with `save_output=False`) 
- (Potential) automatic parallelisation

Contributions are incredibly welcome! If you already have write access to the repository, please create a new well-named branch with your contributions and create a pull request when ready. Otherwise, please fork the repository and create a pull request when ready.

### Contributing a new Dataset class

If your data is an image, time series or raw feature type, it is preferable to add flexibility to these existing classes rather than write a new class. If you need to read in an entirely new type of data however, a new Dataset class can be added. The new module should be added to the `dataset_management` folder and the new class should inherit from the base Dataset class like this:

```
from astronomaly.base.base_dataset import Dataset

class NewDatasetClass(Dataset):
  def __init__(an_argument=2, another_argument='hello', **kwargs):
    super().__init__(an_argument=an_argument, another_argument=another_argument, **kwargs)
```
    
Arguments must be explictly passed to the parent class' init function to ensure correct logging. The functions `get_sample` and `get_display_data` must be implemented. The class variable `metadata` must be defined and should be a DataFrame with the index corresponding to a unique key for each sample. The class variable `data_type` must also be defined and should correspond to a known string in the webapp in order for the data to be displayed correctly.

### Contributing a new PipelineStage class

Every step in the Astronomaly framework after the data has been read in can be represented with a PipelineStage object. If you would like to contribute a new feature extraction method or anomaly detection algorithm, these must be coded up as a class. Similar to the Dataset class, the PipelineStage class follows the following structure:

```
from astronomaly.base.base_pipeline import PipelineStage

class NewPipelineStage(PipelineStage):
  def __init__(an_argument=2, another_argument='hello', **kwargs):
    super().__init__(an_argument=an_argument, another_argument=another_argument, **kwargs)
    
  def _excute_function(self, data):
    <your actual function here>
    return output
```

The only function that has to be implemented in the new class is the `_execute_function`. This function should take a single input and produce a single output. All arguments required should be declared in `__init__` as class variables so that they can be logged properly. The input type depends on whether this PipelineStage is meant to operate on the raw data (i.e. doing feature extraction) or any stage after that. `_execute_function` should never be called directly, instead a user should call one of two functions in the base `PipelineStage` class: `run` and `run_on_dataset`.

`run_on_dataset`, as the name suggests, should be called with a `Dataset` argument as input. It will automatically repeatedly call `_execute_function` on each sample in the `Dataset`, producing a DataFrame as output.

`run` on the other hand takes a DataFrame as input and then runs `_execute_function` on the entire DataFrame. This would be used for dimensionality reduction, anomaly detection etc.

Finally the new PipelineStage class should be put in a well-named module and placed in the appropriate folder within astronomaly.

### Contributing to the webapp

If you have expertise in JavaScript, particularly with React, we'd welcome contributions to the front end. The files can be compiled using `npm run watch` in the `webapp` folder where changes can be viewed immediately. 

