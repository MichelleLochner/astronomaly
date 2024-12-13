<image src="Astronomaly_logo.png" width="200" align="left"/> 

# Astronomaly
A flexible framework for anomaly detection in astronomy.<br>
[![DOI](https://zenodo.org/badge/196393655.svg)](https://doi.org/10.5281/zenodo.14441057)

<br>
<br>
<br>
<br>
Astronomaly is made up of two components: a python backend housed under "astronomaly", which can be used as a standalone library
or in conjunction with the JavaScript frontend, housed under "webapp".

## Warning

Astronomaly is being actively developed and may contain bugs or instabilities! I strongly recommend getting in touch with me if you're planning to use the code so I can help support you. Also if you're looking for light curve analysis (such as used in https://arxiv.org/abs/2008.04666), this is still being worked into Astronomaly and will be supported in future.

## Super quickstart

Clone or download the Astronomaly repository:<br>
`git clone https://github.com/MichelleLochner/astronomaly/`

Navigate to the Astronomaly folder. 

### If you use virtualenv and pip do this::

Make sure you've installed <a href="https://virtualenv.pypa.io/en/latest/installation.html">virtualenv</a> to create virtual environments with native python.

Create the virtual environment: <br>
`virtualenv venv_astronomaly` 

Activate the environment: <br>
`source venv_astronomaly/bin/activate`

Install required packages: <br>
`pip install -r requirements.txt`

### If you use anaconda, do this instead::

Create a new environment:<br>
`conda env create -f astronomaly_env.yml` 

Don't forget to activate the environment:<br>
`activate astronomaly`

### Installing and running Astronomaly:

Install the code:<br>
`pip install .`

Run the Galaxy Zoo example:<br>
`python astronomaly/frontend/run_server.py astronomaly/scripts/galaxy_zoo_example.py`

After running, you should see output tell you to open your browser at a particular address, usually http://127.0.0.1:5000/.

Explore the web interface!

## Documentation

The python documentation is hosted here:
https://astronomaly.readthedocs.io/en/latest/

## Citation

Please cite Lochner and Bassett (2020) if you use Astronomaly (https://arxiv.org/abs/2010.11202).<br>
Please also cite Lochner and Rudnick (2024) is you use Protege (https://arxiv.org/abs/2411.04188). <br>
Thank you to Sara Webb for designing the logo!




