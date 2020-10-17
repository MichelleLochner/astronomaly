<image src="Astronomaly_logo.png" width="200" align="left"/> 

# Astronomaly
A flexible framework for anomaly detection in astronomy.

<br>
<br>
<br>
<br>
Astronomaly is made up of two components: a python backend housed under "astronomaly", which can be used as a standalone library
or in conjunction with the JavaScript frontend, housed under "webapp".
<br>
<b>Documentation</b>: https://astronomaly.readthedocs.io/en/latest/

## Warning

Astronomaly is being actively developed and may contain bugs or instabilities! I strongly recommend getting in touch with me if you're planning to use the code so I can help support you. Also if you're looking for light curve analysis (such as used in https://arxiv.org/abs/2008.04666), this is still being worked into Astronomaly and will be supported in future.

## Super quickstart

Clone or download the Astronomaly repository:<br>
`git clone https://github.com/MichelleLochner/astronomaly/`

Navigate to the Astronomaly folder and install the code:<br>
`python setup.py install`

Run the Galaxy Zoo example:<br>
`python astronomaly/frontend/run_server.py astronomaly/scripts/galaxy_zoo_example.py`

After running, you should see output tell you to open your browser at a particular address, usually http://127.0.0.1:5000/.

Explore the web interface!


## Documentation

The python documentation is hosted here and will be updated in the near future:
https://astronomaly.readthedocs.io/en/latest/

## Citation

Please cite Lochner and Bassett (in prep) if you use Astronomaly.




