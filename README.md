# astronomaly
A flexible framework for anomaly detection in astronomy.

Astronomaly is made up of two components: a python backend housed under "astronomaly", which can be used as a standalone library
or in conjunction with the JavaScript frontend, housed under "webapp".

To install the python backend, use the standard `python setup.py install` or `python setup.py develop` if you plan to make
changes to the files. An example pipeline can be found under `scripts`. 

To run the frontend, navigate to `astronomaly/frontend` and type `python run.py`. From the console output you will see that 
a pipeline script is automatically run. All you need to do is change the input image in the `interface.py` file. Some good
examples can be found there. The console will display a url, usually http://127.0.0.1:5000/. Navigate there to see the frontend,
no JavaScript development, running or compiling required.
