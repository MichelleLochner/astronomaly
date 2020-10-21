import setuptools
import re

VERSIONFILE = "_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setuptools.setup(
    name="astronomaly",
    version=verstr,
    author="Michelle Lochner",
    author_email="dr.michelle.lochner@gmail.com",
    description="A general anomaly detection framework for Astronomical data",
    long_description_content_type="text/markdown",
    url="https://github.com/MichelleLochner/astronomaly",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3 License",
        "Operating System :: OS Independent",
    ],
)
