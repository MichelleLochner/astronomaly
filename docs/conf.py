# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'astronomaly'
copyright = '2020, Michelle Lochner'
author = 'Michelle Lochner'

# The full version, including alpha/beta/rc tags
release = '0.1'


def run_apidoc(_):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(cur_dir, 'apidocs')
    module = '../astronomaly'
    cmd_path = 'sphinx-apidoc'
    if hasattr(sys, 'real_prefix'):  # Check to see if we are in a virtualenv
        # If we are, assemble the path manually
        cmd_path = os.path.abspath(os.path.join(sys.prefix, 
                                                'bin', 'sphinx-apidoc'))
    subprocess.check_call([cmd_path, '-o', output_path, 
                           module, '--force'])


def setup(app):
    app.connect('builder-inited', run_apidoc)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax'
]

intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None)}

numpydoc_show_class_members = False
autosummary_generate = True
autoclass_content = "class"
autodoc_default_flags = ["members", "no-special-members"]
autodoc_mock_imports = ["opencv"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'apidocs/astronomaly.rst', 'apidocs/modules.rst']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The suffixes of source files.
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
