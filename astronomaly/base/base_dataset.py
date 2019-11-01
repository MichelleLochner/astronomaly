import os
from astronomaly.base import logging_tools


class Dataset(object):
    def __init__(self, *args, **kwargs):
        self.data_type = None

        if 'filename' in kwargs:
            filename = kwargs['filename']
        else:
            filename = ''
        if 'directory' in kwargs:
            directory = kwargs['directory']
        else:
            directory = ''
        if 'list_of_files' in kwargs:
            list_of_files = kwargs['list_of_files']
        else:
            list_of_files = [],
        if len(filename) != 0:
            self.files = [filename]
        elif len(list_of_files) != 0 and len(directory) == 0:
            # Assume the list of files are absolute paths
            self.files = list_of_files
        elif len(list_of_files) != 0 and len(directory) != 0:
            # Assume the list of files are relative paths to directory
            fls = list_of_files
            self.files = [os.path.join(directory, f) for f in fls]
        elif len(directory) != 0:
            # Assume directory contains all the files we need
            fls = os.listdir(directory)
            fls.sort()
            self.files = [os.path.join(directory, f) for f in fls]
        else:
            self.files = []

        # Handles automatic file reading and writing
        if 'output_dir' in kwargs:
            self.output_dir = kwargs['output_dir']
        else:
            self.output_dir = './'


        # This allows the automatic logging every time this class is instantiated (i.e. every time this pipeline stage
        # is run). That means any class that inherits from this base class will have automated logging.

        logging_tools.setup_logger(os.path.join(self.output_dir, 'astronomaly.log'))

        class_name = type(locals()['self']).__name__
        function_call_signature = logging_tools.format_function_call(class_name, *args, **kwargs)
        logging_tools.log(function_call_signature)

    def get_sample(self, idx):
        raise NotImplementedError

    def get_display_data(self, idx):
        raise NotImplementedError