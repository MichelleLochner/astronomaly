import os
import logging

from astronomaly.base import logging_tools


class Dataset(object):
    def __init__(self, *args, **kwargs):
        """
        Base Dataset object that all other dataset objects should inherit from.
        Whenever a child of this class is implemented, super().__init__()
        should be called and explicitly passed all kwargs of the child class,
        to ensure correct logging and saving of files.

        Parameters
        ----------
        filename : str
            If a single file (of any time) is to be read from, the path can be
            given using this kwarg. 
        directory : str
            A directory can be given instead of an explicit list of files. The
            child class will load all appropriate files in this directory.
        list_of_files : list
            Instead of the above, a list of files to be loaded can be
            explicitly given.
        output_dir : str
            The directory to save the log file and all outputs to. Defaults to
            './' 
        """
        self.data_type = None
        logging.debug('Begin base dataset init')

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
            list_of_files = []
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
            logging.debug('Setting output dir from kwargs to ', self.output_dir)
        else:
            self.output_dir = './'

        # This allows the automatic logging every time this class is 
        # instantiated (i.e. every time this pipeline stage
        # is run). That means any class that inherits from this base class 
        # will have automated logging.

        logging_tools.setup_logger(log_directory=self.output_dir, 
                                   log_filename='astronomaly.log')

        class_name = type(locals()['self']).__name__
        function_call_signature = logging_tools.format_function_call(
            class_name, *args, **kwargs)
        logging_tools.log(function_call_signature)

    def clean_up(self):
        """
        Allows for any clean up tasks that might be required.
        """
        pass

    def get_sample(self, idx):
        """
        Returns a single instance of the dataset given an index.

        Parameters
        ----------
        idx : str
            Index (should be a string to avoid ambiguity)

        Raises
        ------
        NotImplementedError
            This function must be implemented when the base class is inherited.
        """
        raise NotImplementedError

    def get_display_data(self, idx):
        """
        Returns a single instance of the dataset in a form that is ready to be
        displayed by the web front end.

        Parameters
        ----------
        idx : str
            Index (should be a string to avoid ambiguity)

        Raises
        ------
        NotImplementedError
            This function must be implemented when the base class is inherited.
        """
        raise NotImplementedError
