import numpy as np
import xarray
from astronomaly.preprocessing import image_preprocessing
import os

try:
    from keras.models import load_model
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
    from keras.models import Model
except ImportError:
    print("Failed to import Keras. Deep learning will be unavailable.")


class Autoencoder:
    def __init__(self, model_file=''):
        """
        Class containing autoencoder training methods.
        Parameters
        ----------
        model_file : string, optional
            Allows for loading of previously trained Keras model in HDF5 format. Note these models are very sensitive,
            the exact same preprocessing steps must be used to reproduce results.
        """
        if len(model_file) != 0:
            try:
                self.autoencoder = load_model(model_file)
                self.encoder = Model(inputs=self.autoencoder.input,
                                     outputs=self.autoencoder.get_layer('encoder').output)
            except OSError:
                print('Model file ', model_file, 'is invalid. Weights not loaded.')
                self.autoencoder = None
        else:
            self.autoencoder = None

    def shape_check(self, images):
        """
        Convenience function to reshape images appropriate for deep learning.

        Parameters
        ----------
        images : np.ndarray, list
            Array of list of images

        Returns
        -------
        np.ndarray
            Converted array compliant with CNN

        """
        images = np.array(images)

        if len(images.shape) == 2:
            images = images.reshape([-1, images.shape[0], images.shape[1], 1])
        if len(images.shape) == 3:
            images = images.reshape([-1, images.shape[1], images.shape[2], 1])

        return images

    def compile_autoencoder_model(self, input_image_shape):
        """
        Compiles the default autoencoder model. Note this model is designed to operate on 128x128 images. While it can
        run on different size images this can dramatically change the size of the final feature space.

        Parameters
        ----------
        input_image_shape : tuple
            The expected shape of the input images. Can either be length 2 or 3 (to include number of channels).

        Returns
        -------

        """

        if len(input_image_shape) == 2:
            input_image_shape = (input_image_shape[0], input_image_shape[1], 1)

        input_img = Input(shape=input_image_shape)  # Assumes "channels last" format

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((4, 4), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoder = MaxPooling2D((4, 4), padding='same', name='encoder')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoder = Conv2D(input_image_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoder)
        autoencoder.compile(loss='mse', optimizer='adam')
        self.autoencoder = autoencoder

        self.encoder = Model(inputs=autoencoder.input,
                        outputs=autoencoder.get_layer('encoder').output)

    def fit(self, training_data, batch_size=32, epochs=10):
        """
        Actually train the autoencoder.

        Parameters
        ----------
        training_data : np.ndarray, list
            Either array or list of images. It's recommended that this data be augmented with translation or rotation
            (or both).
        batch_size : int, optional
            Number of samples used to update weights in each iteration. A larger batch size can be more accurate but
            requires more memory and is slower to train.
        epochs : int, optional
            Number of full passes through the entire training set.

        Returns
        -------

        """

        X = self.shape_check(training_data)

        self.autoencoder.fit(X, X,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1,
                             shuffle=True)

    
    def encode(self, images):
        """
        Returns the deep encoded features for an array of images.

        Parameters
        ----------
        images : np.ndarray
            Input images (nobjects x image_shape). For a single image, provide [image] as an array is expected.

        Returns
        -------
        np.ndarray
            Deep features (nobjects x nfeatures)
        """

        return self.encoder.predict(self.shape_check(images))

    def save(self, filename):
        """
        Saves Keras model in HDF5 format

        Parameters
        ----------
        filename : string
            Location for saved model

        Returns
        -------

        """

        self.autoencoder.save(filename)


def extract_features_autoencoder(pipeline_dict, model_file='', input_key='cutouts',output_key='features_autoencoder'):
    """
    Runs a standard autoencoder on image cutouts. This function needs to be more flexible in terms of model parameters!

    Parameters
    ----------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    model_file : str, optional
        Location of saved keras model file to allow quick rerunning of autoencoder. If a file path is provided but does
        not exist, the model will be trained and then saved to this file path.
    input_key : str, optional
        The input key of pipeline_dict to run the function on.
    output_key : str, optional
        The output key of pipeline_dict

    Returns
    -------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    """
    window_size_x, window_size_y = pipeline_dict[input_key][0].shape
    # We want a smaller sliding window when generating training data to
    # produce enough data and also ensure translation invariance
    window_shift = window_size_x//2

    # We need to work out a way to do this that ensures the training data is created in the same way as the test data
    # Also may need to worry about a training/test split if the autoencoder is going to be used on other images
    print('Running autoencoder...')


    autoenc = Autoencoder(model_file=model_file)
    if autoenc.autoencoder is None:
        print('Generating training data...')
        pipeline_dict = image_preprocessing.generate_cutouts(pipeline_dict,
                                                             window_size=window_size_x, window_shift=window_shift,
                                                             output_key='training_data',
                                                             transform_function=image_preprocessing.image_transform_log)
        print('Compiling autoencoder model...')
        autoenc.compile_autoencoder_model((window_size_x, window_size_y))
        print('Done!')
        print('Training autoencoder...')
        autoenc.fit(pipeline_dict['training_data'])
        print('Done!')
    else:
        print('Trained autoencoder read from file.')

    print('Encoding images...')
    feats = autoenc.encode(pipeline_dict[input_key])
    feats = np.reshape(feats, [len(feats), np.prod(feats.shape[1:])])
    feats = xarray.DataArray(
        feats, coords={'id': pipeline_dict['metadata'].id}, dims=['id', 'features'], name=output_key)
    print('Done!')
    pipeline_dict[output_key] = feats

    if len(model_file) != 0 and not os.path.exists(model_file):
        autoenc.save(model_file)
    return pipeline_dict
