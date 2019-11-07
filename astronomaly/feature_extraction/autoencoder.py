import numpy as np
import os
from astronomaly.base.base_pipeline import PipelineStage

try:
    from keras.models import load_model
    from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
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
            Allows for loading of previously trained Keras model in HDF5 
            format. Note these models are very sensitive, the exact same 
            preprocessing steps must be used to reproduce results.
        """
        if len(model_file) != 0:
            try:
                self.autoencoder = load_model(model_file)
                inputs = self.autoencoder.input
                outputs = self.autoencoder.get_layer('encoder').output
                self.encoder = Model(inputs=inputs, outputs=outputs)
            except OSError:
                print('Model file ', model_file, 
                      'is invalid. Weights not loaded.')
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
        Compiles the default autoencoder model. Note this model is designed to 
        operate on 128x128 images. While it can run on different size images 
        this can dramatically change the size of the final feature space.

        Parameters
        ----------
        input_image_shape : tuple
            The expected shape of the input images. Can either be length 2 or 3
            (to include number of channels).

        Returns
        -------

        """

        if len(input_image_shape) == 2:
            input_image_shape = (input_image_shape[0], input_image_shape[1], 1)

        # Assumes "channels last" format
        input_img = Input(shape=input_image_shape)  

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
        decoder = Conv2D(input_image_shape[-1], (3, 3), activation='sigmoid', 
                         padding='same')(x)

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
            Either array or list of images. It's recommended that this data be 
            augmented with translation or rotation (or both).
        batch_size : int, optional
            Number of samples used to update weights in each iteration. A 
            larger batch size can be more accurate but requires more memory and
            is slower to train.
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
            Input images (nobjects x image_shape). For a single image, 
            provide [image] as an array is expected.

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


class AutoencoderFeatures(PipelineStage):
    def __init__(self, training_dataset=None, retrain=False, **kwargs):
        """
        Runs a standard autoencoder on image cutouts. This function needs to be 
        more flexible in terms of model parameters!

        Parameters
        ----------   
        """
        super().__init__(training_dataset=training_dataset, **kwargs)

        if training_dataset is None:
            raise ValueError('A training dataset object must be provided.')

        model_file = os.path.join(self.output_dir, 'autoencoder.h5')

        if retrain or ('force_rerun' in kwargs and kwargs['force_rerun']):
            self.autoenc = Autoencoder()
        else:
            self.autoenc = Autoencoder(model_file=model_file)

        if self.autoenc.autoencoder is None:

            print('Compiling autoencoder model...')
            self.autoenc.compile_autoencoder_model((
                training_dataset.window_size_x, 
                training_dataset.window_size_y))
            print('Done!')
            print('Training autoencoder...')
            self.autoenc.fit(training_dataset.cutouts, epochs=10)
            print('Done!')

            if self.save_output:
                print('Autoencoder model saved to', model_file)
                self.autoenc.save(model_file)

        else:
            print('Trained autoencoder read from file', model_file)

    def _execute_function(self, image):
        feats = self.autoenc.encode(image)
        feats = np.reshape(feats, [np.prod(feats.shape[1:])])
        if len(self.labels) == 0:
            self.labels = ['enc_%d' % i for i in range(len(feats))] 
        return feats
