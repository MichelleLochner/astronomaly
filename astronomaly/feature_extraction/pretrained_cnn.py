import numpy as np
from astronomaly.base.base_pipeline import PipelineStage

try:
    import torch
    from torchvision import models
    from torchvision import transforms
except ImportError:
    err_string = "pytorch and torchvision must be installed to use this module"
    raise ImportError(err_string)

try:
    from zoobot.pytorch.training import finetune
    zoobot_available = True
except ImportError:
    zoobot_available = False


class CNN_Features(PipelineStage):
    def __init__(self, 
                 model_choice='resnet18', 
                 zoobot_checkpoint_location='',
                 **kwargs):
        """
        Runs a pretrained CNN and extracts the deep features before the 
        classification layer.

        Parameters
        ----------
        model_choice: string
            The model to use. Options are:
            'zoobot', 'resnet18' or 'resnet50'. These also use predefined
            transforms
        """

        super().__init__(model_choice=model_choice, **kwargs)

        self.model_choice = model_choice
        # Easiest to set these once this has been run once
        self.labels = []

        # All the models use these
        default_transforms = [transforms.ToTensor(),
                              transforms.Resize(256, antialias=True),
                              transforms.CenterCrop(224)]
        # Normalizations used by resnet
        resnet_normalization = [transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]

        if model_choice == 'zoobot':
            if not zoobot_available:
                err_string = "Please install zoobot to use this model: "
                err_string += "https://github.com/mwalmsley/zoobot"
                raise ImportError(err_string)

            if len(zoobot_checkpoint_location) == 0:
                err_string = ("Please download the weights for the zoobot" 
                              " model and provide the location of the" 
                              " checkpoint file in"
                              " zoobot_checkpoint_location")
                raise FileNotFoundError(err_string)

            self.transforms = transforms.Compose(default_transforms)

            self.model = finetune.load_encoder(zoobot_checkpoint_location)

        else:
            # It's one of the resnets
            transform_list = default_transforms + resnet_normalization

            self.transforms = transforms.Compose(transform_list)

            if model_choice == 'resnet18':
                wgts = models.ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=wgts)
            else:
                wgts = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=wgts)

            # Strip off the last layer to get a normal feature extractor
            self.model = torch.nn.Sequential(*list(model.children())[:-1])

    def _execute_function(self, image):
        """
        Runs the appropriate CNN model

        Parameters
        ----------
        image : np.ndarray
            Input image

        Returns
        -------
        array
            Contains the extracted deep features
        """
        # The transforms can't handle floats to convert to uint8
        image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:  # Greyscale
            # Make a copy of this channel to all others
            image = np.stack((image,) * 3, axis=-1)

        print('image shape', image.shape)
        processed_image = self.transforms(image)
        # Add the extra alpha channel the nets expect
        processed_image = torch.unsqueeze(processed_image, 0)

        # Run the model, detach from the GPU, turn it into a numpy array
        # and remove superfluous dimensions (which will likely be a different)
        # number for different models
        feats = self.model(processed_image).detach().numpy().squeeze()

        if len(self.labels) == 0:
            self.labels = [f'feat{i}' for i in range(len(feats))]

        return feats
