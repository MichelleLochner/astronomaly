import os
import time
import numpy as np
import pandas as pd
from astronomaly.base.base_pipeline import PipelineStage

try:
    import torchvision
    from torchvision.utils import make_grid
    from torchvision.utils import save_image
    from torchvision import transforms, models
    import torch
    from torch.utils.data import Dataset, DataLoader
    from byol_pytorch import BYOL
    import kornia
except ImportError as e:
    err_string = """pytorch, torchvision, kornia and byol_pytorch must be
    installed to use this module."""
    raise ImportError(f"{err_string} \n {e}")  

class AstronomalyDataset(Dataset):
    def __init__(self, image_dataset, transform=None):
        """
        Creates a pytorch compatible dataset object from an 
        Astronomaly ImageDataset object
        which implements the get_sample method.
        """
        self.image_dataset = image_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_dataset.index)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        idx = self.image_dataset.index[i]
        image = self.image_dataset.get_sample(idx)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        if len(image.shape) > 2 and image.shape[-1] <= 4:
            # cvread reads in images with channel last but torch needs
            # channel first
            image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, idx

class BYOL_Features(PipelineStage):
    def __init__(
                self, 
                image_size=128,
                model=None, 
                strip_layer=False,
                projection_layer_size=100,
                normalize=False,
                grayscale=True,
                base_learning_rate=5e-4,
                n_epochs=100,
                batch_size=32,
                num_workers=16,
                save_model=True,
                load_model=False,
                restart=False,
                model_output_file_root="byol",
                train_model=True,
                augmentation_params=None,
                **kwargs):
        """
        Runs BYOL and extracts the deep features.

        Parameters
        ----------
        image_size: int
            The size that images will be resized to before being passed to the
            model. Ideally should be the same as original images but can be 
            smaller for speed.
        model: torchvision.model
            The model to use. If none, defaults to Efficentnet_B0 initialised
            with ImageNet weights. Note that the model must be stripped of its 
            last layer and replaced with an appropriately sized linear layer.
            See code for details.
        projection_layer_size: int
            The size of the projection layer in the BYOL model.
        normalize: bool
            Whether or not to normalize the features using standard resnet 
            normalization.
        base_learning_rate: float
            The learning rate for the BYOL model, will be scaled by the batch 
            size/256.
        batch_size: int
            The batch size for the BYOL model.
        num_workers: int
            The number of workers to use for the DataLoader.
        save_model: bool
            Whether or not to save the model to a file. Given the computational
            cost, it is strongly recommended to keep this set to True.
        load_model: bool
            Whether or not to load a model from a file. If True, the model will
            be loaded from the file specified in model_output_file_root.
        restart: bool
            If True, will restart training from a saved checkpoint.
        model_output_file_root: str
            The root name for the model output files. The model itself will be
            saved using this root name with the extension '_model.pt' and
            the loss will be saved as a csv file with the extension 
            '_loss.csv'. All files will be found in output_dir.
        train_model: bool
            Whether or not to train the model. If False, the saved model    
            weights will be used instead.
        augmentation_params: dict
            A dictionary of augmentation parameters to use. If None, default 
            parameters will be used.

            The current default parameters are:
            {
                "aug_rotation_p": 1,
                "aug_rotation_angle": 360,
                "aug_flip_p": 0.5,
                "aug_centre_crop_size": 110,
                "aug_centre_crop_p": 1,
                "aug_random_crop_min": 0.8,
                "aug_random_crop_max": 1,
                "aug_random_crop_p": 1,
                "aug_blurring_p": 0.1,
                "aug_blurring_kernel": 15,
                "aug_jiggle_p": 0.8,
                "aug_jiggle_amount": 0.5
            }
        """

        super().__init__(
            model=model, 
            projection_layer_size=projection_layer_size,
            normalize=normalize,
            base_learning_rate=base_learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            model_output_file_root=model_output_file_root,
            augmentation_params=augmentation_params,
            **kwargs)

        self.image_size = image_size
        self.learning_rate = base_learning_rate * batch_size / 256
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_model = save_model
        self.model_file = os.path.join(
                self.output_dir, model_output_file_root + '_model.pt')
        self.loss_file = os.path.join(
                self.output_dir, model_output_file_root + '_loss.csv')
        self.train_model = train_model
        self.labels = []

        # Default augmentation parameters
        default_augmentation_params = {
            "aug_rotation_p": 1,
            "aug_rotation_angle": 360,
            "aug_flip_p": 0.5,
            "aug_centre_crop_size": 110,
            "aug_centre_crop_p": 1,
            "aug_random_crop_min": 0.8,
            "aug_random_crop_max": 1,
            "aug_random_crop_p": 1,
            "aug_blurring_p": 0.1,
            "aug_blurring_kernel": 15,
            "aug_jiggle_p": 0.8,
            "aug_jiggle_amount": 0.5
        }
        aug_params = default_augmentation_params

        # Replace with user-specified params if necessary
        if augmentation_params is not None:
            for k in augmentation_params.keys():
                if k in aug_params.keys():
                    aug_params[k] = augmentation_params[k]

        aug_params['aug_blurring_sigma_min'] = int(
            2 * aug_params['aug_blurring_kernel'] / 3)
        aug_params['aug_blurring_sigma_max'] = aug_params['aug_blurring_kernel']
        
        # Set up the model

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")

        # Set up the transforms
        transform_list = [transforms.ToPILImage()]

        if grayscale:
            transform_list += [
                transforms.Grayscale(num_output_channels=3)
                ]
        transform_list += [
            transforms.ToTensor()
            ]
        
        if normalize:
            transform_list += [transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        self.transforms = transforms.Compose(transform_list)

        # Set up the augmentations
        self.augmentation_function = self.create_aug_function(aug_params)

        ### Add code for loading a saved model
        if model is None:
            model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
            if strip_layer:
                model.classifier[1] = torch.nn.Linear(
                    1280, projection_layer_size)
                model.classifier[1].weight.data.normal_(0, 0.01)
        self.model = model

        self.learner = BYOL(
            self.model,
            image_size=self.image_size,
            hidden_layer='avgpool',
            augment_fn=self.augmentation_function
        )
        self.learner.to(self.device)

        self.opt = torch.optim.Adam(
            self.learner.parameters(), lr=self.learning_rate)


        if restart and load_model == False:
            # You can't restart if the model is loaded
            load_model = True

        if load_model:
            # Load a previously run model
            print('Loading model from:')
            print(self.model_file)
            checkpoint = torch.load(self.model_file)
            self.learner.load_state_dict(checkpoint['learner_state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            self.trained = True

        if restart:
            self.start_epoch = checkpoint['epoch'] + 1
            loss_df = pd.read_csv(self.loss_file, index_col=0)
            self.train_loss = loss_df['training_loss'].to_list()
            if 'validation_loss' in loss_df.columns:
                self.val_loss = loss_df['validation_loss'].to_list()
            else:
                self.val_loss = []
            print('Restarting from epoch', self.start_epoch)

        else:
            self.start_epoch = 0
            self.train_loss = []
            self.val_loss = []


    def create_aug_function(self, aug_params):
        """
        Creates the augmentation function from dictionary aug_params.

        Parameters
        ----------
        aug_params : dict
            A dictionary of augmentation parameters to use.
        """
        augment_fn = torch.nn.Sequential(
            kornia.augmentation.RandomRotation(
                aug_params['aug_rotation_angle'], 
                p=aug_params['aug_rotation_p']),
            kornia.augmentation.RandomHorizontalFlip(
                p=aug_params['aug_flip_p']),
            kornia.augmentation.RandomVerticalFlip(
                p=aug_params['aug_flip_p']),
            kornia.augmentation.CenterCrop(
                size=aug_params['aug_centre_crop_size'], 
                align_corners=True, 
                p=aug_params['aug_centre_crop_p']),
            kornia.augmentation.RandomResizedCrop(
                (self.image_size, self.image_size), 
                scale=(aug_params['aug_random_crop_min'], aug_params['aug_random_crop_max']),
                p=aug_params['aug_random_crop_p']),
            kornia.augmentation.RandomGaussianBlur(
                kernel_size=[
                    aug_params['aug_blurring_kernel'], 
                    aug_params['aug_blurring_kernel']], 
                sigma=[
                    aug_params['aug_blurring_sigma_min'], 
                    aug_params['aug_blurring_sigma_max']], 
                p=aug_params['aug_blurring_p']),
            kornia.augmentation.ColorJiggle(
                brightness=aug_params['aug_jiggle_amount'], 
                contrast=aug_params['aug_jiggle_amount'], 
                saturation=aug_params['aug_jiggle_amount'], 
                hue=aug_params['aug_jiggle_amount'],
                p=aug_params['aug_jiggle_p'])
            )
        return augment_fn

    def train_byol(self, training_image_dataset, 
                   validation_image_dataset=None):
        """
        Trains the BYOL model on an image dataset. 

        Parameters
        ----------
        training_image_dataset : An Astronomaly ImageDataset object
            The dataset to train the model on.
        validation_image_dataset : An Astronomaly ImageDataset object
            The dataset to validate the model on. If None, the model will not
            be validated.
        """
        self.model.train()

        train_dataset = AstronomalyDataset(
            training_image_dataset, self.transforms)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers)

        if validation_image_dataset is not None:
            validation_dataset = AstronomalyDataset(
                validation_image_dataset, self.transforms)

            validation_loader = DataLoader(
                validation_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=self.num_workers)

        # TRAIN THE MODEL
        
        t1 = time.perf_counter()

        

        for epoch in range(self.start_epoch, self.n_epochs):
            loss_ = 0.0
            for i, image_batch in enumerate(train_loader):
                image_batch = image_batch[0]
                # send image to device
                images = image_batch.to(self.device)
                loss = self.learner(images)

                # optimization steps
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.learner.update_moving_average() 
                loss_ += loss.item()
            self.train_loss.append(loss_ / len(train_loader))

            t2 = (time.perf_counter() - t1) / 60
            print(f"Epoch: {epoch}, "
                    f"Training Loss: {self.train_loss[-1]:.6g}, "
                    f"Total time taken: {t2:.2f} minutes")

            if validation_image_dataset is not None:
                val_loss_ = 0.0
                with torch.no_grad():
                    self.learner.eval()
                    for i, image_batch in enumerate(validation_loader):
                        image_batch = image_batch[0]
                        images = image_batch.to(self.device)
                        loss = self.learner(images)
                        val_loss_ += loss.item()
                self.learner.train()
                self.val_loss.append(val_loss_ / len(validation_loader))

                if epoch % 10 == 0:
                    print(f"Epoch: {epoch} Validation Loss: {self.val_loss[-1]}")
            if self.save_model:
                torch.save({
                        'epoch': epoch,
                        'learner_state_dict': self.learner.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'loss': loss,
                        }, self.model_file)
                loss_output = np.column_stack((
                    np.arange(epoch+1), self.train_loss))
                columns = ['epoch', 'training_loss']
                if len(self.val_loss) > 0:
                    loss_output = np.column_stack((loss_output, self.val_loss))
                    columns += ['validation_loss']
                df = pd.DataFrame(data=loss_output, columns=columns)
                df.to_csv(self.loss_file)

        t2 = (time.perf_counter() - t1) / 60
        print(f'Time taken for {self.n_epochs} epochs: {t2} min')

        self.trained = True

    def extract_features_batched(self, image_dataset):
        """
        Extracts features on a full dataset at once, making use of pytorch's
        DataLoader.

        Parameters
        ----------
        image_dataset : Astronomaly ImageDataset object
            The dataset to extract features for.

        Returns
        -------
        array
            Contains the extracted deep features
        """

        if not self.trained:
            raise ValueError("""Model is untrained. Please train the model 
            first using the train_byol function or load a previously trained 
            model.""")

        extraction_dataset = AstronomalyDataset(
            image_dataset, self.transforms)
        
        data_loader = DataLoader(
            extraction_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers)
        
        all_feats = []
        self.learner.eval()

        t1 = time.perf_counter()

        with torch.no_grad():
            for i, image_batch in enumerate(data_loader):
                image_batch = image_batch[0]
                images = image_batch.to(self.device)

                # Run model on batch
                _, embedding = self.learner(images, return_embedding=True)

                # Convert to numpy and collect
                feats = embedding.detach().cpu().numpy()
                all_feats.append(feats)

                t2 = (time.perf_counter() - t1) / 60
                if i % 100 == 0:
                    print(f"Processed batch {i+1}/{len(data_loader)}")
                    print(f"Total time taken: {t2:.2f} minutes")
                    print()

        # Concatenate all batches into one array
        all_feats = np.concatenate(all_feats, axis=0)

        if len(self.labels) == 0:
            self.labels = [f'feat{i}' for i in range(len(all_feats[0]))]

        return pd.DataFrame(
            data=all_feats, index=image_dataset.index, columns=self.labels)

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
        if not self.trained:
            raise ValueError("""Model is untrained. Please train the model 
            first using the train_byol function or load a previously trained 
            model.""")

        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        if len(image.shape) > 2 and image.shape[-1] <= 4:
            # cvread reads in images with channel last but torch needs
            # channel first
            image = image.permute(2, 0, 1)
        
        processed_image = self.transforms(image)
        # Add the extra alpha channel the nets expect
        processed_image = torch.unsqueeze(processed_image, 0).to(self.device)

        # Run the model, detach from the GPU, turn it into a numpy array
        # and remove superfluous dimensions (which will likely be a different)
        # number for different models
        self.learner.eval()
        projection, embedding = self.learner(
            processed_image, return_embedding=True)
        feats = embedding.detach().cpu().numpy().squeeze()

        if len(self.labels) == 0:
            self.labels = [f'feat{i}' for i in range(len(feats))]

        return feats
