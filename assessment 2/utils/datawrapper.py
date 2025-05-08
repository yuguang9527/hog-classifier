# Copyright (C), Weights & Biases
# Please DO NOT share.

import numpy as np
import torch
from torch.utils import data

from utils.cifar10 import load_data
from utils.features import extract_h_histogram, extract_hog


class CIFAR10Dataset(data.Dataset):
    """The dataset wrapper for CIFAR 10.

    While writing this class is completely unecessary if we were to obtain the
    CIFAR10 data in raw format since PyTorch already has it written. However,
    in our case we will also try to extract data, we will implement it
    ourselves.

    """

    def __init__(self, config, mode):
        """Initialization.
        
        Parameters
        ----------

        config:
            Configuration object that holds the command line arguments.
        
        mode: str
            A string detailing which split for this dataset object to load. For
            example, train, valid, test.
        

        Notes
        -----

        By default, we assume the dataset is at downloads

        """

        # Save configuration
        self.config = config

        print("Loading CIFAR10 Dataset from {} for {}ing ...".format(
            config.data_dir, mode), end="")

        # Load data
        data, label = load_data(config.data_dir, mode)
        self.data = data
        self.label = label

        print(" done.")

        # Extract features and update `self.data`, and at the same time save
        # the shape of each sample (not the entire data shape!)  to
        # `self.sample_shp`.
        # 
        # It turns out extracting data on-the-fly on a normal PC takes longer!
        # Since CIFAR10 is not a large dataset, we'll do it here. With large
        # datasets, we don't have this option.
        #
        # Also store the shape of each sample (depending on the feature
        # extraction that we apply)
        print("Extracting features for the entire {} set ... ".format(
            mode), end="")
        if self.config.feature_type == "hog":
            # HOG features
            self.data = extract_hog(self.data)
            self.sample_shp = (324,)
        elif self.config.feature_type == "h_histogram":
            # Hue Histogram features
            self.data = extract_h_histogram(self.data)
            self.sample_shp = (16,)
        elif self.config.feature_type == "rgb":
            # Flattened RGB features
            self.data = self.data.astype(float).reshape(len(self.data), -1)
            self.sample_shp = (np.prod(data.shape[1:]),)
        elif self.config.feature_type == "raw":
            # Raw data as is
            self.sample_shp = data.shape[1:]
        print("done.")

    def __len__(self):
        """ Returns number of samples. """

        # return the number of elements at `self.data`
        return len(self.data)

    def __getitem__(self, index):
        """Function to grab one data sample 
        
        Parameters
        ----------
        
        index: int
            Index to the sample that we are trying to extract.

        
        Returns
        -------

        data_cur: torch.Tensor
            A torch tensor that holds the data. For GPU compatibility in the
            future, we will develop the habit of converting everythiing in to
            `float32`
        
        label_cur: int
            The label of the current data sample. This is typically a simple
            indexing operation to the label array

        """

        # Grab one data from the dataset and then apply our feature extraction
        data_cur = self.data[index]

        # Make pytorch object
        data_cur = torch.from_numpy(data_cur.astype(np.float32))

        # Label is just the label
        label_cur = self.label[index]

        return data_cur, label_cur


#
# datawrapper.py ends here
