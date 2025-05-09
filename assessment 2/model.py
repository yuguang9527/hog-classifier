# Filename: model.py

# Copyright (C), Weights & Biases
# Please DO NOT share.

import numpy as np
import torch
from torch import nn


class MyNetwork(nn.Module):
    """
    Neural network class for CIFAR10 classification
    """
    def __init__(self, config, input_shp, mean=None, std=None):
        """
        Initialize the neural network.
        
        Args:
            config: Config object with hyperparameters
            input_shp: Shape of the input features
            mean: Mean for normalizing the input (optional)
            std: Standard deviation for normalizing the input (optional)
        """
        super(MyNetwork, self).__init__()
        
        # Store the mean and std for normalization
        self.mean = mean
        self.std = std
        
        # Get the input size (flatten the input shape)
        input_size = np.prod(input_shp)
        
        # Define the network architecture
        # hidden_size = getattr(config, 'hidden_size', 256)  # Default to 256 if not specified
        hidden_size = config.num_unit # Use num_unit from config.
        #num_classes = 10  # CIFAR10 has 10 classes
        num_classes = config.num_class # Use num_class from config.
        num_hidden_layers = config.num_hidden # Use num_hidden from config.

        # Create a simple network with layers matching the saved checkpoint
        # self.linear_0 = nn.Linear(input_size, hidden_size)
        # self.linear_1 = nn.Linear(hidden_size, hidden_size)
        #self.linear_2 = nn.Linear(hidden_size, hidden_size)
        #self.output = nn.Linear(hidden_size, num_classes)
        
        # Activation function
        # self.relu = nn.ReLU()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU() if config.activ_type == 'relu' else nn.Tanh())

        # Hidden layers
        for _ in range(max(0, num_hidden_layers - 1)): 
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU() if config.activ_type == 'relu' else nn.Tanh())

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes) # define the output layer
 
        self.hidden_module = nn.Sequential(*layers) # combine input and hidden layers
            

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Logits for each class
        """
        # Get the batch size
        batch_size = x.shape[0]
        
        # Flatten the input
        x = x.view(batch_size, -1)
        
        # Normalize input if mean and std are provided
        if self.mean is not None and self.std is not None:
            # Convert mean and std to tensors if they're not already
            if not torch.is_tensor(self.mean):
                self.mean = torch.tensor(self.mean, dtype=torch.float32)
                if torch.cuda.is_available():
                    self.mean = self.mean.cuda()
            
            if not torch.is_tensor(self.std):
                self.std = torch.tensor(self.std, dtype=torch.float32)
                if torch.cuda.is_available():
                    self.std = self.std.cuda()
            
            # Normalize
            x = (x - self.mean) / (self.std + 1e-8)
        
        # Forward pass through the layers
        # x = self.relu(self.linear_0(x))
        # x = self.relu(self.linear_1(x))
        # x = self.relu(self.linear_2(x))
        # x = self.output(x)
        
        # x = self.hidden_layers(x) # pass through the hidden layers
        x = self.hidden_module(x) # pass through the hidden module
        x = self.output_layer(x) # pass through the output layer
        return x


