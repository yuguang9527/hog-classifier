# Filename: solution.py

# Copyright (C), Weights & Biases
# Please DO NOT share.

import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from config import get_config, print_usage
from model import MyNetwork
from tensorboardX import SummaryWriter
from utils.cifar10 import load_data
from utils.datawrapper import CIFAR10Dataset
from utils.features import extract_h_histogram, extract_hog
from utils.preprocess import normalize
from torchvision import datasets
from wandb_utils import init_wandb, log_test_metrics, save_model_artifact, finish_wandb

def data_criterion(config):
    """Returns the loss object based on the commandline argument for the data term

    """

    if config.loss_type == "cross_entropy":
        data_loss = nn.CrossEntropyLoss()
    elif config.loss_type == "svm":
        data_loss = nn.MultiMarginLoss()

    return data_loss


def model_criterion(config):
    """Loss function based on the commandline argument for the regularizer term"""

    def model_loss(model):
        loss = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                loss += torch.sum(param**2)

        return loss * config.l2_reg

    return model_loss


def train(config):
    """
    Training process.

    """

    trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True)

    # Initialize datasets for both training and validation
    train_data = CIFAR10Dataset(
        config, mode="train",
    )
    valid_data = CIFAR10Dataset(
        config, mode="valid",
    )

    # We'll load the mean and the standard deviation of the data that we
    # compute previously
    x_tr_mean = np.load("{}_mean.npy".format(config.feature_type))
    x_tr_std = np.load("{}_std.npy".format(config.feature_type))

    # Create data loader for training and validation.
    tr_data_loader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=True
    )
    va_data_loader = DataLoader(
        dataset=valid_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=False
    )

    # Create model instance. We will use `train_data.sample_shp`, mean and std
    # from training data for initialization.
    model = MyNetwork(
        config=config,
        input_shp=train_data.sample_shp,
        mean=x_tr_mean, std=x_tr_std)
    # Move model to gpu if cuda is available
    if torch.cuda.is_available():
        model = model.cuda()
    # Make sure that the model is set for training
    model.train()

    #
    # ALTHOUGH NOT IMPLEMENTED IN THE THIS TEST, SEE BELOW ON HOW WE CONFIGURE
    # OUR LOSS FUNCTIONS
    #
    # Create loss objects
    data_loss = data_criterion(config)
    model_loss = model_criterion(config)

    #
    # ALTHOUGH NOT IMPLEMENTED IN THE THIS TEST, SEE BELOW ON HOW WE CREATE AN
    # OPTIMIZER
    #
    # Create optimizier
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # No need to move the optimizer (as of PyTorch 1.0), it lies in the same
    # space as the model

    #
    # ALTHOUGH NOT IMPLEMENTED IN THE THIS TEST, SEE BELOW ON HOW WE SETUP OUR
    # WRITERS SO THAT THEY WRITE TO A DIFFERENT SUB FOLDER. THIS ALLOWS US TO
    # EASILY DISPLAY SCALAR ON A SINGLE GRAPH IN TENSORBOARD.
    #
    # Create summary writer
    tr_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "train"))
    va_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "valid"))

    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Initialize training
    iter_idx = -1  # make counter start at zero
    best_va_acc = 0  # to check if best validation accuracy
    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(config.save_dir, "checkpoint.pth")
    bestmodel_file = os.path.join(config.save_dir, "best_model.pth")

    # Check for existing training results. If it existst, and the configuration
    # is set to resume `config.resume==True`, resume from previous training. If
    # not, delete existing checkpoint.
    if os.path.exists(checkpoint_file):
        if config.resume:
            # Use `torch.load` to load the checkpoint file and the load the
            # things that are required to continue training. For the model and
            # the optimizer, use `load_state_dict`. It's actually a good idea
            # to code the saving part first and then code this part.
            print("Checkpoint found! Resuming")
            # Read checkpoint file.
            load_res = torch.load(checkpoint_file, weights_only=False) # Load the checkpoint file and the load the things that are required to continue training.
            # Resume iterations
            iter_idx = load_res["iter_idx"]
            # Resume best va result
            best_va_acc = load_res["best_va_acc"]
            # Resume model
            model.load_state_dict(load_res["model"])
            # Resume optimizer
            optimizer.load_state_dict(load_res["optimizer"])
            # Note that we do not resume the epoch, since we will never be able
            # to properly recover the shuffling, unless we remember the random
            # seed, for example. For simplicity, we will simply ignore this,
            # and run `config.num_epoch` epochs regardless of resuming.
        else:
            os.remove(checkpoint_file)

    # Training loop
    for epoch in range(config.num_epoch):
        # For each iteration
        prefix = "Training Epoch {:3d}: ".format(epoch)

        #
        # ALTHOUGH NOT IMPLEMENTED IN THE THIS TEST, SEE BELOW ON HOW WE USE
        # OUR DATALOADER TO LOAD DATA. I'VE ALSO WRAPPED IT IN TQDM FOR
        # EYECANDY.
        #
        for data in tqdm(tr_data_loader, desc=prefix):
            # Counter
            iter_idx += 1

            # Split the data
            x, y = data

            # Send data to GPU if we have one
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Implement the forward pass. Apply
            # `model.forward` to get the scores, then use `data_loss` and
            # `model_loss` to compute the total loss. Then compute the
            # gradients by performing the backward pass, apply the parameter
            # update using the optimizer, and finally prepare the optimizer for
            # the next iteration.

            # Apply the model to obtain scores (forward pass)
            logits = model.forward(x)
            # Compute the loss
            y = torch.tensor(y, dtype = torch.long)
            loss = data_loss(logits, y) + model_loss(model)
            # Compute gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            # Zero the parameter gradients in the optimizer
            optimizer.zero_grad()


            # Monitor results every report interval
            if iter_idx % config.rep_intv == 0:
                # Compute accuracy (No gradients required). We'll wrapp this
                # part so that we prevent torch from computing gradients.
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    acc = torch.mean(torch.eq(pred, y).float()) * 100.0
                # Write loss and accuracy to tensorboard, using keywords `loss`
                # and `accuracy`.
                tr_writer.add_scalar("loss", loss, global_step=iter_idx)
                tr_writer.add_scalar("accuracy", acc, global_step=iter_idx)
                # Let's also save our most recent model so that we can
                # resume. We'll save this model in path defined by
                # `checkpoint_file`.  Note that in order to resume, we would
                # need to save, `iter_idx`, `best_va_acc`, `model`, and
                # `optimizer`. Note that for simplicity, we will not save
                # epoch.
                torch.save({
                    "iter_idx": iter_idx,
                    "best_va_acc": best_va_acc,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, checkpoint_file)

            # Validate results every validation interval
            if iter_idx % config.val_intv == 0:
                # List to contain all losses and accuracies for all the
                # training batches
                va_loss = []
                va_acc = []
                # Set model for evaluation
                model = model.eval()
                #
                # NOTE HOW WE ARE LOOPING OVER THE ENTIRE VALIDATION SET HERE.
                # 
                for data in va_data_loader:

                    # Split the data
                    x, y = data

                    # Send data to GPU if we have one
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()

                    # Apply forward pass to compute the losses
                    # and accuracies for each of the validation batches
                    with torch.no_grad():
                        # Compute logits
                        logits = model.forward(x)
                        # Compute loss and store as numpy
                        y = torch.tensor(y, dtype = torch.long)
                        loss = data_loss(logits, y) + model_loss(model)
                        va_loss += [loss.numpy()]
                        
                        pred = torch.argmax(logits, dim=1)

                        # TODO: Compute accuracy and store as numpy
                        #try:
                            #raise Exception("Complete the TODO: calculate accuracy for val batches")
                        acc = torch.mean(torch.eq(pred, y).float()) * 100.0 # Calculate accuracy for val batches.
                        va_acc += [acc.cpu().numpy()] #Move to CPU and convert to numpy.
                        #except Exception as error:
                            #print('Caught this error: ' + error)

                # Set model back for training
                model = model.train()
                # Take average
                va_loss = np.mean(va_loss)
                va_acc = np.mean(va_acc)

                # Write the average `loss` and `accurac` to
                # the validation writer. Also check if the best validation
                # accuracy is achieved, and save the model and all necessary
                # states using `torch.save` as `bestmodel_file`

                # Write to tensorboard using `va_writer`
                va_writer.add_scalar("loss", va_loss, global_step=iter_idx)
                va_writer.add_scalar("accuracy", va_acc, global_step=iter_idx)
                # Check if best accuracy
                if va_acc > best_va_acc:
                    best_va_acc = va_acc
                    # Save best model using torch.save. Similar to previous
                    # save but at location defined by `bestmodel_file`
                    torch.save({
                        "iter_idx": iter_idx,
                        "best_va_acc": best_va_acc,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, bestmodel_file)
                    save_model_artifact(bestmodel_file) # Save the best model to W&B.   


def test(config):
    """Test routine"""

    # TODO: Initialize Dataset for testing.
    #try:
        #raise Exception('Implement the TODO: initialize data for testing')
    test_data = CIFAR10Dataset( 
        config, mode="test",
    )
    #except Exception as error:
        #print('Caught this error: ' + error)

    # Load the mean and the standard deviation of the data that we compute
    # previously
    x_tr_mean = np.load("{}_mean.npy".format(config.feature_type))
    x_tr_std = np.load("{}_std.npy".format(config.feature_type))

    # Create data loader for the test dataset with two number of workers and no
    # shuffling.
    te_data_loader = DataLoader(
        dataset=test_data,
        batch_size=config.batch_size,
        num_workers=2,
        shuffle=False
    )

    # Create model
    model = MyNetwork(
        config=config,
        input_shp=test_data.sample_shp,
        mean=x_tr_mean,
        std=x_tr_std # Use mean and std from training data for initialization.
        )
    #print(model)
    # Move to GPU if you have one.
    if torch.cuda.is_available():
        model = model.cuda()

    # Create loss objects
    data_loss = data_criterion(config)
    model_loss = model_criterion(config)

    # TODO: Load our best model and set model for testing
    #try:
        #raise Exception('Implement TODO: load best model')
    bestmodel_file = os.path.join(config.save_dir, "best_model.pth")
    #except Exception as error:
        #print('Caught this error: ' + error)

    load_res = torch.load(bestmodel_file, weights_only=False)
    model.load_state_dict(load_res["model"])
    model.eval()

    prefix = "Testing: "
    te_loss = []
    te_acc = []
    for data in tqdm(te_data_loader, desc=prefix):

        # Split the data
        x, y = data

        # Send data to GPU if we have one
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # Apply forward pass to compute the losses
        # and accuracies for each of the test batches
        with torch.no_grad():
            # Compute logits
            logits = model.forward(x)
            # Compute loss and store as numpy
            y = torch.tensor(y, dtype = torch.long)
            loss = data_loss(logits, y) + model_loss(model)
            te_loss += [loss.numpy()]
            # Compute accuracy and store as numpy
            pred = torch.argmax(logits, dim=1)
            acc = torch.mean(torch.eq(pred, y).float()) * 100.0
            te_acc += [acc.numpy()]
    
    # Take average
    te_loss = np.mean(te_loss)
    te_acc = np.mean(te_acc)

    # Report Test loss and accuracy
    print("Test Loss = {}".format(np.mean(te_loss)))
    print("Test Accuracy = {}%".format(np.mean(te_acc)))
    log_test_metrics(np.mean(te_loss), np.mean(te_acc))


def main(config):
    """The main function."""

    init_wandb(config, config.mode)

    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))
    
    finish_wandb()


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)