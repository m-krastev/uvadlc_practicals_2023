################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from tqdm.auto import tqdm

from cifar100_utils import add_augmentation, get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # set to eval in other to not calculate grads for other layers
    for param in model.parameters():
        param.requires_grad = False

    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.fc = nn.Linear(model.fc.weight.shape[1], num_classes)

    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(
    model,
    lr,
    batch_size,
    epochs,
    data_dir,
    checkpoint_name,
    device,
    augmentation_name=None,
    debug=False,
):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train, val = get_train_validation_set(data_dir, augmentation_name=augmentation_name)
    dataloader_train = data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    dataloader_val = data.DataLoader(
        val, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Initialize the optimizer (Adam) to train the last layer of the model.
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    model.to(device)

    # Training loop with validation after each epoch. Save the best model.

    val_accuracies = []
    for i in range(epochs):
        model.fc.train()
        for batch in tqdm(dataloader_train):
            x, y = batch
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss_val = loss(out, y)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        # Evaluate the model on the validation set.
        val_accuracy = evaluate_model(model, dataloader_val, device)
        val_accuracies.append(val_accuracy)
        if val_accuracy == max(val_accuracies):
            torch.save(model.state_dict(), checkpoint_name)
        if debug:
            print(f"Epoch {i+1}/{epochs}: Validation accuracy: {val_accuracy}")

    # Load the best model on val accuracy and return it.
    model.load_state_dict(torch.load(checkpoint_name))

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy

    predicts = []
    targets = []
    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            out: torch.Tensor = model(x)
            predicts.append(out.argmax(dim=-1))
            targets.append(y)

    predicts = torch.cat(predicts)
    targets = torch.cat(targets)
    accuracy = (predicts == targets).float().mean()

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(
    lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise, debug=True,checkpoint_name="best_model.pth"
):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load the model
    model = get_model()

    # Get the augmentation to use ?? what do you expect me to do

    # Train the model
    model = train_model(
        model,
        lr,
        batch_size,
        epochs,
        data_dir,
        checkpoint_name,
        device,
        augmentation_name,
        debug,
    )

    # Evaluate the model on the test set
    test_set = get_test_set(data_dir, test_noise)

    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_accuracy = evaluate_model(model, test_loader, device)

    if debug:
        print(f"Test accuracy: {test_accuracy}")

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")
    parser.add_argument("--epochs", default=30, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=123, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR100 dataset.",
    )
    parser.add_argument(
        "--augmentation_name", default=None, type=str, help="Augmentation to use."
    )
    parser.add_argument(
        "--test_noise",
        default=False,
        action="store_true",
        help="Whether to test the model on noisy images or not.",
    )
    parser.add_argument(
        "--debug",
        default=True,
        action="store_true",
    )
    
    parser.add_argument("--checkpoint_name", default="best_model.pth", type=str, help="Name of the checkpoint to save the best model on validation")

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
