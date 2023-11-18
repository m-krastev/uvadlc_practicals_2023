################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    conf_mat = torch.zeros((predictions.shape[1], predictions.shape[1]))
    preds = torch.argmax(predictions, axis=1)

    for indices in zip(preds, targets):
        conf_mat[indices] += 1

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.0):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    accuracy = torch.trace(confusion_matrix) / torch.sum(confusion_matrix)
    precision = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, axis=0)
    recall = torch.diag(confusion_matrix) / torch.sum(confusion_matrix, axis=1)

    f1_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_beta": f1_beta,
        "confusion_matrix": confusion_matrix,
    }

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10, device="cpu"):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    model.to(device)
    with torch.no_grad():
        preds = []
        targets = []
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            x = x.reshape(x.shape[0], -1)
            out = model.forward(x)
            preds.append(out)
            targets.append(y)
    conf_mat = confusion_matrix(torch.concatenate(preds), torch.concatenate(targets))
    metrics = confusion_matrix_to_metrics(conf_mat)

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(
    hidden_dims,
    lr,
    use_batch_norm,
    batch_size,
    epochs,
    seed,
    data_dir,
    debug=False,
    device=None,
):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
        torch.mps.seed(seed)
        torch.mps.set_rng_state(seed)

    # Set default device
    device = torch.device(
        device
        if device
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=False
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    input_size = next(cifar10_loader["train"].__iter__())[0].shape[1:]
    classes = 10

    model = MLP(
        np.prod(input_size), hidden_dims, classes, use_batch_norm=use_batch_norm
    ).to(device)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    val_accuracies = []
    train_losses = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in tqdm(cifar10_loader["train"]):
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)
            out = model.forward(x)
            loss = loss_module.forward(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_losses.append(np.mean(losses))
        val = evaluate_model(model, cifar10_loader["validation"], classes, device)[
            "accuracy"
        ]
        val_accuracies.append(val)
        if debug:
            print(
                f"Epoch {epoch+1:>2} \t loss: {np.mean(losses):>4.4f} \t Accuracy: {val:>4.4%}"
            )
        # save the best performing model
        if val >= max(val_accuracies):
            best_model = model.state_dict()

    _ = model.load_state_dict(best_model)
    test_metrics = evaluate_model(model, cifar10_loader["test"], device=device)
    test_accuracy = test_metrics["accuracy"]
    logging_info = {"train_losses": train_losses, "test_metrics": test_metrics}
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--use_batch_norm",
        action="store_true",
        help="Use this option to add Batch Normalization layers to the MLP.",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR10 dataset.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use this option to print debug information.",
    )

    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device to use for training. Either 'cpu', 'cuda' or 'mps'.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)

    # Feel free to add any additional functions, such as plotting of the loss curve here
    if args.debug:
        import matplotlib.pyplot as plt

        os.makedirs("plots", exist_ok=True)

        plt.figure()
        plt.plot(val_accuracies)
        plt.title("Validation accuracy (best: {:.2%})".format(max(val_accuracies)))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("plots/val_acc_pt.png")

        plt.figure()
        plt.plot(logging_info["train_losses"])
        plt.title(
            "Training loss (final: {:.4f})".format(logging_info["train_losses"][-1])
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("plots/train_loss_pt.png")

        plt.figure()
        plt.imshow(logging_info["test_metrics"]["confusion_matrix"])
        plt.xticks(np.arange(len(logging_info["test_metrics"]["confusion_matrix"])))
        plt.yticks(np.arange(len(logging_info["test_metrics"]["confusion_matrix"])))
        plt.title("Confusion matrix")
        plt.colorbar()
        plt.savefig("plots/confusion_matrix_pt.png")

        print("Test accuracy: {:.2%}".format(test_accuracy))
        precision = logging_info["test_metrics"]["precision"].numpy()
        recall = logging_info["test_metrics"]["recall"].numpy()
        print("Precision: ", precision)
        print("Recall: ", recall)
        for beta in [0.1, 1, 10]:
            f1_score = (
                (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            )
            print(f"F1 score for beta = {beta}: {f1_score}")
