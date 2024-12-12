import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import math

import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch_optimizer as optim

import matplotlib.pyplot as plt
import argparse

def get_data(dataset):
    data_path = f"data/LIBSVM/{dataset}"
    X, y = load_svmlight_file(data_path)
    X = X.toarray()

    if not np.issubdtype(y.dtype, np.integer):
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])
    n_classes = len(np.unique(y))

    print(f"Dataset: {dataset}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    y_tensor = convert_to_one_hot(y_tensor, n_classes)

    return X_tensor, y_tensor, n_classes

def convert_to_one_hot(y, num_classes):
    # Use torch's one-hot encoding function
    one_hot_encoded = torch.nn.functional.one_hot(y, num_classes=num_classes)
    
    return one_hot_encoded.float()

def softmax(logits, lib):
    logits_exp = lib.exp(logits - lib.max(logits, dim=1, keepdim=True).values)
    return logits_exp / lib.sum(logits_exp, dim=1, keepdim=True)

def softmax_loss(W, X, Y, lambda_reg, lib=torch):
    logits = X @ W
    probs = softmax(logits, lib)
    log_loss = -lib.mean(lib.sum(Y * lib.log(probs + 1e-8),dim=1))
    regularization = 0.5 * lambda_reg * lib.sum(W ** 2)
    return log_loss + regularization

def benchmark_optimizer_logistic(optimizer_name, X, y, n_classes, args, seed=42):
    batches = args.batches
    lambda_reg = args.lambda_reg
    epochs = args.epochs

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    train_dataset = TensorDataset(X_train, y_train)
    batch_size = math.ceil(X_train.shape[0] / batches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    w = torch.zeros(X_train.shape[1], n_classes, requires_grad=True)
    optimizers = {
        "SGD": torch.optim.SGD([w], lr=0.5/math.sqrt(batches)),
        "NAG": torch.optim.SGD([w],lr=0.008/batches,momentum=0.9,nesterov=True),
        "Adam": torch.optim.Adam([w], lr=0.03/batches, betas=(0.9, 0.999)),
        "OSGM": optim.OSGM([w], lr=0.5/batches),
        "OSMM": optim.OSMM([w], lr=0.5/batches, weight_decay=0, stop_step=epochs/3*batches*100, stop_beta=0.9),
        "AdamHD": optim.AdamHD([w], lr=0.1/batches, hypergrad_lr=1e-8),
        "SGDHD": optim.SGDHD([w], lr=0.1/math.sqrt(batches), hypergrad_lr=1e-6),
    }
    optimizer = optimizers[optimizer_name]

    scheduler = ExponentialLR(optimizer, gamma=0.99) # exponential decay

    losses = []

    for epoch in range(epochs):
        loss = softmax_loss(w, X_train, y_train, lambda_reg, torch) # compute loss on the whole training set
        losses.append(loss.item())

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(train_loader):
            next_inputs, next_labels = next(iter(train_loader))
            def closure():
                loss = softmax_loss(w, next_inputs, next_labels, lambda_reg, torch)
                return loss
            
            loss = softmax_loss(w, inputs, labels, lambda_reg, torch)
            loss.backward()
            optimizer.step(closure)

        if batches > 1:
            scheduler.step()
        
    test_loss = softmax_loss(w, X_test, y_test,lambda_reg=0)
    print(f"{optimizer_name} Test Loss: {test_loss:.2f}")

    return w.detach().numpy(), losses

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark different optimizers for logistic regression.")
    parser.add_argument("--lambda_reg", type=float, default=1e-4, help="Regularization parameter.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batches", type=int, default=60, help="Number of batches.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    lambda_reg = args.lambda_reg
    epochs = args.epochs
    batches = args.batches

    datasets = [
        # "dna.scale",
        # "glass.scale",
        # "iris.scale",
        # "vehicle.scale",
        # "vowel.scale",
        # "wine.scale",
        "mnist.scale.bz2",
    ]

    for dataset in datasets:
        X_tensor, y_tensor, n_classes = get_data(dataset)

        optimizer_names = [
                        # "SGD",
                        "Adam",
                        "NAG",
                        # "OSGM",
                        # "OSMM",
                        # "AdamHD",
                        # "SGDHD",
                        ]
        
        for optimizer_name in optimizer_names:
            losses = []
            for seed in range(0,1):
                w, loss = benchmark_optimizer_logistic(optimizer_name, X_tensor, y_tensor, n_classes,\
                                                        args, seed=seed)
                losses.append(loss)
            
            losses = np.array(losses)
            print(losses.shape)

            save_path = f'cvg_plot/stoc_logistic_{batches}/{dataset}'

            os.makedirs(save_path, exist_ok=True)
            path = f'{save_path}/{optimizer_name}.npy'
            print(f"Saving results to {path}")
            np.save(path, losses)
