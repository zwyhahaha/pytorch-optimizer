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

class SqrtLRScheduler:
    def __init__(self, optimizer, step_size=1):
        self.optimizer = optimizer
        self.step_num = 1
        self.step_size = step_size

    def step(self):
        self.step_num += 1
        if self.step_num % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= ((self.step_num-1) ** 0.5)/(self.step_num ** 0.5)

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

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

def benchmark_optimizer_logistic(optimizer_name, X, y, n_classes, lambda_reg=0.1, epochs=100, lr=0.01, batches=1):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    batch_size = math.ceil(X_train.shape[0] / batches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    w = torch.zeros(X_train.shape[1], n_classes, requires_grad=True)
    optimizers = {
        "SGD": torch.optim.SGD([w], lr=0.05/math.sqrt(batches)),
        "NAG": torch.optim.SGD([w],lr=0.01/batches,momentum=0.9,nesterov=True),
        "Adam": torch.optim.Adam([w], lr=0.05/batches),
        "OSGM": optim.OSGM([w], lr=0.01/batches),
        "OSMM": optim.OSMM([w], lr=0.1/batches),
        "AdamHD": optim.AdamHD([w], lr=0.1/batches, hypergrad_lr=1e-8),
        "SGDHD": optim.SGDHD([w], lr=0.1/math.sqrt(batches), hypergrad_lr=1e-6),
    }
    optimizer = optimizers[optimizer_name]

    if optimizer_name == "SGD":
        scheduler = StepLR(optimizer, step_size = 10, gamma=1.0) # constant learning rate
    elif optimizer_name == "OSMM":
        scheduler = SqrtLRScheduler(optimizer, step_size=1)
    else:
        scheduler = ExponentialLR(optimizer, gamma=0.9) # exponential decay

    losses = []

    for epoch in range(epochs):
        loss = softmax_loss(w, X_train, y_train, lambda_reg, torch) # compute loss on the whole training set
        losses.append(loss.item())

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(train_loader):
            # use the next batch to define monotone oracle
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

if __name__ == "__main__":
    datasets = [
        "dna.scale",
        # "glass.scale",
        # "iris.scale",
        # "vehicle.scale",
        # "vowel.scale",
        # "wine.scale",
        # "mnist.scale.bz2",
    ]

    for dataset in datasets:
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
        print(f"Number of classes: {n_classes}")
        print(f"Unique class labels: {np.unique(y)}")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        y_tensor = convert_to_one_hot(y_tensor, n_classes)

        lambda_reg = 0.01
        epochs = 40
        lr = 0.05
        batches = 15

        optimizer_names = [
                        "SGD", # lr=0.05
                        "Adam", # lr=0.01
                        "NAG",
                        "OSGM", # lr=0.1
                        "OSMM", # lr=0.1
                        # "AdamHD",
                        # "SGDHD",
                        ]
        
        for optimizer_name in optimizer_names:
            w, losses = benchmark_optimizer_logistic(optimizer_name, X_tensor, y_tensor, n_classes,\
                                                      lambda_reg=lambda_reg, epochs=epochs, lr=lr,batches=batches)
            save_path = f'cvg_plot/stoc_logistic_{batches}/{dataset}'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.save(f'{save_path}/{optimizer_name}.npy', losses)
