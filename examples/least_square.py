import torch

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch_optimizer as optim

import matplotlib.pyplot as plt
import numpy as np

def generate_data(num_samples=1000, num_features=3, noise=0.1):
    torch.manual_seed(42)
    X = torch.randn(num_samples, num_features)
    true_w = torch.randn(num_features, 1)
    y = X @ true_w + noise * torch.randn(num_samples, 1)
    return X, y, true_w


def least_squares_loss(X, y, w):
    loss = torch.mean((y - X @ w) ** 2) / 2
    return loss

def optimize(X, y, num_iters=1000, learning_rate=0.01, optimizer_type="SGD"):

    w = torch.randn(X.shape[1], 1, requires_grad=True)

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD([w], lr=learning_rate)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam([w], lr=learning_rate)
    elif optimizer_type == "LBFGS":
        optimizer = torch.optim.LBFGS([w], lr=learning_rate, max_iter=100)
    elif optimizer_type == "OSGM":
        optimizer = optim.OSGM([w], lr=learning_rate)
    elif optimizer_type == "OSMM":
        optimizer = optim.OSMM([w], lr=learning_rate)
    elif optimizer_type == "Ada_OSGM":
        optimizer = optim.Ada_OSGM([w], lr=learning_rate)
    elif optimizer_type == "Adafactor":
        optimizer = optim.Adafactor([w], lr=learning_rate)
    elif optimizer_type == "Adahessian":
        optimizer = optim.Adahessian([w], lr=learning_rate)

    losses = []
    gradients = []

    for i in range(num_iters):
        optimizer.zero_grad()
        def closure():
            loss = least_squares_loss(X, y, w)
            return loss
        loss = closure()

        if optimizer_type == "Adahessian":
            loss.backward(create_graph=True, retain_graph=True)
        else:
            loss.backward()

        optimizer.step(closure)

        losses.append(loss.item())
        gradients.append(w.grad.clone())

        if i % 49 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

    return w, losses, gradients

def plot_losses(losses, title="Loss Curve"):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='Loss')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('logistic.png')

def plot_gradients(gradients, title="Gradient Distribution"):
    plt.figure(figsize=(12, 8))
    for i in range(10):
        print(gradients[i].detach().numpy())
        plt.hist(gradients[i].detach().numpy(), bins=30, alpha=0.5, label=f'Iter {i}')
    plt.title(title)
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('grad.png')


if __name__ == "__main__":

    X, y, true_w = generate_data(num_samples=1000, num_features=4, noise=0.1)
    print(f"True weights: {true_w.squeeze().numpy()}")

    num_iters = 100
    learning_rate = 0.5

    # optimizer_type = "Adam"
    optimizer_type = "OSGM"
    # optimizer_type = "Adafactor"
    # optimizer_type = "Adahessian"
    # optimizer_type = "SGD"
    # optimizer_type = "OSMM"

    w, losses, gradients = optimize(X, y, num_iters=num_iters, learning_rate=learning_rate, optimizer_type=optimizer_type)

    print(f"Optimized weights: {w.squeeze().detach().numpy()}")

    np.save('cvg_plot/least_square/{}.npy'.format(optimizer_type),losses)