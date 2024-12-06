import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

import torch 

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch_optimizer as optim

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

def benchmark_optimizer_logistic(optimizer_name, X, y, n_classes, lambda_reg=0.1, epochs=100, lr=0.01):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    w = torch.zeros(X_train.shape[1], n_classes, requires_grad=True)
    optimizers = {
        "SGD": torch.optim.SGD([w], lr=0.05),
        "NAG": torch.optim.SGD([w],lr=0.1,momentum=0.9,nesterov=True),
        "Adam": torch.optim.Adam([w], lr=0.01),
        "OSGM": optim.OSGM([w], lr=0.1),
        "OSMM": optim.OSMM([w], lr=0.1),
    }
    optimizer = optimizers[optimizer_name]
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        def closure():
            loss = softmax_loss(w, X_train, y_train, lambda_reg, torch)
            return loss
        
        loss = closure()
        loss.backward()
        optimizer.step(closure)
        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    test_loss = softmax_loss(w, X_test, y_test,lambda_reg=0)
    print(f"{optimizer_name} Test Loss: {test_loss:.2f}")

    return w.detach().numpy(), losses

if __name__ == "__main__":
    datasets = [
        # "dna",
        # "glass",
        # "iris",
        # "letter",
        "vehicle",
        "vowel",
        "wine",
    ]
    # dataset = "vehicle"

    for dataset in datasets:
        data_path = f"data/LIBSVM/{dataset}.scale"
        # data_path = f"/home/zhangwanyu/pytorch-optimizer/dataset/LIBSVM/{dataset}.scale"
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
        epochs = 300
        lr = 0.05

        optimizer_names = [
                        "SGD", # lr=0.05
                        "Adam", # lr=0.01
                        "NAG",
                        "OSGM", # lr=0.1
                        "OSMM" # lr=0.1
                        ]
        
        for optimizer_name in optimizer_names:
            w, losses = benchmark_optimizer_logistic(optimizer_name, X_tensor, y_tensor, n_classes, lambda_reg=lambda_reg, epochs=epochs, lr=lr)
            save_path = f'cvg_plot/logistic_regression/{dataset}'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.save(f'{save_path}/{optimizer_name}.npy', losses)
