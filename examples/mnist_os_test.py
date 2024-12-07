import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch_optimizer as optim
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(conf, model, device, train_loader, optimizer, epoch, writer, optimizer_name):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        def closure():
            output = model(data)
            loss = F.nll_loss(output, target)
            return loss 
        loss = closure()

        if optimizer_name == "Adahessian":
            loss.backward(create_graph=True, retain_graph=True)
        else:
            loss.backward()

        optimizer.step(closure)
        
        if batch_idx % conf.log_interval == 0:
            loss = loss.item()
            idx = batch_idx + epoch * (len(train_loader))
            writer.add_scalar("Loss/train", loss, idx)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss,
                )
            )


def test(conf, model, device, test_loader, epoch, writer, optimizer_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    fmt = "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    writer.add_scalar("Accuracy", correct, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)


def prepare_loaders(conf, use_cuda=False):
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=conf.batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=conf.test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


class Config:
    def __init__(
        self,
        batch_size: int = 64,
        test_batch_size: int = 1000,
        epochs: int = 2,
        lr: float = 0.01,
        gamma: float = 0.7,
        seed: int = 42,
        log_interval: int = 10,
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.seed = seed
        self.log_interval = log_interval

def execute_experiments(optimizers,lr,batch_size,epochs,beta,beta_lr):
    conf = Config(lr=lr,epochs=epochs,batch_size=batch_size)
    seed = conf.seed 

    for optimizer_class in optimizers:
        optimizer_name = optimizer_class.__name__
        log_dir = "runs/mnist_{}".format(optimizer_name)
        print("Tensorboard: tensorboard --logdir={}".format(log_dir))

        save_path = f"cvg_plot/mnist_batch{conf.batch_size}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # f = open('{}/{}_lr{}_beta{}_beta_lr_{}.txt'.format(save_path,optimizer_name,conf.lr,beta,beta_lr), 'w')
        f = open('{}/{}_lr{}.txt'.format(save_path,optimizer_name,conf.lr), 'w')
        sys.stdout = f
        with SummaryWriter(log_dir) as writer:
            use_cuda = torch.cuda.is_available()
            torch.manual_seed(conf.seed)
            device = torch.device("cuda" if use_cuda else "cpu")
            train_loader, test_loader = prepare_loaders(conf, use_cuda)

            model = Net().to(device)

            # create grid of images and write to tensorboard
            images, labels = next(iter(train_loader))
            img_grid = utils.make_grid(images)
            writer.add_image("mnist_images", img_grid)

            if optimizer_class.__name__ == "SGD":
                optimizer = optimizer_class(model.parameters(), lr=conf.lr,momentum=0.9,nesterov=True)
            elif optimizer_class.__name__ == "OSMM":
                optimizer = optimizer_class(model.parameters(), lr=conf.lr,beta=beta,beta_lr=beta_lr)
            else:
                optimizer = optimizer_class(model.parameters(), lr=conf.lr)

            scheduler = StepLR(optimizer, step_size=1, gamma=conf.gamma)
            for epoch in range(1, conf.epochs + 1):
                train(conf, model, device, train_loader, optimizer, epoch, writer,optimizer_name)
                test(conf, model, device, test_loader, epoch, writer,optimizer_name)
                scheduler.step()
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, epoch)
                    writer.add_histogram("{}.grad".format(name), param.grad, epoch)
        f.close()
        sys.stdout = sys.__stdout__

def parse_args():
    """
    Parses command-line arguments for the script.

    Returns:
        Namespace: Parsed arguments including 'lr', 'beta', and 'beta_lr'.
    """
    parser = argparse.ArgumentParser(description="Parse hyperparameters for training.")
    parser.add_argument('--lr', type=float, required=True, help="Learning rate")
    parser.add_argument('--batch_size', type=int, required=False,default=64, help="Learning rate")
    parser.add_argument('--optimizer_name', type=str, required=True, help="optimizer name")
    parser.add_argument('--beta', type=float, required=False, default=0.995, help="Beta parameter")
    parser.add_argument('--beta_lr', type=float, required=False, default=1e-4, help="Learning rate for beta")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """ 
    Self-defined MNIST test, support OSGM and OSMM, 
    as well as txt output for easy plot
    """
    optimizers = [
                # torch.optim.Adam,
                # torch.optim.SGD, # 0.01
                # optim.DiffGrad,
                # optim.OSGM,
                optim.OSMM,
                # optim.Adafactor,
                # optim.Adahessian,
                ]
    
    args = parse_args()

    lr = args.lr
    batch_size = args.batch_size
    optimizer_name = args.optimizer_name
    beta = args.beta
    beta_lr = args.beta_lr
    epochs = 10

    if optimizer_name == "NAG" or optimizer_name == "SGD":
        optimizers = [torch.optim.SGD]
    elif optimizer_name == "Adam":
        optimizers = [torch.optim.Adam]
    elif optimizer_name == "OSGM":
        optimizers = [optim.OSGM]
    elif optimizer_name == "OSMM":
        optimizers = [optim.OSMM]

    execute_experiments(optimizers,lr,batch_size,epochs,beta,beta_lr)
