import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from hyperopt import fmin, hp, tpe

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch_optimizer as optim

# plt.style.use("seaborn-white")


def rosenbrock(tensor):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rastrigin(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    A = 10
    f = (
        A * 2
        + (x**2 - A * lib.cos(x * math.pi * 2))
        + (y**2 - A * lib.cos(y * math.pi * 2))
    )
    return f

def sphere(tensor):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    return 10*x**2 + y**2 - 2*x*y

def logistic(tensor, lib=torch):
    x, y = tensor
    return lib.log(1+lib.exp(-x))+lib.log(1+lib.exp(-y))

def execute_steps2(
    func, initial_state, optimizer_class, optimizer_config, num_iter=500
):
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = []
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    grads = np.zeros((2, num_iter + 1))
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        def closure():
            f = func(x)
            return f
        
        f = closure()
        if optimizer_class.__name__ == "Adahessian":
            f.backward(create_graph=True, retain_graph=True)
        else:
            f.backward()
        
        torch.nn.utils.clip_grad_norm_(x, 1.0)

        optimizer.step(closure)

        steps[:, i] = x.detach().numpy()
        grads[:, i] = x.grad.clone().detach().numpy()
    return steps,grads 

def execute_steps(
    func, initial_state, optimizer_class, optimizer_config, num_iter=500
):
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = []
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()

        def closure():
            f = func(x)
            return f
        
        f = closure()
        f.backward(create_graph=True, retain_graph=True)
        
        torch.nn.utils.clip_grad_norm_(x, 1.0)

        optimizer.step(closure)

        steps[:, i] = x.detach().numpy()
    return steps

def objective_rastrigin(params):
    lr = params["lr"]
    optimizer_class = params["optimizer_class"]
    initial_state = (-2.0, 3.5)
    minimum = (0, 0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    steps = execute_steps(
        rastrigin, initial_state, optimizer_class, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def objective_rosenbrok(params):
    lr = params["lr"]
    optimizer_class = params["optimizer_class"]
    minimum = (1.0, 1.0)
    initial_state = (-2.0, 2.0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    steps = execute_steps(
        rosenbrock, initial_state, optimizer_class, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def objective_sphere(params):
    lr = params["lr"]
    optimizer_class = params["optimizer_class"]
    minimum = (0.0, 0.0)
    initial_state = (-2.0, 2.0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    steps = execute_steps(
        sphere, initial_state, optimizer_class, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2

def objective_logistic(params):
    lr = params["lr"]
    optimizer_class = params["optimizer_class"]
    minimum = (0.0, 0.0)
    initial_state = (-2.0, 2.0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    steps = execute_steps(
        logistic, initial_state, optimizer_class, optimizer_config, num_iter
    )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2

def plot_rastrigin(grad_iter, optimizer_name, lr):
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)

    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y], lib=np)

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 20, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")
    ax.set_title(
        "Rastrigin func: {} with "
        "{} iterations, lr={:.6}".format(optimizer_name, len(iter_x), lr)
    )
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("docs/rastrigin_{}.png".format(optimizer_name))

    loss = rastrigin(grad_iter, lib=np)
    np.save("cvg_plot/rastrigin/{}.npy".format(optimizer_name),loss)


def plot_rosenbrok(grad_iter, optimizer_name, lr):
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")

    ax.set_title(
        "Rosenbrock func: {} with {} "
        "iterations, lr={:.6}".format(optimizer_name, len(iter_x), lr)
    )
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("docs/rosenbrock_{}.png".format(optimizer_name))

    loss = rosenbrock(grad_iter)
    np.save("cvg_plot/rosenbrock/{}.npy".format(optimizer_name),loss)

def plot_sphere(grad_iter, optimizer_name, lr):
    x = np.linspace(-3, 3, 250)
    y = np.linspace(-3, 3, 250)
    minimum = (0.0, 0.0)

    X, Y = np.meshgrid(x, y)
    Z = sphere([X, Y])

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")

    ax.set_title(
        "Sphere func: {} with {} "
        "iterations, lr={:.6}".format(optimizer_name, len(iter_x), lr)
    )
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("docs/sphere_{}.png".format(optimizer_name))

    loss = sphere(grad_iter)
    np.save("cvg_plot/sphere/{}.npy".format(optimizer_name),loss)

def plot_logistic(grad_iter, optimizer_name, lr):
    x = np.linspace(-3, 3, 250)
    y = np.linspace(-3, 3, 250)
    minimum = (0.0, 0.0)

    X, Y = np.meshgrid(x, y)
    Z = logistic([X, Y],lib=np)

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")

    ax.set_title(
        "logistic func: {} with {} "
        "iterations, lr={:.6}".format(optimizer_name, len(iter_x), lr)
    )
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("docs/logistic_{}.png".format(optimizer_name))

    loss = logistic(grad_iter,lib=np)
    np.save("cvg_plot/logistic/{}.npy".format(optimizer_name),loss)

def plot_grads(grads,func_name,optimizer_name,lr):
    plt.figure(figsize=(8, 8))
    plt.plot(range(1,grads.shape[1]+1), grads[0,:],label='grad_x')
    plt.plot(range(1,grads.shape[1]+1), grads[1,:],label='grad_y')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient')
    plt.title("{} func: lr {} with {} iterations".format(func_name, lr,len(grads)))
    plt.legend()
    plt.grid(True)
    plt.savefig("grad_plot/{}_{}.png".format(func_name,optimizer_name))

def execute_experiments(
    optimizers, objective, func, plot_func, initial_state, seed=1
):
    seed = seed
    for item in optimizers:
        optimizer_class, lr_low, lr_hi = item
        space = {
            "optimizer_class": hp.choice("optimizer_class", [optimizer_class]),
            "lr": hp.loguniform("lr", lr_low, lr_hi),
        }
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            rstate=np.random.default_rng(seed),
        )
        print(best["lr"], optimizer_class)

        steps = execute_steps(
            func,
            initial_state,
            optimizer_class,
            {"lr": best["lr"]},
            num_iter=500,
        )
        plot_func(steps, optimizer_class.__name__, best["lr"])
        steps,grads = execute_steps2(
            func,
            initial_state,
            optimizer_class,
            {"lr": best["lr"]},
            num_iter=500,
        )


def LookaheadYogi(*a, **kw):
    base = optim.Yogi(*a, **kw)
    return optim.Lookahead(base)


if __name__ == "__main__":
    # python examples/viz_optimizers.py

    # Each optimizer has tweaked search space to produce better plots and
    # help to converge on better lr faster.
    optimizers = [
        # baselines
        (torch.optim.Adam, -8, 0.5),
        (torch.optim.SGD, -8, -1.0),
        # Adam based
        (optim.AdaBound, -8, 0.3),
        (optim.Adahessian, -1, 8),
        (optim.AdaMod, -8, 0.2),
        (optim.AdamP, -8, 0.2),
        (optim.DiffGrad, -8, 0.4),
        (optim.Lamb, -8, -2.9),
        (optim.MADGRAD, -8, 0.5),
        (optim.NovoGrad, -8, -1.7),
        (optim.RAdam, -8, 0.5),
        (optim.Yogi, -8, 0.1),
        # SGD/Momentum based
        (optim.AccSGD, -8, -1.4),
        (optim.SGDW, -8, -1.5),
        (optim.SGDP, -8, -1.5),
        (optim.PID, -8, -1.0),
        (optim.QHM, -6, -0.2),
        (optim.QHAdam, -8, 0.1),
        (optim.Ranger, -8, 0.1),
        (optim.RangerQH, -8, 0.1),
        (optim.RangerVA, -8, 0.1),
        (optim.Shampoo, -8, 0.1),
        (LookaheadYogi, -8, 0.1),
        (optim.AggMo, -8, -1.5),
        (optim.SWATS, -8, -1.5),
        (optim.Adafactor, -8, 0.5),
        (optim.A2GradUni, -8, 0.1),
        (optim.A2GradInc, -8, 0.1),
        (optim.A2GradExp, -8, 0.1),
        (optim.AdaBelief, -8, 0.1),
        (optim.Apollo, -8, 0.1),
    ]
    optimizers = [
                # (torch.optim.Adam, -8, 0.5),
                # (optim.Adafactor, -8, 0.5),
                # (torch.optim.SGD, -8, -1.0),
                # (optim.Adahessian, -1, 8),
                (optim.OSMM, -8, 1),
                # (optim.OSGM, -8, 1),
                  ]
    
    execute_experiments(
        optimizers,
        objective_rastrigin,
        rastrigin,
        plot_rastrigin,
        (-2.0, 3.5),
    )

    execute_experiments(
        optimizers,
        objective_rosenbrok,
        rosenbrock,
        plot_rosenbrok,
        (-2.0, 2.0),
    )

    execute_experiments(
        optimizers,
        objective_sphere,
        sphere,
        plot_sphere,
        (-2.0, 2.0),
    )

    execute_experiments(
        optimizers,
        objective_logistic,
        logistic,
        plot_logistic,
        (-2.0, 2.0),
    )