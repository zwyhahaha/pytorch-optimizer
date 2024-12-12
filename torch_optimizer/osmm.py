import torch
from torch.optim.optimizer import Optimizer

import numpy as np

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("OSMM",)

class OSMM(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: OptFloat = None,
        beta = 0.995,
        random_scaling = False,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        stop_step: OptFloat = None,
        stop_beta: float = 0.9,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr,eps=eps,beta=beta,random_scaling=random_scaling
                        ,weight_decay=weight_decay,stop_step=stop_step,stop_beta=stop_beta)
        super(OSMM,self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                weight_decay = group["weight_decay"]
                if weight_decay != 0:
                    grad = p.grad.data.add(p.data, alpha=weight_decay)
                else:
                    grad = p.grad.data

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    # state["beta"] = torch.tensor(group["beta"])
                    state["Gm"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["Q"] = torch.zeros_like(p)
                    state["G"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(grad)
                else:
                    state["step"] += 1

                    prev_grad = state["prev_grad"]
                    m = state["m"]
                    eps = group["eps"]
                    lr = group["lr"]
                    stop_step = group["stop_step"]
                    stop_beta = group["stop_beta"]
                    
                    gr = - prev_grad.mul(grad) / (prev_grad.norm() ** 2 + 1e-20) # gradient of preconditioner
                    
                    state["G"].addcmul_(gr, gr, value=1) # Adagrad normalizer
                    state["Q"].addcdiv_(gr, state["G"].add(eps).sqrt(), value=-lr) # adagrad preconditioner update
                    gm = (prev_grad * m).sum() / (prev_grad.norm() ** 2 + 1e-20) # gradient of momentum coef

                    if stop_step is not None and state["step"] <= stop_step:
                        state["Gm"] += gm ** 2 # Adagrad normalizer for momentum coef
                        group["beta"] += - lr * gm / (state["Gm"].add(eps).sqrt()) # adagrad preconditioner update
                        # group["beta"].clamp_(0.9, 0.9995) # beta clipping
                    else:
                        group["beta"] = torch.tensor(stop_beta)

                    pcopy = p.detach().clone()
                    p.addcmul_(state["Q"], grad, value=-1).add_(group["beta"] * m)

                    loss_new = closure()

                    if loss_new > loss:
                        p.data = pcopy

                    state["m"] = p - pcopy

                    del pcopy

                state["prev_grad"] = grad.clone()

        return loss