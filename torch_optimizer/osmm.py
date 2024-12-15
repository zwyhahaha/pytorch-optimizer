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
        beta_lr: OptFloat = 1.0, # lr(beta)/lr(P)
        beta = 0.,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        stop_step: OptFloat = None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr,eps=eps,beta=beta,
                        weight_decay=weight_decay,stop_step=stop_step,
                        beta_lr=beta_lr)
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
                    state["beta_avg"] = torch.tensor(group["beta"])
                    state["Gm"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["Q"] = torch.zeros_like(p)
                    state["Q_avg"] = torch.zeros_like(p)
                    state["G"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(grad)
                else:
                    state["step"] += 1

                    prev_grad = state["prev_grad"]
                    m = state["m"]
                    eps = group["eps"]
                    lr = group["lr"]
                    beta_lr = group["beta_lr"]
                    stop_step = group["stop_step"]
                    step = state["step"]
                    if stop_step is None:
                        stop_step = np.inf
                    
                    if step % stop_step == 0: # restart
                        state["Q"] = torch.zeros_like(p)
                        group["beta"] = torch.tensor(0)
                        state["Q_avg"] = torch.zeros_like(p)
                        state["beta_avg"] = torch.tensor(0)
                        state["Gm"] = 0
                        state["G"] = torch.zeros_like(p)
                    else:
                        gr = - prev_grad.mul(grad) / (prev_grad.norm() ** 2 + 1e-20) # gradient of preconditioner
                        state["G"].addcmul_(gr, gr, value=1) # Adagrad normalizer
                        state["Q"].addcdiv_(gr, state["G"].add(eps).sqrt(), value=-lr) # adagrad preconditioner update
                        state["Q_avg"] = state["Q_avg"]*(step-1)/step + state["Q"]/step
                        
                        gm = (grad * m).sum() / (prev_grad.norm() ** 2 + 1e-20) # gradient of momentum coef
                        state["Gm"] += gm ** 2 # Adagrad normalizer for momentum coef
                        group["beta"] = group["beta"] - beta_lr*lr * gm / (state["Gm"].add(eps).sqrt()) # adagrad preconditioner update
                        state["beta_avg"] = state["beta_avg"]*(step-1)/step + group["beta"]/step

                    pcopy = p.data.clone()
                    p.addcmul_(state["Q"], grad, value=-1).add_(group["beta"] * m)

                    loss_new = closure()

                    if loss_new > 2 * loss:
                        p.data = pcopy

                    state["m"] = p - pcopy

                    del pcopy

                state["prev_grad"] = grad.clone()

        return loss