import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("OSGM",)

class OSGM(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: OptFloat = None,
        eps: float = 1e-08,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr,eps=eps)
        super(OSGM,self).__init__(params, defaults)

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
                grad = p.grad.data
                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["Q"] = torch.zeros_like(p)
                    state["G"] = torch.zeros_like(p)
                    state["prev_grad"] = torch.zeros_like(grad)
                else:
                    
                    state["step"] += 1
                    prev_grad = state["prev_grad"]
                    eps = group["eps"]
                    lr = group["lr"]
                    
                    gr = - prev_grad.mul(grad) / (prev_grad.norm() ** 2 + 1e-20) # gradient of preconditioner
                    
                    state["G"].addcmul_(gr, gr, value=1) # Adagrad normalizer
                    state["Q"].addcdiv_(gr, state["G"].add(eps).sqrt(), value=-lr) # adagrad preconditioner update
                    
                    pcopy = p.detach().clone()
                    p.addcmul_(state["Q"], grad, value=-1.0)

                    loss_new = closure()

                    if loss_new > loss:
                        p.data = pcopy

                    del pcopy

                state["prev_grad"] = grad.clone()

        return loss