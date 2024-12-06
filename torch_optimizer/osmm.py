import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ("OSMM",)

class OSMM(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: OptFloat = None,
        beta = 0.995,
        beta_lr = 1e-4,
        eps: float = 1e-15,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr,eps=eps,beta=beta,beta_lr=beta_lr)
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
                grad = p.grad.data
                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["beta"] = group["beta"]
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
                    
                    gr = -prev_grad.mul(grad)/(prev_grad.norm()**2+eps) # gradient of preconditioner
                    
                    state["G"].addcmul_(gr,gr,value=1) # Adagrad normalizer
                    state["Q"].addcdiv_(gr,state["G"].sqrt().add(eps),value=-lr) # adagrad preconditioner update

                    gm = prev_grad * m/(prev_grad.norm()**2+eps) # gradient of momentum coef

                    state["Gm"] += gm**2 # Adagrad normalizer for momentum coef
                    state["beta"] += -group["beta_lr"] * gm / (state["Gm"].sqrt().add(eps)) # adagrad preconditioner update
                    
                    pcopy = p.detach().clone()
                    p.addcmul_(state["Q"], grad, value = -1).add_(state["beta"]*m)

                    loss_new = closure()

                    if loss_new > loss:
                        p.data = pcopy

                    state["m"] = p - pcopy

                    del pcopy

                state["prev_grad"] = grad.clone()

        return loss