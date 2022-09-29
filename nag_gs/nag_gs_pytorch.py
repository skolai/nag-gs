import torch as T

from typing import List, Optional

__all__ = ('NAG4', 'nag4')


class NAG4(T.optim.Optimizer):
    """Class NAG4 implements algorithm without update on gamma which stands for
    the semi-implicit integration of the Nesterov Accelerated Gradient (NAG)
    flow.

    Arguments
    ---------
        params (collection): Collection of parameters to optimize.
        lr (float, optional): Learning rate (or alpha).
        mu (float, optional): Momentum mu.
        gamm (float, optional): Gamma factor.
    """

    def __init__(self, params, lr=1e-2, mu=1.0, gamma=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid alpha: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid mu: {mu}")

        defaults = dict(lr=lr, mu=mu, gamma=gamma)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @T.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments
        ---------
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with T.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            v_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    # Initialization
                    if len(state) == 0:
                        state['v'] = T.clone(p).detach()

                    v_list.append(state['v'])

            nag4(params_with_grad,
                 grads,
                 v_list,
                 alpha=group['lr'],
                 mu=group['mu'],
                 gamma=group['gamma'])

        return loss


def nag4(params: List[T.Tensor], grads: List[T.Tensor],
         v_list: List[Optional[T.Tensor]], alpha: float, mu: float,
         gamma: float):
    """Function nag4 performs NAG4 algorithm computation.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        v = v_list[i]
        # Update v
        v.add_(mu*param - grad, alpha=alpha/gamma)
        v.div_(1 + alpha*mu/gamma)

        # Update parameters
        param.add_(v, alpha=alpha)
        param.div_(1+alpha)
