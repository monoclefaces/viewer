import torch
import torch.nn as nn
from .base import Viewer

class VanillaGrad(Viewer):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.rescale_mode = 1 if kwargs.get("rescale_mode") is None else kwargs.get("rescale_mode")
        self.abs_grad = False if kwargs.get("abs_grad") is None else kwargs.get("abs_grad")

    def view(self, x, t):
        """vanilla gradient"""
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        o = self.model(x)
        grad = self._one_hot(t, module_name="fc")
        o.backward(grad)
        x_grad = x.grad.data.clone().detach()
        x.requires_grad_(requires_grad=False)
        if self.abs_grad:
            x_grad = torch.abs(x_grad)
        x_grad = self.rescale(x_grad, mode=self.rescale_mode)
        return x_grad

    