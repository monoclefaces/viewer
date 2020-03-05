import torch
import torch.nn as nn
from .base import Viewer

class VanillaGrad(Viewer):
    """Vanilla Gradient Attribution Method"""
    def __init__(self, model, **kwargs):
        r"""
        Reference Paper: https://arxiv.org/abs/1312.6034

        Args:
            model: 
            rescale_mode: `Viewer.rescale` argument
            no_abs_grad: (option) not absolute the gradient of features
        """
        super().__init__(model)
        self.rescale_mode = 1 if kwargs.get("rescale_mode") is None else kwargs.get("rescale_mode")
        self.no_abs_grad = False if kwargs.get("no_abs_grad") is None else kwargs.get("no_abs_grad")

    def view(self, x, t):
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        o = self.model(x)
        grad = self.one_hot(outputs=o, targets=t)
        o.backward(grad)
        x_grad = x.grad.data.clone().detach()
        x.requires_grad_(requires_grad=False)
        x_grad = x_grad if self.no_abs_grad else torch.abs(x_grad)
        x_grad = self.rescale(x_grad, mode=self.rescale_mode)
        return x_grad  # (B, C, H, W)

    