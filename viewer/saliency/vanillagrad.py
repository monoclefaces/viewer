import torch
import torch.nn as nn
from .base import Viewer
from ..manager import PostProcessor, one_hot

class VanillaGrad(Viewer):
    """Vanilla Gradient Attribution Method"""
    def __init__(self, model, **kwargs):
        r"""
        Reference Paper: https://arxiv.org/abs/1312.6034

        Args:
            model: model to view saliency maps
            (option) collapse_mode: whether to collapse the channel dimension of saliency maps to 1, option arguments:
                * [default]: 2
                * 0: no collapse
                * 1: mean
                * 2: max 
                * 3: gray-scale (have to rescale to 0~255)
        """
        super().__init__(model)
        # PostProcessor arugments
        self.collapse_mode = 2 if kwargs.get("collapse_mode") is None else kwargs.get("collapse_mode")

    def view(self, x:torch.FloatTensor, t:torch.LongTensor, **kwargs):
        r"""
        Generate saliency maps
        Args:
            x: input datas, size of (B, C, H, W)
            t: target datas, size of (B,)
        
        Return:
            Saliency map, size of (B, C, H, W)
        """
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        o = self.model(x)
        grad = one_hot(outputs=o, targets=t)
        o.backward(grad)
        saliency = x.grad.data.clone().detach()
        x.requires_grad_(requires_grad=False)

        saliency = PostProcessor.process(saliency, self.collapse_mode)
        return saliency