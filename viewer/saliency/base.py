import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict


class Viewer(nn.Module):
    def __init__(self, model, **kwargs):
        super(Viewer, self).__init__()
        r"""
        Basic Class of Attribution Models

        Args: 
            model: a neural network implemented using PyTorch
            seq_name: (optional) the variable name which is pointed to convolutional layer in your network 
        """
        self.reset_maps()
        # convolution layer
        self.seq_name = "features" if "seq_name" not in kwargs.keys() else kwargs.get("seq_name")
        # evaluation mode
        self.model = deepcopy(model)
        self.model.cpu()
        self.model.eval()

        # define hook layer (forward or backword)
        self.hook_layers()

    def view(self, datas, targets, **kwargs):
        r"""
        Generate saliency maps
        """
        
        raise NotImplementedError

    def hook_layer(self):
        r"""
        Define Hook layer that you've interested in(either forward or backward) 
        Have to define `hook_function` to pass
        """
        
        raise NotImplementedError

    def save(self, manager, savedir=None, **kwargs):
        r"""
        Save saliency maps
        """

        raise NotImplementedError

    def rescale(self, tensor, mode=1):
        r"""
        Rescale the saliency maps and convert the pixels into 0~255 range.
        Default rescaling option is 'Min-Max'.
        Args:
            tensor: tensor to rescale
            mode:
                0: No Rescaling
                1: Min-Max $\dfrac{X - X_{min}}{X_{max} - X_{min}}$ 
                2: Mean-Std $\dfrac{X - X_{mean}}{X_{std}}$ 
        Return:
            Byte Tensor, size of (B, C, H, W)
        """

        if mode == 0:
            return tensor
        B, C, H, W = tensor.size()
        tensor = tensor.view(B, -1)

        if mode == 1:
            t_min = tensor.min(dim=-1, keepdim=True)[0]
            t_max = tensor.max(dim=-1, keepdim=True)[0]
            tensor = tensor.sub(t_min).div(t_max - t_min + 1e-10)
        elif mode == 2:
            t_mean = tensor.mean(dim=-1, keepdim=True)
            t_std = tensor.std(dim=-1, keepdim=True)
            tensor = tensor.sub(t_mean).div(t_std + 1e-10)

        return (tensor.view(B, C, H, W) * 255).byte()

    def reset_maps(self):
        r"""
        Reset `self.maps`
        """
        self.maps = OrderedDict()