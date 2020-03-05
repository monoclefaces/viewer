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
        """
        self.reset_maps()
        # evaluation mode
        self.model = deepcopy(model)
        self.model.cpu()
        self.model.eval()

        # define hook layer (forward or backword)
        self.hook_layers()
        
    def view(self, datas, targets, **kwargs) -> Viewer:
        r"""
        Generate saliency maps
        """
        
        raise NotImplementedError

    def hook_layer(self) -> Viewer:
        r"""
        Define Hook layer that you've interested in(either forward or backward) 
        Have to define `hook_function` to pass
        """
        
        raise NotImplementedError

    def save(self, manager, savedir=None, **kwargs) -> Viewer:
        r"""
        Save saliency maps
        """

        raise NotImplementedError

    def rescale(self, tensor, mode=1) -> Viewer:
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

    def one_hot(self, outputs, targets) -> Viewer:
        r"""
        Create one-hot vector
        Args:
            outputs: final outputs of network, size=(B, C)
            targets: target vectors, size=(B,)

            * B: batch size
            * C: class size
        """
        onehot = torch.zeros_like(outputs.detach()).scatter(1, targets.unsqueeze(1), 1.0)
        return onehot

    def reset_maps(self) -> Viewer:
        r"""
        Reset `self.maps`
        """
        self.maps = OrderedDict()