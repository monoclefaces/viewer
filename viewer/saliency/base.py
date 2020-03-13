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

    def reset_maps(self) -> Viewer:
        r"""
        Reset `self.maps`
        """
        self.maps = OrderedDict()