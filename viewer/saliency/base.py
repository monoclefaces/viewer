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

        Kwargs:
            conf_path: 
        """        
        self.reset_maps()
        # evaluation mode
        self.model = deepcopy(model)
        self.model.cpu()
        self.model.eval()

    def view(self, datas, targets, **kwargs):
        r"""
        Generate saliency maps
        Args:
            x: input datas, size of (B, C, H, W)
            t: target datas, size of (B,)
        
        Return:
            Saliency map, size of (B, C, H, W)
        """
        
        raise NotImplementedError

    def __call__(self, datas, targets, **kwargs):
        return self.view(datas, targets, **kwargs)

    def hook_layers(self):
        r"""
        Define Hook layer that you've interested in(either forward or backward) 
        Have to define `hook_function` to pass. 
        ```python
        def hook_function(m, i, o):
            # use input arguments whatever you want to do
            for x in [m, i, o]:
                print(type(x))
            print(m)
            print(i)
            print(o)
            
        layer = nn.Linear(5, 3)
        layer.register_forward_hook(hook_function)
        layer.register_backward_hook(hook_function)
        # forward
        output = layer(torch.randn(2, 5))
        # backward
        output.backward(torch.randn(2, 3))
        ```
        Arguments about hook function looks like following:
        [when running the hook_function at forward]
            - m: module class
            - i: forward input from previous layer
            - o: forward output to next layer
                 you can return same shape of tensor to pass the modified outputs
        [when running the hook_function at backward]
            - m: module class
            - i: gradient input to next layer (backward out), 
                 you can return same shape of tuple to pass the modified gradients
            - o: gradient output from previous layer (backward in)
        """
        
        raise NotImplementedError

    def save(self, manager, savedir=None, **kwargs):
        r"""
        Save saliency maps
        """

        raise NotImplementedError

    def reset_maps(self):
        r"""
        Reset `self.maps`
        """
        self.maps = OrderedDict()

    def _get_all_activation(self):
        r"""
        Can get almost all activation function in pytorch 
        """
        import inspect
        acts_list = [(n.lower(), m) for n, m in inspect.getmembers(nn.modules.activation) 
            if (isinstance(m, type)) and ("activation" in str(m))]
        return dict(acts_list)

    def _convert(self, model, convert_from, convert_to, **kwargs):
        r"""
        Convert all layer that satisfy the condition `convert_from`.
        keyword arguments are for `convert_to` function(or class)

        example:
        ```python
        class Example(Viewer):
            def __init__(self):
                super(Example, self).__init__(model)
                self._convert(model, convert_from=nn.ReLU, convert_to=nn.LeakyReLU, inplace=True)

        model = torchvision.models.resnet152(pretrained=False)
        ex = Example(model)
        # you can find all ReLU is converted to LeakyReLU
        ex
        ```
        """
        for child_name, child in model.named_children():
            if isinstance(child, convert_from):
                setattr(model, child_name, convert_to(**kwargs))
            else:
                self._convert(child, convert_from, convert_to, **kwargs)