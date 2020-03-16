import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .base import Viewer
from .modules import GuidedReLU
from ..manager import PostProcessor, one_hot

class GradCAM(Viewer):
    """GradCAM"""
    def __init__(self, model, module_name:str, key:int=-1, use_guided:bool=False, **kwargs):
        r"""
        Reference Paper: https://arxiv.org/abs/1610.02391

        Args:
            model: model to view saliency maps
            module_name: the module or container name, it will find the layer from 
                `model._modules.get(module_name)`. 
                * Only support `nn.Sequential` for now.
            key: which activation layer to calculate gradient, it will find reflected 
                convolution layer(pipeline: `nn.Conv2d` > ... > `self.activation`) from the module that 
                searched by `module_name`, and assign to `key`-th reflected convolution layer.
                * [default]: -1
                * Only support `nn.ReLU` activation function for now
            use_guided: whether to use combining guided backpropagation algorithim
            (option) collapse_mode: whether to collapse the channel dimension of saliency maps to 1, option arguments:
                * [default]: 0
                * 0: no collapse
                * 1: mean
                * 2: max 
                * 3: gray-scale (have to rescale to 0~255)
            
        """
        super(GradCAM, self).__init__(model)
        self.module_name = module_name
        self.key = key
        self.use_guided = use_guided
        self.activation = nn.ReLU
        self.collapse_mode = 0 if kwargs.get("collapse_mode") is None else kwargs.get("collapse_mode")
        # Note: only support `nn.ReLU` activation function for now
        # self.activation = nn.ReLU if kwargs.get("act") self._get_all_activation().get(kwargs.get("act"))
        if self.use_guided:
            inplace = False if kwargs.get("inplace") is None else True
            self._convert(self.model, convert_from=self.activation, convert_to=GuidedReLU, inplace=inplace)
        self.hook_layers(module_name, key)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # PostProcessor arugments
        self.no_abs = False if kwargs.get("no_abs") is None else kwargs.get("no_abs")
        
    def hook_layers(self, module_name, key):
        r"""
        Select the layer by module_name and hook the activation by the `key`

        Args:
            module_name: str type, the module or container name in the model
            key: int type

        # About using same layer in a Module

        Some models are using the same layer with same name(like the activation layer) many times 
        in a Module. If this happens, have to save all outputs & gradients from the layer,
        and then always choose the first one from candidates(`self._gradient`) 
        when backpropagation, and choose the last one from candidate(`self._conv_outputs`) 
        when forward propagation. See `_confirm_candidates` function.
        """
        self._reset_candidates()
        # select layer to hook
        module = self.model._modules.get(module_name)
        assert isinstance(module, nn.Sequential), "Only support `nn.Sequential` type module"
        self.layers_dict = OrderedDict()
        for name, layer in module.named_modules():
            if isinstance(layer, self.activation):
                self.layers_dict[name] = layer
        hook_layer = list(self.layers_dict.values())[key]
        
        # hook the gradient of the scores with respect to feature maps
        def forward_hook_fn(m, i, o):
            self._conv_outputs.append(o.clone())

        def backward_hook_fn(m, i, o):
            self._gradients.append(o[0].clone())

        hook_layer.register_forward_hook(forward_hook_fn)
        hook_layer.register_backward_hook(backward_hook_fn)
        return None
    
    def _reset_candidates(self):
        self._conv_outputs = []
        self._gradients = []

    def _confirm_candidates(self):
        r"""
        Confirm the output and gradient from activation maps(tensors)
        """
        self.conv_output = self._conv_outputs[-1]
        self.gradient = self._gradients[0]
        self._reset_candidates()

    def gradcam_process(self, H, W):
        r"""
        Calculate GradCAM 
        """
        self._confirm_candidates()
        alpha = self.global_avgpool(self.gradient)
        cam = torch.relu((alpha * self.conv_output).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        return cam   # (B, 1, H, W) 0~255 byte

    def view(self, x, t, **kwargs):
        r"""
        Generate saliency maps
        Args:
            x: input datas, size of (B, C, H, W)
            t: target datas, size of (B,)
            return_seperate: (optional) return saliency map and guided backpropagation seperately
        Return:
            if `return_preprocess`: 
                return `cam`, `guided` before PostProcess
            elif `return_all`:
                return `saliency`, `cam` after PostProcess, if `self.use_guided` is False, two tensor
                will be the same tensor.
            else:
                return `salienc`, when using guided backpropagation size will be (B, C, H, W), 
                otherwise size will be (B, 1, H, W) 
        """
        H, W = x.size()[2:]
        x.requires_grad_(requires_grad=True)
        self.model.zero_grad()
        o = self.model(x)
        grad = one_hot(outputs=o, targets=t)
        o.backward(grad)
        guided = x.grad.clone() if self.use_guided else 1.
        x.requires_grad_(requires_grad=False)

        cam = self.gradcam_process(H, W)
        if kwargs.get("return_preprocess") == True:
            return cam.detach(), guided.detach()
        saliency = PostProcessor.process(cam*guided, self.collapse_mode)
        if kwargs.get("return_all") == True:
            return saliency.detach(), cam.detach()
        return saliency.detach()
