import torch
import torch.nn as nn


class PostProcessor(object):
    def __init__(self, no_abs, collapse_mode):
        r"""
        Post Processor
        1. absolute scores: 
            Absolute the attribution scores, you can choose not to do it by
            giving `no_abs_grad` to `True`
        2. (option) Collaspe channel
            Collaspe the channel of scores giving `collapse_mode`
            * 0: no collapse
            * 1: mean
            * 2: gray-scale
        3. rescale: min-max (0~1), multiply 255, change to “byte” type
        """
        self.no_abs = no_abs
        self.collapse_mode = collapse_mode
        self.method_dict = {
            1: [1/3, 1/3, 1/3],            # mean
            2: [0.2126, 0.7152, 0.0722],   # gray-scale
        }

    def absolute_scores(self, tensor) -> PostProcessor:
        r"""
        Absolute the attribution scores
        """
        if self.no_abs:
            return tensor
        else:
            return torch.abs(tensor)

    def collaspe_channel(self, tensor, collapse_mode=0:int) -> PostProcessor:
        r"""
        Collaspe color dimension if `tensor` has more than 1 in channel dimension.
        The `2: gray-scale` option is set to "ITU-R BT.709" according to reference 
        https://en.wikipedia.org/wiki/Grayscale
        Args:
            tensor: `torch.FloatTensor` type, must have size of $(B, C, *)$
        """
        if collapse_mode == 0:
            return tensor
        else:
            def weighted_sum(x, w):
                """
                return (B, C, *) tensor with weighted sum
                """
                view_size = ((x.size(0), len(w),) + (1,)*len(x.size()[2:]))
                w = torch.FloatTensor(w).unsqueeze(0).repeat(x.size(0), 1).view(view_size)
                return (x * w).sum(1, keepdim=True)
            
            w = self.method_dict[collapse_mode]
            tensor = weighted_sum(tensor, w)
            return tensor
            
    def rescale(self, tensor) -> PostProcessor:
        r"""
        Rescale the saliency maps and convert the pixels into 0~255 range.
        Default rescaling methods: Min-Max = $\dfrac{X - X_{min}}{X_{max} - X_{min}}$ 
        Args:
            tensor: tensor to rescale
        Return:
            Byte Tensor, size of (B, C, H, W)
        """

        if mode == 0:
            return tensor
        B, C, H, W = tensor.size()
        tensor = tensor.view(B, -1)

        t_min = tensor.min(dim=-1, keepdim=True)[0]
        t_max = tensor.max(dim=-1, keepdim=True)[0]
        tensor = tensor.sub(t_min).div(t_max - t_min + 1e-10)
        return (tensor.view(B, C, H, W) * 255).byte()

    def process(self, scores):
        """
        Run Post Processor
        1. absolute scores
        2. if collapse_mode exist, collapse the channel dimension
        3. rescale and turn into byte type
        """
        scores = self.absolute_scores(tensor=scores)
        if self.collapse_mode != 0:
            scores = self.collapse_mode(tensor=scores, collapse_mode=self.collapse_mode)
        scores = self.rescale(scores)
        return scores