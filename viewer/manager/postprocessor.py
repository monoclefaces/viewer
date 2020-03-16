import torch

class PostProcessor(object):
    r"""
    Post Processor
    1. (option) Collaspe channel
        Collaspe the channel of scores giving `collapse_mode`
        * 0: no collapse
        * 1: mean
        * 2: max
        * 3: gray-scale (have to rescale to 0~255)
    2. rescale: min-max (0~1), multiply 255, change to “byte” type
    """

    @staticmethod
    def collaspe_channel(tensor, collapse_mode:int=0):
        r"""
        Collaspe color dimension if `tensor` has more than 1 in channel dimension.
        The `2: gray-scale` option is set to "ITU-R BT.709" according to reference 
        https://en.wikipedia.org/wiki/Grayscale
        Args:
            tensor: `torch.FloatTensor` type, must have size of $(B, C, *)$
        """
        method_dict = {
            1: [1/3, 1/3, 1/3],            # mean
            3: [0.2126, 0.7152, 0.0722],   # gray-scale (have to rescale to 0~255)
        }
        if collapse_mode == 0:
            return tensor
        elif collapse_mode == 2:
            return torch.max(tensor, dim=1, keepdim=True)[1]
        else:
            def weighted_sum(x, w):
                """
                return (B, C, *) tensor with weighted sum
                """
                view_size = ((x.size(0), len(w),) + (1,)*len(x.size()[2:]))
                w = torch.FloatTensor(w).unsqueeze(0).repeat(x.size(0), 1).view(view_size)
                return (x * w).sum(1, keepdim=True)
            
            w = method_dict[collapse_mode]
            return weighted_sum(tensor, w)
    
    @staticmethod
    def rescale(tensor):
        r"""
        Rescale the saliency maps and convert the pixels into 0~255 range.
        Default rescaling methods: Min-Max = $\dfrac{X - X_{min}}{X_{max} - X_{min}}$ 
        Args:
            tensor: tensor to rescale
        Return:
            Byte Tensor, size of (B, C, H, W)
        """
        B, C, H, W = tensor.size()
        tensor = tensor.view(B, -1)

        t_min = tensor.min(dim=-1, keepdim=True)[0]
        t_max = tensor.max(dim=-1, keepdim=True)[0]
        tensor = tensor.sub(t_min).div(t_max - t_min + 1e-10)
        return (tensor.view(B, C, H, W) * 255).byte()

    @classmethod
    def process(cls, scores, collapse_mode:int=0):
        r"""
        Run Post Processor
        1. if collapse_mode exist, collapse the channel dimension
        2. rescale and turn into byte type
        """
        if collapse_mode != 0:
            scores = cls.collaspe_channel(tensor=scores, collapse_mode=collapse_mode)
        scores = cls.rescale(scores)
        return scores