import torch
import torch.nn as nn

class GuidedReLU(nn.ReLU):
    """GuidedReLU"""
    def __init__(self, inplace=False):
        r"""
        This layer will recognize as `nn.ReLU`
        """
        super(GuidedReLU, self).__init__(inplace)
        self.register_forward_hook(self.f_hook)
        self.register_backward_hook(self.b_hook)
        self.outputs = []
    
    def f_hook(self, m, i, o):
        self.outputs.append(o)

    def b_hook(self, m, i, o):
        r"""
        backward hook
        i: (input,) -> this is backward output
        o: (output,) -> this is backward input
        """
        deconv_grad = o[0].clamp(min=0)  # o: backward input
        forward_output = self.outputs.pop(-1)
        forward_mask = forward_output.ne(0.0).type_as(forward_output)
        grad_in = deconv_grad * forward_mask
        return (grad_in, )
