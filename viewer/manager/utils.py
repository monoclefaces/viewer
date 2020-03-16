import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def one_hot(outputs, targets) -> None:
    r"""
    Create one-hot vector
    Args:
        outputs: final outputs of network, size=(B, C)
        targets: target vectors, size=(B,)

        * B: batch size
        * C: class size

    # maybe need numpy version
    """
    onehot = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1.0)
    return onehot

class ImageProcessor(object):
    """ImageProcessor"""
    @staticmethod
    def save_image(img, path):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if isinstance(img, np.ndarray):
            if img.ndim >= 4:
                img = np.squeeze(img)
            elif img.ndim == 3:
                size = img.shape
                if size[-1] != 3:
                    img = img.reshape(*size[1:], size[0])
            img = Image.fromarray(img)
            img.save(path)
        if isinstance(img, plt.Figure):
            img.savefig(path)

    @staticmethod
    def img_process(pil_img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[ 0.485, 0.456, 0.406 ],
                                std=[ 0.229, 0.224, 0.225 ]),
        ])
        return transform(pil_img).unsqueeze(0)

    @staticmethod
    def img_inv_process(tensor):
        """process for single image"""
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        inv_transform = transforms.Compose([
            transforms.Normalize(
                mean=[ -0.485/0.229, -0.456/0.224, -0.406/0.225 ],
                std=[ 1/0.229, 1/0.224, 1/0.225 ]
            ), 
            transforms.ToPILImage()
        ])
        return inv_transform(tensor)

    @staticmethod
    def open_img(img_path, resize=None, crop=None):
        img = Image.open(img_path)
        if resize is not None:
            assert isinstance(resize, int), "insert int number for resize"
            tf = [transforms.Resize(resize)] 
            if crop is not None:
                tf += [transforms.CenterCrop(crop)]
            transform_resize = transforms.Compose(tf)
            img = transform_resize(img)
        return img
    
    @classmethod
    def init_img(cls, img_path, zero=False, resize=None, crop=False, device="cpu"):
        img = cls.open_img(img_path, resize, crop)
        img_tensor = cls.img_process(img).to(device)
        if zero:
            img_tensor = torch.zeros_like(img_tensor)
        return img_tensor