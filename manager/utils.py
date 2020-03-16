import torch

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