from torch import nn


class BackboneBase(nn.Module):
    def __init__(self):
        super().__init__()

    def get_named_param_lr_pairs(self, args, prefix: str):
        raise NotImplementedError
