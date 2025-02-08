import torch


Module = torch.nn.Module
Tensor = torch.Tensor


class IntensityCorrection(Module):
    def __init__(self, val: float = 0) -> None:
        super().__init__()
        self.val = val

    def forward(self, x: Tensor) -> Tensor:
        # return torch.pow(x, 2) + self.val
        return torch.exp(x) + self.val
