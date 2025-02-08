import torch

Module = torch.nn.Module
Tensor = torch.Tensor


class DynamicsFunction(Module):
    def __init__(self, f: Module) -> None:
        super().__init__()
        self.f = f

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        return self.f(z)
