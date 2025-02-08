import torch
import torch.nn as nn
from stpp.interp import interpolate


Module = torch.nn.Module
Tensor = torch.Tensor


class ContinuousDecoder(Module):
    """Maps latent state z(t) and spatial coordinate x to u(t, x).

    Attributes:
        d_z (int): Dimensionality of the latent state.
        d_x (int): Dimensionality of the spatial coodinates.
        d_u (int): Dimensionality of the latent spatiotemporal state.
        f (Module): Mapping from (z(t), x) to u(t, x).
    """
    def __init__(self, d_z: int, d_x: int, d_u: int, f: Module, interp_method: str) -> None:
        super().__init__()
        self.space_proj = nn.Linear(d_x, d_z, bias=False)
        self.f = f
        self.interp_method = interp_method

    def forward(self, t_eval: Tensor, x_eval: Tensor, t: Tensor, z: Tensor) -> Tensor:
        """Evaluates the latent spatiotemporal state u(t, x) for a single trajectory t, z.

        Args:
            t_eval: Evaluation time points, has shape (n, ).
            x_eval: Evaluation spatial locations, has shape (n, d_x).
            t: Trajectory time points, has shape (time, ).
            z: Trajectory values at time points `t`, has shape (time, d_z).

        Returns:
            Latent spatiotemporals state at (t_eval, x_eval). Has shape (n, d_u).
        """
        if t_eval.ndim != 1 or t.ndim != 1:
            raise ValueError("t and t_eval should be a 1-dimensional arrays.")
        if x_eval.ndim != 2:
            raise ValueError("x should be a 2-dimensional arrays.")
        if z.ndim != 2:
            raise ValueError("z should be a 2-dimensional arrays.")
        if t_eval.shape[0] != x_eval.shape[0]:
            raise ValueError("t_eval and x_eval must have matching first dimension.")
        if t.shape[0] != z.shape[0]:
            raise ValueError("t and z must have matching first dimension.")

        z_eval = interpolate(t_eval, t, z, method=self.interp_method)
        return self.f(z_eval + self.space_proj(x_eval))


    # def forward(self, z_t: Tensor, x: Tensor) -> Tensor:
    #     """Evaluates the latent spatiotemporal state u(t, x).

    #     Args:
    #         z_t: Latent states, has shape (batch, N_i, d_z).
    #         x_eval: Evaluation spatial locations, has shape (batch, N_i, d_x).

    #     Returns:
    #         Latent spatiotemporals states. Has shape (batch, N_i, d_z).
    #     """
    #     return self.f(z_t + self.space_proj(x))
