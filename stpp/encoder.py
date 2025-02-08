import torch
import torch.nn as nn
from einops import repeat


Module = torch.nn.Module
Tensor = torch.Tensor


class Encoder(Module):
    """Encoder mapping context sequences to parameters of the posterior q(z1).

    Attributes:
        d_x (int): Dimensionality of the spatial coodinates.
        d_y (int): Dimensionality of the observations.
        d_z (int): Dimensionality of the latent state.
        d_model (int): Latent dimension for each transformer block.
        n_attn_heads (int): Number of attention heads for each transformer block.
        n_tf_layers (int): Number of transformer blocks in the stack.
    """

    def __init__(
        self,
        d_x: int, d_y: int, d_z: int,
        d_model: int, n_attn_heads: int, n_tf_layers: int, dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.time_proj = nn.Sequential(nn.Linear(1, d_model, bias=False))
        self.space_proj = nn.Sequential(nn.Linear(d_x, d_model, bias=False))
        self.obs_proj = nn.Sequential(nn.Linear(d_y, d_model, bias=False))

        self.transformer_stack = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_attn_heads,
                    dim_feedforward=2*d_model,
                    batch_first=True,
                    dropout=dropout_prob,
                ) for _ in range(n_tf_layers)
            ]
        )

        self.gamma_proj = nn.Sequential(nn.Linear(d_model, d_z))
        self.tau_proj = nn.Sequential(nn.Linear(d_model, d_z))

        self.agg_token = nn.Parameter(torch.randn((1, 1, d_model)))

        self.coord_dim = 1 + d_x

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Maps context sequences `x` to parameters of the posterior q(z1).

        Args:
            x: Context sequences with shape (batch, context len., 1+d_x+d_y).
            mask: Mask for padding tokens, has extra column for agg_token.

        Returns:
            Mean and variance of the posterior q(z1). Each has shape (batch, d_z).
        """
        t_emb = self.time_proj(x[:, :, [0]])
        coords_emb = self.space_proj(x[:, :, 1:self.coord_dim])
        obs_emb = self.obs_proj(x[:, :, self.coord_dim:])

        x = torch.cat(
            [
                t_emb + coords_emb + obs_emb,
                repeat(self.agg_token, "() () d -> b () d", b=x.shape[0]),
            ],
            dim=1,
        )

        for layer in self.transformer_stack:
            x = layer(x, src_key_padding_mask=mask)
        x = x[:, -1, :]

        return self.gamma_proj(x), torch.exp(self.tau_proj(x)-7)
