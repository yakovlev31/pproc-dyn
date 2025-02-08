from types import SimpleNamespace

import numpy as np

import torch
from torch.distributions.normal import Normal

from torchdiffeq import odeint

import torchquad

from einops import pack, unpack

from stpp.data import Trajectory

from stpp.encoder import Encoder
from stpp.dynamics import DynamicsFunction
from stpp.decoder import ContinuousDecoder

from stpp.utils.utils import kl_norm_norm, pad_context


Tensor = torch.Tensor
Module = torch.nn.Module
Sequential = torch.nn.Sequential


class Model(Module):
    def __init__(
        self,
        enc: Encoder,
        dyn: DynamicsFunction,
        phi: ContinuousDecoder,
        lm: Module,
        dec: Module,
        space_range: list[list[float]],
    ) -> None:

        super().__init__()
        self.enc = enc
        self.dyn = dyn
        self.phi = phi
        self.lm = lm
        self.dec = dec

        self.integrator = torchquad.MonteCarlo()
        self.space_range = space_range

    def forward(
            self,
            batch: list[Trajectory],
            cfg: SimpleNamespace,
            device: torch.device
    ):
        context = [traj.context(merge=True)[0].astype(np.float32) for traj in batch]
        context, mask = (torch.tensor(t, device=device) for t in pad_context(context))

        gamma, tau = self.enc(context, mask)
        z_0 = gamma + tau * torch.randn_like(tau)

        t_unif = torch.linspace(0, cfg.T-cfg.t_ctx, cfg.t_unif_res, device=device)
        z_unif = odeint(self.dyn, z_0, t_unif, rtol=cfg.rtol, atol=cfg.atol, method=cfg.solver)
        z_unif = z_unif.transpose(0, 1)  # type: ignore

        process_loglik = torch.tensor([0.0], device=device)
        obs_loglik = torch.tensor([0.0], device=device)
        batch_mae = []
        for j in range(len(batch)):
            t, x, y = [torch.tensor(b, dtype=torch.float32, device=device) for b in batch[j].observations()]

            u_hat = self.phi(t, x, t_unif, z_unif[j])
            lm_hat = self.lm(u_hat)
            mu_hat = self.dec(u_hat)

            process_loglik += torch.sum(torch.log(lm_hat))
            process_loglik -= self.integrator.integrate(
                lambda tx: self.lm(self.phi(tx[:, 0].contiguous(), tx[:, 1:], t_unif, z_unif[j])),
                dim=1+cfg.d_x,
                N=32,
                integration_domain=[[0, cfg.T-cfg.t_ctx], *self.space_range],
                backend="torch",
            )

            obs_loglik += Normal(mu_hat, cfg.sig_y).log_prob(y).sum()

            batch_mae.append(torch.mean(torch.abs(mu_hat - y)).item())

        kl_qp = kl_norm_norm(gamma, torch.zeros_like(gamma), tau, torch.ones_like(tau)).sum()

        aux = {
            "pred_mae": sum(batch_mae) / len(batch_mae),
            "context": context[-1].detach().cpu().numpy(),
            "observations": batch[-1].observations(merge=True)[0],
            "u_hat": u_hat.detach().cpu().numpy(),  # type: ignore
            "lm_hat": lm_hat.detach().cpu().numpy(),  # type: ignore
            "mu_hat": mu_hat.detach().cpu().numpy(),  # type: ignore
        }

        return obs_loglik, process_loglik, kl_qp, aux


# class Model(Module):
#     def __init__(
#         self,
#         enc: Encoder,
#         dyn: DynamicsFunction,
#         phi: ContinuousDecoder,
#         lm: Module,
#         dec: Module,
#         space_range: list[list[float]],
#     ) -> None:

#         super().__init__()
#         self.enc = enc
#         self.dyn = dyn
#         self.phi = phi
#         self.lm = lm
#         self.dec = dec

#         self.integrator = torchquad.MonteCarlo()
#         self.space_range = space_range

#     def forward(
#             self,
#             batch: list[Trajectory],
#             cfg: SimpleNamespace,
#             device: torch.device
#     ):
#         # Use the encoder to map context to variational parameters of q(z_1)
#         # and sample latent initial states.
#         context = [traj.context(merge=True)[0].astype(np.float32) for traj in batch]
#         context, mask = (torch.tensor(t, device=device) for t in pad_context(context))
#         gamma, tau = self.enc(context, mask)
#         z_0 = gamma + tau * torch.randn_like(tau)

#         # Simulate the latent dynamics over [t_ctx, T] and save the state at t_unif_res points.
#         t_unif = torch.linspace(0, cfg.T-cfg.t_ctx, cfg.t_unif_res, device=device)
#         z_unif = odeint(self.dyn, z_0, t_unif, rtol=cfg.rtol, atol=cfg.atol)
#         z_unif = z_unif.transpose(0, 1)  # type: ignore

#         # Interpolate the latent states.
#         z_t = interpolate(batch, t_unif, z_unif)

#         # Evaluate u(t, x) and use it to evaluate intensty and observation functions.
#         x = torch.nested.nested_tensor([traj.observations()[1] for traj in batch], dtype=torch.float32, device=device)
#         u = self.phi(z_t, x)
#         u, ps = pack(u.unbind(), "* du")
#         lm = self.lm(u)
#         mu = self.dec(u)

#         # Evaluate elbo components.
#         y, _ = pack([torch.tensor(traj.observations()[2], dtype=torch.float32, device=device) for traj in batch], "* du")
#         obs_loglik = Normal(mu, cfg.sig_y).log_prob(y).sum()

#         process_loglik = torch.sum(torch.log(lm))
#         lm = unpack(lm, ps, "* du")
#         vol = np.prod([cfg.T-cfg.t_ctx, *[r[1]-r[0] for r in self.space_range]])
#         for i in range(len(lm)):
#             with torch.no_grad():
#                 w = 1.0 / lm[i]
#                 w /= w.sum()
#             process_loglik -= vol * (w * lm[i]).sum()

#         kl_qp = kl_norm_norm(gamma, torch.zeros_like(gamma), tau, torch.ones_like(tau)).sum()

#         # Eval metrics and log
#         with torch.no_grad():
#             batch_mae = torch.mean(torch.abs(mu - y)).item()
#         N = len(batch[0].observations()[0])
#         aux = {
#             "pred_mae": batch_mae,
#             "context": context[0].detach().cpu().numpy(),
#             "observations": batch[0].observations(merge=True)[0],
#             "u_hat": u[0:N].detach().cpu().numpy(),  # type: ignore
#             "lm_hat": lm[0].detach().cpu().numpy(),  # type: ignore
#             "mu_hat": mu[0:N].detach().cpu().numpy(),  # type: ignore
#         }

#         return obs_loglik, process_loglik, kl_qp, aux


# def interpolate(batch: list[Trajectory], t_unif: Tensor, z_unif: Tensor) -> Tensor:
#     """Interpolates z_unfi at t_unfif to t.

#     Args:
#         batch: List fo trajectories, where each trajectory has an interpolation matrix.
#         t_unif: Temporal locations of z_unif, has shape (N_unif, ).
#         z_unif: Latent states used for interpolation, has shape (batch, N_unif, d_z).

#     Returns:
#         Nested tensor with shape (batch, N_i, d_z)
#     """
#     # Dummy value.
#     # z_t = torch.nested.nested_tensor([torch.rand((sum(traj.obs_mask), 32)) for traj in batch], device=device)

#     # Sequential default.
#     z_t = []
#     for i in range(len(batch)):
#         t = torch.tensor(batch[i].observations()[0], dtype=torch.float32, device=t_unif.device)
#         inds = torch.searchsorted(t_unif, t)
#         z_t.append(z_unif[i, inds, :])
#     z_t = torch.nested.as_nested_tensor(z_t)  # as_nested_tensor to save autograd history

#     return z_t
