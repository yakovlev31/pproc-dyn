from types import SimpleNamespace

import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint

import wandb
from tqdm import tqdm

from stpp.utils.utils import load_model, pad_context, set_seed
import stpp.utils.scalarflow2d as utils


torch.backends.cudnn.benchmark = True  # type: ignore


# Read parameters.
argparser = utils.create_argparser()
cfg = SimpleNamespace(**vars(argparser.parse_args()))
cfg.tags.append("test")


# Load data.
_, _, test_data = utils.create_datasets(cfg)


# Create model.
device = torch.device(cfg.device)
model = utils.create_model(cfg).to(device)
load_model(model, cfg.model_dir, cfg.name, device)
model.eval()

wandb.init(
    mode="disabled",  # online/disabled
    project="stpp",
    group=cfg.group,
    tags=cfg.tags,
    name=cfg.name,
    config=vars(cfg),
    save_code=True,
)


set_seed(cfg.training_seed)
mae, negloglik = [], []
for i in tqdm(range(0, len(test_data))):
    t, x, y = [torch.tensor(arr, dtype=torch.float32, device=device) for arr in test_data[i].observations()]

    context = [traj.context(merge=True)[0].astype(np.float32) for traj in [test_data[i]]]
    context, mask = (torch.tensor(t, device=device) for t in pad_context(context))

    gamma, tau = model.enc(context, mask)
    traj_mae, traj_negloglik = [], []
    for j in range(cfg.n_mc_samples):
        # z_0 = gamma + tau * torch.randn_like(tau)
        z_0 = gamma  # use the mean

        t_unif = torch.linspace(0, cfg.T-cfg.t_ctx, cfg.t_unif_res, device=device)
        z_unif = odeint(model.dyn, z_0, t_unif, rtol=cfg.rtol, atol=cfg.atol).transpose(0, 1)  # type: ignore

        u_hat = model.phi(t, x, t_unif, z_unif[0])
        mu_hat = model.dec(u_hat)
        traj_mae.append(nn.L1Loss()(mu_hat, y).item())

        process_loglik = torch.sum(torch.log(model.lm(u_hat)))
        process_loglik -= model.integrator.integrate(
            lambda tx: model.lm(model.phi(tx[:, 0].contiguous(), tx[:, 1:].contiguous(), t_unif, z_unif[0])),
            dim=1+cfg.d_x,
            N=256,
            integration_domain=[[0, cfg.T-cfg.t_ctx], *model.space_range],
            backend="torch",
        )
        traj_negloglik.append(-process_loglik.item() / len(t))  # per-event

    mae.append(np.mean(traj_mae))
    negloglik.append(np.mean(traj_negloglik))

print(np.mean(mae), np.std(mae))
print(np.mean(negloglik), np.std(negloglik))  # this is the same as in NSTPP, so leave it.

wandb.run.summary.update(  # type: ignore
    {
        "mean_test_mae": np.mean(mae),
        "mean_test_negloglik": np.mean(negloglik),
    }
)
