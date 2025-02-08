import os
import glob
import pickle
import argparse

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, LinearLR, CosineAnnealingLR

from torchquad import set_up_backend

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from stpp.data import Trajectory

from stpp.encoder import Encoder
from stpp.dynamics import DynamicsFunction
from stpp.intensity import IntensityCorrection
from stpp.decoder import ContinuousDecoder
from stpp.model import Model

from stpp.utils.utils import set_seed, init_weights, create_mlp


ndarray = np.ndarray
Module = torch.nn.Module
Tensor = torch.Tensor


DATASET_NAME = "NavierStokes2D"


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data.
    parser.add_argument("--data_dir", type=str, default="./data/NavierStokes2D/medium_u_cont/", help="Path to the dataset.")

    parser.add_argument("--T", type=float, default=2.0, help="Full time interval.")
    parser.add_argument("--t_ctx", type=float, default=0.5, help="Context time interval.")

    parser.add_argument("--d_x", type=int, default=2, help="Spatial coordinates dimension")
    parser.add_argument("--d_z", type=int, default=368, help="Latent state dimension")
    parser.add_argument("--d_u", type=int, default=1, help="Spatiotemporal latent state dimension")
    parser.add_argument("--d_y", type=int, default=1, help="Observations dimension")

    parser.add_argument("--val_size", type=float, default=0.1, help="Val. set size in %.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size in %.")

    parser.add_argument("--data_seed", type=int, default=13, help="Data random seed.")


    # Model (general).
    parser.add_argument("--model_seed", type=int, default=13, help="Model random seed.")
    parser.add_argument("--t_unif_res", type=int, default=100, help="Number of point in unfirom temporal grid used for intepolation.")
    parser.add_argument("--interp_method", type=str, default="linear", help="Latent state interpolation method (nearest/linear).")


    # Model (encoder).
    parser.add_argument("--d_model", type=int, default=192, help="Transformer stack feature dimension.")
    parser.add_argument("--n_attn_heads", type=int, default=4, help="Transformer stack attention heads.")
    parser.add_argument("--n_tf_layers", type=int, default=5, help="Number of layers in transformer stack.")


    # Model (dynamics).
    parser.add_argument("--dyn_hid_layers", type=int, default=3, help="Number of hidden layers in dynamics function.")
    parser.add_argument("--dyn_latent_dim", type=int, default=368, help="Hidden layer dimension in dynamics function.")


    # Model (phi).
    parser.add_argument("--phi_hid_layers", type=int, default=3, help="Number of hidden layers in state decoder.")
    parser.add_argument("--phi_latent_dim", type=int, default=512, help="Hidden layer dimension in state decoder.")


    # Model (lm).
    parser.add_argument("--lm_hid_layers", type=int, default=3, help="Number of hidden layers in intensity function.")
    parser.add_argument("--lm_latent_dim", type=int, default=256, help="Hidden layer dimension in intensity function.")


    # Training/validation/testing.
    parser.add_argument("--obs_lik_scaler", type=float, default=1, help="Scaler for observation likelihood term.")
    parser.add_argument("--sig_y", type=float, default=0.001, help="Observation variance.")

    parser.add_argument("--n_iters", type=int, default=25000, help="Number of training iterations.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")

    parser.add_argument("--solver", type=str, default="dopri5", help="Name of the ODE solver (see torchdiffeq).")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for ODE solver.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for ODE solver.")

    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda)")
    parser.add_argument("--training_seed", type=int, default=13, help="Training random seed.")
    parser.add_argument("--group", default="None", help="Group for wandb.")
    parser.add_argument("--tags", default=["no_tag"], nargs="+", help="Tags for wandb.")
    parser.add_argument("--name", type=str, default="tmp", help="Name of the run.")

    parser.add_argument("--visualize", type=int, default=1, help="Visualize predictions on validation set flag (0 - no, 1 - yes).")
    parser.add_argument("--n_mc_samples", type=int, default=10, help="Number of samples for Monte Carlo integration.")

    parser.add_argument("--model_dir", type=str, default="./models/NavierStokes2D/", help="Folder for saving/loading models.")

    return parser


def create_datasets(cfg: SimpleNamespace) -> tuple[list[Trajectory], ...]:
    set_seed(cfg.data_seed)

    # Read all data.
    data = []
    for path in tqdm(glob.glob(cfg.data_dir+"*.pkl"), desc="Reading data"):
        with open(path, "rb") as f:
            traj = pickle.load(f)[0:3]  # t, x, y

            if traj[0].min() < 0 or traj[0].max() > 2:
                raise RuntimeError("Time points must be between 0 and 2.")
            if traj[1].min() < 0 or traj[1].max() > 1:
                raise RuntimeError("Spatial locations must be between 0 and 1.")

            traj = Trajectory(traj[0], cfg.t_ctx, traj[1], traj[2])
            data.append(traj)

    # Filter data.
    filt_data = []
    for traj in tqdm(data, desc="Filtering data"):
        # if sum(traj.ctx_mask) >= 30:
        #     filt_data.append(traj)
        filt_data.append(traj)
    print(f"Data size before filtering: {len(data)}")
    print(f"Data size after filtering: {len(filt_data)}")

    # Split data.
    train, test = train_test_split(filt_data, test_size=cfg.test_size, random_state=cfg.data_seed)
    train, val = train_test_split(train, test_size=cfg.val_size, random_state=cfg.data_seed)

    return train, val, test


def create_model(cfg: SimpleNamespace) -> Model:
    set_seed(cfg.model_seed)

    enc = Encoder(
        d_x=cfg.d_x,
        d_y=cfg.d_y,
        d_z=cfg.d_z,
        d_model=cfg.d_model,
        n_attn_heads=cfg.n_attn_heads,
        n_tf_layers=cfg.n_tf_layers,
        dropout_prob=0.05,
    )

    dyn = DynamicsFunction(
        f=create_mlp(
            input_size=cfg.d_z,
            output_size=cfg.d_z,
            hidden_size=cfg.dyn_latent_dim,
            num_hidden_layers=cfg.dyn_hid_layers,
            activation_func=nn.GELU,
            use_dropout=False,
            dropout_prob=0.05,
        )
    )

    # dyn = DynamicsFunction(
    #     f=nn.Sequential(
    #         nn.Linear(cfg.d_z, cfg.d_z)
    #     )
    # )

    # Set backend for torchquad
    if cfg.device == "cuda":
        set_up_backend("torch", data_type="float32")

    phi = ContinuousDecoder(
        d_z=cfg.d_z,
        d_x=cfg.d_x,
        d_u=cfg.d_u,
        f=create_mlp(
            input_size=cfg.d_z,
            output_size=cfg.d_u,
            hidden_size=cfg.phi_latent_dim,
            num_hidden_layers=cfg.phi_hid_layers,
            activation_func=nn.GELU,
            use_layer_norm=True,
            use_dropout=True,
            dropout_prob=0.05,
        ),
        interp_method=cfg.interp_method,
    )

    lm = nn.Sequential(
        create_mlp(
            input_size=cfg.d_u,
            output_size=1,
            hidden_size=cfg.lm_latent_dim,
            num_hidden_layers=cfg.lm_hid_layers,
            activation_func=nn.GELU,
            use_dropout=False,
            dropout_prob=0.05,
        ),
        IntensityCorrection(0.0001),
    )

    dec = nn.Sequential(
        nn.Identity(),
        # create_mlp(
        #     input_size=cfg.d_u,
        #     output_size=cfg.d_y,
        #     hidden_size=256,
        #     num_hidden_layers=2,
        #     activation_func=nn.GELU,
        #     use_dropout=False,
        #     dropout_prob=0.05,
        # )
    )

    space_range = [[0.0, 1.0], [0.0, 1.0]]

    model = Model(enc, dyn, phi, lm, dec, space_range)
    model.apply(init_weights)

    return model


def get_scheduler(optimizer, warmup_iters):
    sched = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, 1e-3, 1), ConstantLR(optimizer, 1)],
        milestones=[warmup_iters]
    )
    return sched


def visualize(
    context: ndarray,
    observations: ndarray,
    u: ndarray,
    lm: ndarray,
    mu: ndarray,
    title: str,
    dir: str,
    name: str,
    cfg: SimpleNamespace,
) -> None:

    t = observations[:, 0]
    nrows = 10
    window_size = 1.0 / nrows

    fig, ax = plt.subplots(nrows, 5, figsize=(4*5, 3*nrows), constrained_layout=True)

    im0 = ax[0, 0].scatter(
        context[:, 1], context[:, 2],
        c=context[:, 3],
        marker="X", s=30, cmap="coolwarm"
    )
    ax[0, 0].set_xlim(0, 1)
    ax[0, 0].set_ylim(0, 1)

    for i in range(nrows):
        mask = (t >= (i * window_size)) & (t <= ((i + 1) * window_size))

        if sum(mask) < 10:
            continue

        im1 = ax[i, 1].tricontourf(
            observations[mask, 1], observations[mask, 2], observations[mask, 3].ravel(),
            cmap='coolwarm'
        )
        ax[i, 1].scatter(observations[mask, 1], observations[mask, 2], c="k", marker=".", s=30, alpha=0.3)
        ax[i, 1].set_xlim(0, 1)
        ax[i, 1].set_ylim(0, 1)

        im2 = ax[i, 2].tricontourf(
            observations[mask, 1], observations[mask, 2], mu[mask].ravel(),
            cmap='coolwarm'
        )
        ax[i, 2].set_xlim(0, 1)
        ax[i, 2].set_ylim(0, 1)

        for im_, ax_ in zip([im0, im1, im2], ax[i, 0:3]):
            fig.colorbar(im_, ax=ax_)

    # im3 = ax[3].tricontourf(
    #     observations[:, 0], observations[:, 1:1+cfg.d_x].ravel(), lm.ravel(),
    #     cmap='coolwarm'
    # )

    # im4 = ax[4].tricontourf(
    #     observations[:, 0], observations[:, 1:1+cfg.d_x].ravel(), u.ravel(),
    #     cmap='coolwarm'
    # )

    # for im_, ax_ in zip([im0, im1, im2, im3, im4], ax):
    #     fig.colorbar(im_, ax=ax_)

    fig.suptitle(title)

    for i, title in enumerate(["Context", "Observations", "mu", "lm", "u"]):
        ax[0, i].set_title(title)

    # for i in range(len(ax)):
    #     ax[i].set_xlabel("Time")
    #     ax[i].set_ylabel("Space")

    if not os.path.isdir(dir):
        os.makedirs(dir)

    plt.savefig(dir+name)
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
