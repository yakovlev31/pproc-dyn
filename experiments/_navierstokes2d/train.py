from types import SimpleNamespace

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tqdm import tqdm

from stpp.utils.utils import (BatchMovingAverage, set_seed, save_model)
import stpp.utils.navierstokes2d as utils


torch.backends.cudnn.benchmark = True  # type: ignore


# Read parameters.
argparser = utils.create_argparser()
cfg = SimpleNamespace(**vars(argparser.parse_args()))
cfg.tags.append("train")


# Load data.
train_data, val_data, _ = utils.create_datasets(cfg)


# Create model.
device = torch.device(cfg.device)
model = utils.create_model(cfg).to(device)

# print("Number of Model Parameters:", utils.count_parameters(model))

# Training.
optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
bma = BatchMovingAverage(k=10)
scheduler = utils.get_scheduler(optimizer, warmup_iters=int(0.01*cfg.n_iters))

wandb.init(
    mode="online",  # online/disabled
    project="stpp",
    group=cfg.group,
    tags=cfg.tags,
    name=cfg.name,
    config=vars(cfg),
    save_code=True,
)

set_seed(cfg.training_seed)
for i in tqdm(range(cfg.n_iters)):
    model.train()

    train_batch_inds = np.random.choice(len(train_data), size=cfg.batch_size, replace=False)
    train_batch = [train_data[i] for i in train_batch_inds]

    obs_lik, process_lik, kl_qp, aux = model(train_batch, cfg, device)

    obs_lik *= (len(train_data) / cfg.batch_size) * cfg.obs_lik_scaler
    process_lik *= len(train_data) / cfg.batch_size
    kl_qp *= len(train_data) / cfg.batch_size

    loss = -(obs_lik + process_lik - kl_qp)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if i % max(1, int(0.005 * cfg.n_iters)) == 0 or i == cfg.n_iters - 1:
        with torch.no_grad():
            model.eval()

            val_batch_inds = np.random.choice(len(val_data), size=cfg.batch_size, replace=False)
            val_batch = [val_data[i] for i in val_batch_inds]

            obs_lik_val, process_lik_val, kl_qp_val, aux_val = model(val_batch, cfg, device)

            obs_lik_val *= (len(train_data) / cfg.batch_size) * cfg.obs_lik_scaler
            process_lik_val *= len(train_data) / cfg.batch_size
            kl_qp_val *= len(train_data) / cfg.batch_size

            loss_val = -(obs_lik_val + process_lik_val - kl_qp_val)

            bma.add_value(aux_val["pred_mae"])
            if bma.get_average() <= bma.get_min_average():
                save_model(model, cfg.model_dir, cfg.name)
                # print(f"Model saved at iteration {i}.")

            wandb.log(
                {
                    "obs_loglik_train": obs_lik.item(),
                    "process_loglik_train": process_lik.item(),
                    "kl_qp_train": kl_qp.item(),
                    "mae_train": aux["pred_mae"],

                    "obs_loglik_val": obs_lik_val.item(),
                    "process_loglik_val": process_lik_val.item(),
                    "kl_qp_val": kl_qp_val.item(),
                    "mae_val": aux_val["pred_mae"],

                    "lr": optimizer.param_groups[0]['lr'],
                },
                step=i,
            )

        # if cfg.visualize == 1:
        #     utils.visualize(
        #         aux_val["context"],
        #         aux_val["observations"],
        #         aux_val["u_hat"],
        #         aux_val["lm_hat"],
        #         aux_val["mu_hat"],
        #         title=f"{utils.DATASET_NAME}/{cfg.name}/iter_{i}",
        #         dir=f"./img/{utils.DATASET_NAME}/{cfg.name}/",
        #         name=f"iter_{i}.png",
        #         cfg=cfg,
        #     )
