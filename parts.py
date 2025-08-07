import os
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from model.diffusion import Diffusion
from scipy.io import savemat
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from typing import Optional

from common.logger import logger
from common.loss import SSIMLoss
from common.utils import calculate_psnr, calculate_ssim, seconds_to_dhms
from common.wrapper import error_wrap
from components.datawrapper import DataWrapper
from components.metriccontroller import MetricController
from params import ModelConfig, config, modelconfig


NETWORK = Diffusion | torch.nn.DataParallel[Diffusion]
OPTIM = Adam | AdamW


def safe_mean_loss(loss, dims=(1, 2, 3), keepdim=True):
    if loss.dim() == 0:
        return loss
    else:
        valid_dims = tuple(d for d in dims if d < loss.dim())
        if len(valid_dims) > 0:
            return torch.mean(loss, dim=valid_dims, keepdim=keepdim)
        else:
            return loss


def get_network(
    device: torch.device,
    model_type: str,
    modelconfig: ModelConfig,
) -> NETWORK:
    if device is None:
        raise TypeError("device must not be None")

    if model_type in ["diffusion"]:
        if not isinstance(modelconfig, ModelConfig):
            raise TypeError("Expected a single ModelConfig for 'diffusion' model_type")
        return Diffusion(device=device, modelconfig=modelconfig)
    else:
        raise KeyError(f"Unsupported model type: {model_type}")


def get_optim(
    network: NETWORK,
    optimizer: str,
) -> OPTIM:
    if optimizer == "adam":
        return Adam(
            network.parameters(),
            betas=(0.9, 0.99),
        )
    elif optimizer == "adamw":
        return AdamW(
            network.parameters(),
            betas=(0.9, 0.99),
            weight_decay=0.005,
        )
    else:
        raise KeyError("optimizer not matched")


def get_loss_func(
    loss_model: str,
) -> Callable:
    if loss_model == "l1":
        return torch.nn.L1Loss(reduction="none")
    elif loss_model == "l2":
        return torch.nn.MSELoss(reduction="none")
    elif loss_model == "ssim":
        return SSIMLoss(reduced=False)
    else:
        raise KeyError("loss func not matched")


def get_learning_rate(
    epoch: int,
    lr: float,
    lr_decay: float,
    lr_tol: int,
) -> float:
    factor = epoch - lr_tol if lr_tol < epoch else 0
    return lr * (lr_decay**factor)


def set_optimizer_lr(
    optimizer: OPTIM,
    learning_rate: float,
) -> OPTIM:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    return optimizer


def log_summary(
    init_time: float,
    state: MetricController,
    log_std: bool = False,
) -> None:
    spend_time = seconds_to_dhms(time.time() - init_time)
    for key in state.state_dict:
        if log_std:
            summary = (
                f"{spend_time} | {key}: {state.mean(key):0.3e} + {state.std(key):0.3e} "
            )
            logger.info(summary)
        else:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e}"
            logger.info(summary)


# depends on modelconfig


def save_checkpoint(
    network: NETWORK,
    optimizer: torch.optim.Optimizer,
    run_dir: Path,
    epoch: int | None = None,
    best: bool = False,
) -> None:
    if epoch is None:
        epoch = "best"
    if best:
        save_dir = run_dir / "checkpoints" / "best"
    else:
        save_dir = run_dir / "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    model_config_dict = asdict(modelconfig)

    network = network.module if isinstance(network, torch.nn.DataParallel) else network

    checkpoint = {
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "model_config": model_config_dict,
    }

    ckpt_path = save_dir / f"checkpoint_{epoch}.ckpt"
    torch.save(checkpoint, ckpt_path)


@error_wrap
def train_epoch(
    train_state: MetricController,
    train_loader: DataLoader,
    train_len: int,
    network: NETWORK,
    optimizer: OPTIM,
) -> None:
    loss_func = get_loss_func(config.loss_model)
    network.train()

    logging_cnt: int = 1
    img_cnt: int = 0
    num_batches: int = 0

    for _data in train_loader:
        if _data is None:
            continue
        if config.model_type in ["diffusion", "sbflow", "hybrid"]:
            img = _data[DataWrapper.IMG].to(config.device)
            cond = _data[DataWrapper.COND].to(config.device)
            mask = _data[DataWrapper.MASK].to(config.device)
            if img is None or cond is None:
                continue
            if not (
                (isinstance(img, torch.Tensor)) and (isinstance(cond, torch.Tensor))
            ):
                raise KeyError("elements has to be tensor")

            output, target = network.forward(
                lab=img,
                cond=cond,
                mask=mask,
            )

            loss = loss_func(output, target)
            if modelconfig.algorithm in ["biflow"]:
                loss = safe_mean_loss(loss, dims=(1, 2, 3, 4), keepdim=True)
            else:
                loss = safe_mean_loss(loss, dims=(1, 2, 3), keepdim=True)
            optimizer.zero_grad()
            torch.mean(loss).backward()
            optimizer.step()

            train_state.add("loss", loss.mean().view(1, 1, 1, 1))
        else:
            raise KeyError("model type not matched")

        img_cnt += img.shape[0]
        if img_cnt > (train_len / config.logging_density * logging_cnt):
            log_summary(init_time=config.init_time, state=train_state)
            logging_cnt += 1
        num_batches += 1

    logger.debug(f"Number of batches this epoch: {num_batches}")
    log_summary(init_time=config.init_time, state=train_state)


@error_wrap
def test_part(
    epoch: int,
    metric_state: MetricController,
    data_loader: DataLoader,
    network: NETWORK,
    run_dir: Path,
    save_val: bool,
) -> tuple[float, float, float]:
    loss_func = get_loss_func(config.loss_model)
    network.eval()

    img_cnt: int = 0
    for _data in data_loader:
        if _data is None:
            continue
        if config.model_type in ["diffusion", "sbflow", "hybrid"]:
            img = _data[DataWrapper.IMG].to(config.device)
            cond = _data[DataWrapper.COND].to(config.device)
            mask = _data[DataWrapper.MASK].to(config.device)
            if img is None or cond is None:
                continue

            assert isinstance(img, torch.Tensor)
            assert isinstance(cond, torch.Tensor)

            output = network(
                cond=cond,
                interval=config.interval,
                mode="recon",
            )

        else:
            raise KeyError("model type not matched")

        loss = loss_func(output, img)
        loss = safe_mean_loss(loss, dims=(1, 2, 3), keepdim=True)
        if loss.dim() == 0:
            loss = loss.view(1, 1, 1, 1)
        elif loss.dim() == 1:
            loss = loss.view(-1, 1, 1, 1)
        metric_state.add("loss", loss)
        metric_state.add("psnr", calculate_psnr(output, img, mask))
        metric_state.add("ssim", calculate_ssim(output, img, mask))

        batch_cnt = img.shape[0]
        if not save_val:
            img_cnt += batch_cnt
            continue

        if img_cnt > config.save_max_idx:
            continue

        test_dir = run_dir / f"test/ep_{epoch}"
        os.makedirs(test_dir, exist_ok=True)
        for i in range(batch_cnt):
            idx = img_cnt + i + 1

            save_dict = {
                "out": output.cpu().detach().numpy()[i, ...],
                "img": img.cpu().detach().numpy()[i, ...],
                "cond": cond.cpu().detach().numpy()[i, ...],
                "mask": mask.cpu().detach().numpy()[i, ...],
            }
            savemat(f"{test_dir}/{idx}_res.mat", save_dict)

        img_cnt += batch_cnt

    log_summary(init_time=config.init_time, state=metric_state, log_std=True)

    primary_psnr = metric_state.mean("psnr")
    primary_loss = metric_state.mean("loss")
    primary_ssim = metric_state.mean("ssim")

    return primary_psnr, primary_loss, primary_ssim
