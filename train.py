import os
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common.logger import logger, logger_add_handler
from common.utils import (
    call_next_id,
    separator,
)

from torch.utils.tensorboard import SummaryWriter

from common.wrapper import error_wrap
from components.datawrapper import get_data_wrapper_loader
from components.metriccontroller import MetricController
from params import config, modelconfig , gmm_modelconfig
from parts import (
    NETWORK,
    get_learning_rate,
    get_network,
    get_optim,
    save_checkpoint,
    set_optimizer_lr,
    test_part,
    train_epoch,
)

warnings.filterwarnings("ignore")


class Trainer:
    run_dir: Path
    network: NETWORK
    train_loader: DataLoader
    train_len: int
    valid_loader: DataLoader

    def __init__(self) -> None:

        self.start_epoch = 0

        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

        config.init_time = time.time()
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dir setting
        self.run_dir = config.run_dir / f"{call_next_id(config.run_dir):05d}_train"
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        os.makedirs(self.run_dir, exist_ok=True)
        logger_add_handler(logger, f"{self.run_dir/'log.log'}", config.log_lv)

        # log config
        logger.info(separator())
        logger.info("General Config")
        config_dict = asdict(config)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")
        logger.info(separator())
        logger.info("Model Config")
        config_dict = asdict(modelconfig)
        for k in config_dict:
            if k in ["marginal_prob_std_fn", "diffusion_coeff_fn"]:
                continue
            logger.info(f"{k}:{config_dict[k]}")
        logger.info(f"sigma (for diffusion functions): {modelconfig.sigma}")

    def __call__(
        self,
    ) -> None:
        self._set_data()
        self._set_network()
        self._train()

    @error_wrap
    def _set_data(
        self,
    ) -> None:
        logger.info(separator())
        self.train_loader, _, self.train_len = get_data_wrapper_loader(
            file_path=config.train_dataset,
            config=config,
            training_mode=True,
            batch=config.train_batch,
            num_workers=config.num_workers,
            shuffle=True,
            debug_mode=config.debugmode,
        )
        logger.info(f"Train dataset length : {self.train_len}")

        self.valid_loader, _, valid_len = get_data_wrapper_loader(
            file_path=config.valid_dataset,
            config=config,
            training_mode=False,
            batch=config.valid_batch,
            num_workers=config.num_workers,
            shuffle=False,
            debug_mode=config.debugmode,
        )
        logger.info(f"Valid dataset length : {valid_len}")

        self.test_loader, _, test_len = get_data_wrapper_loader(
            file_path=config.test_dataset,
            config=config,
            training_mode=False,
            batch=config.valid_batch,
            num_workers=config.num_workers,
            shuffle=False,
            debug_mode=config.debugmode,
        )
        logger.info(f"Test dataset length : {test_len}")

    @error_wrap
    def _set_network(
        self,
    ) -> None:
        self.network = get_network(
            device=config.device,
            model_type=config.model_type,
            modelconfig=modelconfig,
            gmm_modelconfig=gmm_modelconfig,
        )
        self.optim = get_optim(
            network=self.network,
            optimizer=config.optimizer,
        )

        if config.parallel:
            self.network = torch.nn.DataParallel(self.network)

        self.network = self.network.to(config.device)

    @error_wrap
    def _train(
        self,
    ) -> None:
        logger.info(separator())
        logger.info("Train start")
        train_state = MetricController()
        best_metric: float = 0.1
        writer = SummaryWriter(log_dir=self.run_dir)

        for epoch in range(self.start_epoch, config.train_epoch):
            logger.info(f"Epoch: {epoch}")
            train_state.reset()
            lr_epoch = get_learning_rate(
                epoch=epoch,
                lr=config.lr,
                lr_decay=config.lr_decay,
                lr_tol=config.lr_tol,
            )
            optimizer = set_optimizer_lr(
                optimizer=self.optim,
                learning_rate=lr_epoch,
            )
            logger.info(f"Learning rate: {lr_epoch:0.3e}")

            train_epoch(
                train_state=train_state,
                train_loader=self.train_loader,
                train_len=self.train_len,
                network=self.network,
                optimizer=optimizer,
            )
            writer.add_scalar("Loss/train_epoch", train_state.mean("loss"), epoch)
            
            save_checkpoint(
                network=self.network,
                optimizer=optimizer,
                run_dir=self.run_dir,
                epoch=epoch,
            )

            if epoch < config.valid_tol:
                continue

            if epoch % config.valid_interval == 0:
                primary_psnr, primary_loss, primary_ssim = self._valid(epoch)

                writer.add_scalar("PSNR/val_epoch", primary_psnr, epoch)
                writer.add_scalar("Loss/val_epoch", primary_loss, epoch)
                writer.add_scalar("SSIM/val_epoch", primary_ssim, epoch)

                if config.primary_metric == "loss":
                    primary_metric = primary_loss
                elif config.primary_metric == "psnr":
                    primary_metric = primary_psnr
                elif config.primary_metric == "ssim":
                    primary_metric = primary_ssim
                else:
                    raise ValueError(f"Unknown primary_metric: {config.primary_metric}")

                if primary_metric > best_metric:
                    best_metric = primary_metric
                    logger.success("Best model renew")
                    save_checkpoint(
                        network=self.network,
                        optimizer=optimizer,
                        run_dir=self.run_dir,
                        epoch=epoch,
                        best = True,
                    )

                    self._test(epoch)

        writer.close()

    @error_wrap
    def _valid(
        self,
        epoch: int,
    ) -> float:
        metric_state = MetricController()
        metric_state.reset()
        logger.info("Valid")
        with torch.no_grad():
            primary_psnr, primary_loss, primary_ssim = test_part(
                epoch=epoch,
                metric_state=metric_state,
                data_loader=self.valid_loader,
                network=self.network,
                run_dir=self.run_dir,
                save_val=False,
            )
        return primary_psnr, primary_loss, primary_ssim

    @error_wrap
    def _test(
        self,
        epoch: int,
    ) -> float:
        metric_state = MetricController()
        metric_state.reset()
        logger.info("Test")
        with torch.no_grad():
            test_part(
                epoch=epoch,
                metric_state=metric_state,
                data_loader=self.test_loader,
                network=self.network,
                run_dir=self.run_dir,
                save_val=True,
            )


if __name__ == "__main__":
    trainer = Trainer()
    trainer()