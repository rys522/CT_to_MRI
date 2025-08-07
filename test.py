import argparse
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import nibabel as nib
import glob

import torch
from model.diffusion import Diffusion, HybridDiffusion, SBFlow
from scipy.io import savemat
from torch.utils.data import DataLoader

from common.logger import logger, logger_add_handler
from common.utils import calculate_psnr, calculate_ssim, call_next_id, separator
from common.wrapper import error_wrap
from components.datawrapper import DataWrapper, get_data_wrapper_loader
from components.metriccontroller import MetricController
from params import ModelConfig, GMM_ModelConfig
from parts import get_loss_func, get_network, log_summary

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent
log_v = 'log_2025_07_30_clip_otflow_baseline_256_256'
CK_ROOT = ROOT_DIR / Path(f"log/{log_v}")
DATA_ROOT = Path("/fast_storage/intern1/CT_MRI_data_256_256/brain")
CKPT = 160

TEST_MRI_PATH = str(DATA_ROOT / "test_mri.h5")
TEST_CT_PATH = str(DATA_ROOT / "test_ct.h5")
TEST_MASK_PATH = str(DATA_ROOT / "test_mask.h5")

SAVE_ROOT = Path(f"/home/NAS_intern/intern/Personal_Folder/Yoonseok/experiments/{log_v}/nifti_converted")

@dataclass
class TestConfig:
    # Dataset
    
    ckpt_path = CK_ROOT / Path(f"/0000_train/checkpoints/best/checkpoint_{CKPT}.ckpt")
    test_mri_path: str = TEST_MRI_PATH
    test_ct_path: str = TEST_CT_PATH
    test_mask_path: str = TEST_MASK_PATH

    test_dataset = (test_mri_path, test_ct_path, test_mask_path)
    
    debugmode: bool = False

    #data process
    ct_clip: tuple[float, float] = (-600, 700)
    scale: float = 0.15

    loss_model: Literal["l1", "l2", "ssim"] = "l2"

    # Logging
    log_lv: Literal ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    run_dir: Path = ROOT_DIR / "log/log_2025_07_30_clip_sbflow_baseline_256_256"
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["diffusion", "sbflow" ,"hybrid"] = "diffusion"

    gpu: str = "0,1,2,3"
    batch_size: int = 32
    device: torch.device | None = None

    # Hyper
    interval: int = 5

    tag: str = ""


parser = argparse.ArgumentParser(description="Training Configuration")
test_dict = asdict(TestConfig())
for key, default_value in test_dict.items():
    if isinstance(default_value, bool):
        parser.add_argument(
            f"--{key}",
            type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"),
            default=None,
            help=f"Set {key} (true/false, default: {default_value})",
        )
    else:
        parser.add_argument(
            f"--{key}",
            type=type(default_value),
            default=None,
            help=f"Override for {key}",
        )
args = parser.parse_args()

NETWORK = Diffusion | HybridDiffusion | SBFlow


@error_wrap
def test_part(
    device: torch.device,
    valid_state: MetricController,
    valid_loader: DataLoader,
    network: NETWORK,
    run_dir: Path,
    save_val: bool,
    loss_model: str,
    model_type: str,
    init_time: str,
    interval: str,
) -> float:
    loss_func = get_loss_func(
        loss_model=loss_model,
    )    
    network.eval()

    mri_dir = SAVE_ROOT / "mri"
    ct_dir  = SAVE_ROOT / "ct"
    gen_dir = SAVE_ROOT / "gen"
    if save_val:
        for d in (mri_dir, ct_dir, gen_dir):
            d.mkdir(parents=True, exist_ok=True)

    img_cnt: int = 0
    for _data in valid_loader:
        if model_type in ["diffusion", "hybrid"]:
            img = _data[DataWrapper.IMG].to(device)
            cond = _data[DataWrapper.COND].to(device)
            mask = _data[DataWrapper.MASK].to(device)

            output = network(
                cond=cond,
                interval=interval,
                mode = "recon",
            )
        else:
            raise KeyError("model type not matched")

        if not (
            (isinstance(img, torch.Tensor))
            and (isinstance(cond, torch.Tensor))
            and (isinstance(output, torch.Tensor))
        ):
            raise KeyError("elements has to be tensor")

        loss = loss_func(output, img)
        loss = torch.mean(loss, dim=(1, 2, 3), keepdim=True)
        valid_state.add("loss", loss)
        valid_state.add("psnr", calculate_psnr(output, img , mask))
        valid_state.add("ssim", calculate_ssim(output, img , mask))

        batch_cnt = img.shape[0]
        if not save_val:
            img_cnt += batch_cnt
            continue

        test_dir = run_dir / "test"
        os.makedirs(test_dir, exist_ok=True)
        for i in range(batch_cnt):
            idx = img_cnt + i + 1

            mri_arr = img.cpu().detach().numpy()[i, ...]
            ct_arr  = cond.cpu().detach().numpy()[i, ...]
            gen_arr = output.cpu().detach().numpy()[i, ...]

            if mri_arr.shape[0] == 1:
                mri_arr = np.squeeze(mri_arr, axis=0)
                ct_arr  = np.squeeze(ct_arr, axis=0)
                gen_arr = np.squeeze(gen_arr, axis=0)

            affine = np.eye(4)

            nib.save(nib.Nifti1Image(mri_arr, affine), mri_dir / f"{idx:04d}_mri.nii.gz")
            nib.save(nib.Nifti1Image(ct_arr,  affine), ct_dir  / f"{idx:04d}_ct.nii.gz")
            nib.save(nib.Nifti1Image(gen_arr, affine), gen_dir / f"{idx:04d}_gen.nii.gz")

        img_cnt += batch_cnt

    log_summary(state=valid_state, log_std=True, init_time=init_time)

    primary_metric = valid_state.mean("loss")
    return primary_metric


class Tester:
    run_dir: Path
    network: NETWORK
    test_loader: DataLoader
    config: TestConfig
    modelconfig: ModelConfig
    gmmmodelconfig: GMM_ModelConfig
    def __init__(
        self,
    ) -> None:
        self.config = TestConfig()
        for key, value in vars(args).items():
            if value is not None and hasattr(self.config, key):
                if isinstance(getattr(self.config, key), bool):
                    setattr(self.config, key, bool(value))
                else:
                    setattr(self.config, key, value)

        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu

        self.config.init_time = time.time()
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dir setting
        self.run_dir = self.config.run_dir / f"{call_next_id(self.config.run_dir):05d}_test"
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        os.makedirs(self.run_dir, exist_ok=True)
        logger_add_handler(logger, f"{self.run_dir/'log.log'}", self.config.log_lv)

        # log config
        logger.info(separator())
        logger.info("Text Config")
        config_dict = asdict(self.config)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

    def __call__(
        self,
    ) -> None:
        self._set_data()
        self._set_network()
        self._test()

    @error_wrap
    def _set_data(
        self,
    ) -> None:
        logger.info(separator())
        self.test_loader, _, test_len = get_data_wrapper_loader(
            file_path=self.config.test_dataset,
            config = self.config,
            training_mode=False,
            batch=self.config.valid_batch,
            num_workers=self.config.num_workers,
            shuffle=False,
            debug_mode=self.config.debugmode,
        )
        logger.info(f"Test dataset length : {test_len}")

    @error_wrap
    def _set_network(
        self,
    ) -> None:
        checkpoint_data = torch.load(
            self.config.checkpoints,
            map_location="cpu",
            weights_only=True,
        )

        if not (("model_state_dict" in checkpoint_data) and ("model_config" in checkpoint_data)):
            for key in checkpoint_data:
                logger.warning(f"Checkpoint keys : {key}")
            logger.error("Invalid Checkpoint")
            raise KeyError("Invalid Checkpoint")

        self.modelconfig = ModelConfig(**checkpoint_data["model_config"])
        self.gmm_modelconfig = GMM_ModelConfig(**checkpoint_data["gmm_model_config"])


        self.network = get_network(
            device=self.config.device,
            model_type=self.config.model_type,
            modelconfig=self.modelconfig,
            gmmmodelconfig=self.gmmmodelconfig,
        )

        load_state_dict = checkpoint_data["model_state_dict"]

        try:
            self.network.load_state_dict(load_state_dict, strict=True)
        except Exception as err:
            logger.warning(f"Strict load failure. Trying to load weights available: {err}")
            self.network.load_state_dict(load_state_dict, strict=False)

        logger.info(separator())
        logger.info("Model Config")
        config_dict = asdict(self.modelconfig)
        logger.info("GMM Model Config")
        config_dict = asdict(self.gmmmodelconfig)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

        self.network = self.network.to(self.config.device)

    @error_wrap
    def _test(self) -> float:
        test_state = MetricController()
        test_state.reset()
        logger.info(separator())
        logger.info("Test")
        with torch.no_grad():
            test_part(
                device=self.config.device,
                valid_state=test_state,
                valid_loader=self.test_loader,
                network=self.network,
                run_dir=self.run_dir,
                save_val=True,
                loss_model=self.config.loss_model,
                model_type=self.config.model_type,
                init_time=self.config.init_time,
                interval = self.config.interval,
            )


if __name__ == "__main__":
    test = Tester()
    test()
