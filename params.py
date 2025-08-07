import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Dict
import sys

import torch

# -------------------------------
# Constants for Dataset File Paths
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/fast_storage/intern1/CT_MRI_data_192_192/brain")


TRAIN_MRI_PATH = str(DATA_ROOT / "train_mri.h5")
TRAIN_CT_PATH = str(DATA_ROOT / "train_ct.h5")
TRAIN_MASK_PATH = str(DATA_ROOT / "train_mask.h5")

VALID_MRI_PATH = str(DATA_ROOT / "valid_mri.h5")
VALID_CT_PATH = str(DATA_ROOT / "valid_ct.h5")
VALID_MASK_PATH = str(DATA_ROOT / "valid_mask.h5")

TEST_MRI_PATH = str(DATA_ROOT / "test_mri.h5")
TEST_CT_PATH = str(DATA_ROOT / "test_ct.h5")
TEST_MASK_PATH = str(DATA_ROOT / "test_mask.h5")


@dataclass
class GeneralConfig:
    # Dataset
    train_mri_path: str = TRAIN_MRI_PATH
    train_ct_path: str = TRAIN_CT_PATH
    train_mask_path: str = TRAIN_MASK_PATH

    valid_mri_path: str = VALID_MRI_PATH
    valid_ct_path: str = VALID_CT_PATH
    valid_mask_path: str = VALID_MASK_PATH

    test_mri_path: str = TEST_MRI_PATH
    test_ct_path: str = TEST_CT_PATH
    test_mask_path: str = TEST_MASK_PATH

    train_dataset = (train_mri_path, train_ct_path, train_mask_path)
    valid_dataset = (valid_mri_path, valid_ct_path, valid_mask_path)
    test_dataset = (test_mri_path, test_ct_path, test_mask_path)

    debugmode: bool = False

    # data process
    ct_clip: tuple[float, float] = (-1000, 1000)
    scale: float = 1

    # Logging
    log_lv: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"

    run_dir: Path = ROOT_DIR / "log/log_2025_08_06_clip_otflow_nonclip_masked_192_192"
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["diffusion", "sbflow", "hybrid"] = "diffusion"

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adam"
    loss_model: Literal["l1", "l2", "ssim"] = "l2"

    # primary metric
    primary_metric: Literal["loss", "ssim", "psnr"] = "ssim"

    lr: float = 1e-4
    lr_decay: float = 0.92
    lr_tol: int = 4

    axes: Literal["x", "y", "z"] = "z"
    img_crop: tuple[int, int] = (192, 192)

    # Train params
    gpu: str = "0,1,2,3"
    train_batch: int = 16
    valid_batch: int = 16
    train_epoch: int = 200
    logging_density: int = 4
    valid_interval: int = 3
    valid_tol: int = 9
    num_workers: int = 4
    parallel: bool = True
    device: torch.device | None = None
    save_max_idx: int = 500

    # Hyper
    interval: int = 20

    tag: str = ""


@dataclass
class ModelConfig:

    algorithm: Literal["ddpm", "ddim", "flow", "score"] = "flow"
    sampler_type: Literal["ode", "euler", "pc"] = "pc"
    unet_input_chan: int = 2
    unet_output_chan: int = 1
    unet_chans: int = 64  # 32
    num_pool_layers: int = 3
    time_emb_dim: int = 256

    # for ddpm & ddim
    ddim_eta: float = 0.0
    beta_schedule: Literal["linear", "cosine", "sigmoid"] = "cosine"
    ddpm_target: Literal["noise", "velocity", "start"] = "noise"

    # for score
    sigma: float = 25.0

    # for flow
    sigma_min: float = 1e-4
    flow_type: Literal["vp", "ot", "rect"] = "ot"

    # for sb
    sigma_sb: float = 1e-1


@dataclass
class GMM_ModelConfig:
    num_tissues: int = 3
    weight_tissue_real_begin: float = 0
    weight_tissue_real_end: float = 0.4
    tissue_dict: Dict[str, float] = field(
        default_factory=lambda: {
            "csf": 15.0,  # mandatory
            "white_matter": 25.0,  # mandatory
            "gray_matter": 37.0,  # mandatory
        }
    )

    std_dict: Dict[str, float] = field(
        default_factory=lambda: {
            "csf": 10.0,  # mandatory
            "white_matter": 10.0,  # mandatory
            "gray_matter": 10.0,  # mandatory
        }
    )
    lambda_schedule_step: int = 100 * 1300
    lambda_scheduler_type: Literal["linear", "cosine", "constant"] = "cosine"
    adaptive_metric_type: Literal["entropy", "confidence", None] = "entropy"


@dataclass
class MRSignal_ModelConfig:
    TR: int = 2400
    # TE: int = 3
    TI: int = 1000
    sequence: Literal["mprage", "tfe"] = "mprage"
    # modality: Literal["T1w", "T2w", "Flair", "PD"] = "T1w"


# Argparser
parser = argparse.ArgumentParser(description="Training Configuration")
general_config_dict = asdict(GeneralConfig())
model_config_dict = asdict(ModelConfig())

for key, default_value in general_config_dict.items():
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

for key, default_value in model_config_dict.items():
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

# Apply argparser
config = GeneralConfig()
modelconfig = ModelConfig()
gmm_modelconfig = GMM_ModelConfig()
mrsignal_modelconfig = MRSignal_ModelConfig()


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        else:
            return False  # Other type (terminal, etc.)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    sys.argv = [""]

args = parser.parse_args()

for key, value in vars(args).items():
    if value is not None:
        if hasattr(config, key):
            if isinstance(getattr(config, key), bool):
                setattr(config, key, bool(value))
            else:
                setattr(config, key, value)

        if hasattr(modelconfig, key):
            if isinstance(getattr(modelconfig, key), bool):
                setattr(modelconfig, key, bool(value))
            else:
                setattr(modelconfig, key, value)
