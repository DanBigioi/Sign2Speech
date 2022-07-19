import logging
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn import preprocessing

def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()

def read_poses_json(json_path):

    with open(json_path) as f:
        data = json.load(f)
        keypoint_array = []
        label_array = []
        for i in data["pose"]:
            keypoint_array.append(i)

        for i in data["label"]:
            label_array.append(i[1])

        # Encoding the labels so that pytorch is able to trasnform them into tensors
        label_array = np.asarray(label_array)
        le = preprocessing.LabelEncoder()
        label_array = le.fit_transform(label_array)

        keypoint_array = np.asarray(keypoint_array)
        keypoint_array = np.reshape(
            keypoint_array, (320, 63)
        )  # TODO: Why are those values hard coded here?

    return keypoint_array, label_array


def load_mfcc(filepath):
    mfcc = np.load(filepath)
    print("mfcc shape = " + str(mfcc.shape))
    return mfcc


def load_landmarks(filepath):
    landmarks = np.load(filepath)
    print("landmarks shape = " + str(landmarks.shape))
    return landmarks

def plot_pose(ax, pose):
    prev = pose[0, :]
    for row in np.ndindex(pose.shape[0]):
        cur = pose[row, :]
        if row[0] in [5, 9, 13, 17]:
            prev = pose[0, :]
        cur, prev = cur.flatten(), prev.flatten()
        x, y, z = (
            np.linspace(prev[0], cur[0], 100),
            np.linspace(prev[1], cur[1], 100),
            np.linspace(prev[2], cur[2], 100),
        )
        ax.plot(x, y, z, color="red")
        ax.text(cur[0], cur[1], cur[2], f"{row[0]}", color="red")
        prev = cur
#     scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
#     ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))


def plot_3D_hand(pose):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_pose(ax, pose.reshape(21, 3))
    plt.show()

