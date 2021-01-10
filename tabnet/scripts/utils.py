import os
import sys
import random
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def init_logger(directory, log_file_name):
    formatter = logging.Formatter('\n%(asctime)s\t%(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_path = Path(directory, log_file_name)
    if not log_path.parent.exists():
        log_path.parent.mkdir(exist_ok=True, parents=True)
    handler = logging.FileHandler(filename=log_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def init_tb_logger(directory, log_file_name):
    log_path = Path(directory, log_file_name)
    tb_logger = SummaryWriter(log_dir=log_path)
    return tb_logger