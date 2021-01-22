import logging
import random
import pickle
import numpy as np
import torch

logger = logging.getLogger(__name__)


def manual_seed(num: int = 1) -> None:
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)


Path = str


def load_pkl(fp, verbose=True):
    if verbose:
        logger.info(f'load data from {fp}')
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data
