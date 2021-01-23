import csv
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


def load_pkl(fp, verbose=True):
    if verbose:
        logger.info(f'load data from {fp}')
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data, fp, verbose: bool = True):
    if verbose:
        logger.info(f'save data in {fp}')

    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def load_csv(fp, is_tsv=False, verbose=True):
    if verbose:
        logger.info(f'load csv from {fp}')

    dialect = 'excel-tab' if is_tsv else 'excel'
    with open(fp, encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)


def save_csv(data, fp, save_in_tsv, write_head=True, verbose=True):
    if verbose:
        logger.info(f'save csv file in: {fp}')

    with open(fp, 'w', encoding='utf-8') as f:
        fieldnames = data[0].keys()
        dialect = 'excel-tab' if save_in_tsv else 'excel'
        writer = csv.DictWriter(f, fieldnames=fieldnames, dialect=dialect)
        if write_head:
            writer.writeheader()
        writer.writerows(data)


def seq_len_to_mask(seq_len, max_len=None, mask_pos_to_true=True):
    """
    将一个表示sequence length的一维数组转换为二维的mask，默认pad的位置为1。
    转变 1-d seq_len到2-d mask.

    :param list, np.ndarray, torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, logger.error(f"seq_len can only have one dimension, got {seq_len.dim()} != 1.")
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len.device)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise logger.error("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


def to_one_hot(x: torch.Tensor, length: int) -> torch.Tensor:
    """
    :param x:        [B]    一般是 target 的值
    :param length:    L     一般是关系种类树
    :return:         [B, L]  每一行，只有对应位置为1，其余为0
    """
    B = x.size(0)
    x_one_hot = torch.zeros(B, length)
    for i in range(B):
        x_one_hot[i, x[i]] = 1.0

    return x_one_hot.to(device=x.device)
