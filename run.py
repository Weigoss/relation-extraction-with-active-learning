import hydra
import torch
import logging
import random
import torch.nn as nn
from torch import optim
from hydra import utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from numpy import *
import models
import data_select

from utils.preprocess import preprocess
from utils.dataset import CustomDataset, collate_fn
from trainer import train, validate
from utils.util import manual_seed, load_pkl

logger = logging.getLogger(__name__)


@hydra.main(config_path='conf/config.yaml')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd
    cfg.lm_file = cwd + '/pretrained/data'
    cfg.pos_size = 2 * cfg.pos_limit + 2
    logger.info(f'\n{cfg.pretty()}')

    __Model__ = {
        'cnn': models.PCNN,
        'rnn': models.BiLSTM,
        'transformer': models.Transformer,
        'gcn': models.GCN,
        'capsule': models.Capsule,
        'lm': models.LM,
    }

    __Select__ = {
        'random': data_select.random_sample,
        'uncertainty': data_select.uncertainty_sample,
        'diversity': data_select.diversity_sample,
        'iter_diff': data_select.iter_diff_sample,
        'similarity': data_select.similarity_sample,
        'ensemble_sample_by_weight': data_select.ensemble_sample_by_weight,
        'uncertain_diverse': data_select.ensemble_sample_by_uncertain_similar,
        'uncertain_similar': data_select.ensemble_sample_by_uncertain_diverse,
        'multi_criterion_sampling': data_select.multi_criterion_sampling
    }

    # device
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')

    # 如果不修改预处理的过程，这一步最好注释掉，不用每次运行都预处理数据一次
    if cfg.preprocess:
        preprocess(cfg)

    train_data_path = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_data_path = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_data_path = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    vocab_path = os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl')

    if cfg.model_name == 'lm':
        vocab_size = None
    else:
        vocab = load_pkl(vocab_path)
        vocab_size = vocab.count
    cfg.vocab_size = vocab_size

    valid_ds = CustomDataset(valid_data_path)
    test_ds = CustomDataset(test_data_path)
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
    test_dataloader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))

    all_train_ds = load_pkl(train_data_path)
    random.shuffle(all_train_ds)

    start_size, all_size, size, per_log_num = 4000, 9800, 200, 400
    select_method = __Select__[cfg.select_method]
    pre_labels = None
    print(len(all_train_ds))
    cur_labeled_ds = all_train_ds[:start_size]
    unlabeled_ds = all_train_ds[start_size:]

    writer = SummaryWriter('tensorboard')

    logger.info('=' * 10 + ' Start training ' + '=' * 10)
    test_f1_scores, test_losses = [], []
    while len(cur_labeled_ds) <= all_size:
        model = __Model__[cfg.model_name](cfg)
        if len(cur_labeled_ds) == start_size:
            logger.info(f'\n {model}')
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.lr_factor, patience=cfg.lr_patience)
        criterion = nn.CrossEntropyLoss()

        train_dataloader = DataLoader(cur_labeled_ds, batch_size=cfg.batch_size, shuffle=True,
                                      collate_fn=collate_fn(cfg))
        train_losses, valid_losses, one_f1_scores = [], [], []
        for epoch in range(1, cfg.epoch + 1):
            # 保证随机种子每一轮不一样
            manual_seed(cfg.seed + epoch)
            train_loss = train(epoch, model, train_dataloader, optimizer, criterion, device, writer, cfg)
            # 验证集
            valid_f1, valid_loss = validate(epoch, model, valid_dataloader, criterion, device, cfg)
            # 根据valid_loss来调整学习率
            scheduler.step(valid_loss)
            # model_path = model.save(epoch, cfg)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            one_f1_scores.append(valid_f1)

        if cfg.show_plot and cfg.plot_utils == 'tensorboard' and (len(cur_labeled_ds) - start_size) % per_log_num == 0:
            # logger.info(f'one_f1_scores:{one_f1_scores}')
            for i in range(len(train_losses)):
                writer.add_scalars(f'valid_copy/valid_loss_{len(cur_labeled_ds)}', {
                    'train': train_losses[i],
                    'valid': valid_losses[i]
                }, i)
                writer.add_scalars(f'valid/valid_f1_score_{len(cur_labeled_ds)}', {
                    'valid_f1_score': one_f1_scores[i]
                }, i)

        test_f1, test_loss = validate(-1, model, test_dataloader, criterion, device, cfg)
        test_f1_scores.append(test_f1)
        test_losses.append(test_loss)
        # 最后5次迭代的平均值作为当前样本数下的f1_score表现
        # f1_scores.append(mean(one_f1_scores[-5:]))
        if len(cur_labeled_ds) == all_size:
            break

        cur_labeled_ds, unlabeled_ds = select_method(cur_labeled_ds, unlabeled_ds, size, model, device, cfg)

    if cfg.show_plot and cfg.plot_utils == 'tensorboard':
        for j in range(len(test_f1_scores)):
            writer.add_scalars('test/test_losses', {
                'test_losses': test_losses[j]
            }, j)
            writer.add_scalars('test/test_f1_scores', {
                'test_f1_scores': test_f1_scores[j],
            }, j)
        writer.close()

    # 测试集
    validate(-1, model, test_dataloader, criterion, device, cfg)


if __name__ == '__main__':
    import time

    # 查看显卡使用情况：nvidia-smi
    cur = time.time()
    main()
    logger.info(f'run time:{time.time() - cur}')
