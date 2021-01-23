import torch
import logging
from utils.metrics import PRMetric

logger = logging.getLogger(__name__)


def train(epoch, model, dataloader, optimizer, criterion, device, writer, cfg):
    model.train()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            # 让键指向放入device设备后的value
            x[key] = value.to(device)
        y = y.to(device)
        # 清零上次迭代的梯度
        optimizer.zero_grad()
        y_pred = model(x)

        if cfg.model_name == 'capsule':
            loss = model.loss(y_pred, y)
        else:
            loss = criterion(y_pred, y)

        loss.backward()
        # 更新模型参数
        optimizer.step()

        metric.update(y_true=y, y_pred=y_pred)
        losses.append(loss.item())
        # TODO batch_size * 10 正确吗？
        data_total = len(dataloader.dataset)
        data_cal = data_total if batch_idx == len(dataloader) else batch_idx * len(y)
        if (cfg.train_log and batch_idx % cfg.log_interval == 0) or batch_idx == len(dataloader):
            # p r f1 皆为 macro，因为micro时三者相同，定义为acc
            acc, p, r, f1 = metric.compute()
            logger.info(f'Train Epoch {epoch}: [{data_cal}/{data_total}]({100. * data_cal / data_total:.0f}%)\t'
                        f'Loss: {loss.item():.6f}\t metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    if cfg.show_plot and not cfg.only_comparison_plot and cfg.plot_utils == 'tensorboard':
        for i in range(len(losses)):
            writer.add_scalar(f'epoch_{epoch}_training_loss', losses[i], i)

    return losses[-1]


def validate(epoch, model, dataloader, criterion, device, cfg):
    model.eval()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(x)

            if cfg.model_name == 'capsule':
                loss = model.loss(y_pred, y)
            else:
                loss = criterion(y_pred, y)

            metric.update(y_true=y, y_pred=y_pred)
            losses.append(loss.item())

    loss = sum(losses) / len(losses)
    acc, p, r, f1 = metric.compute()
    data_total = len(dataloader.dataset)

    if epoch >= 0:
        logger.info(f'Valid Epoch {epoch}: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}\t'
                    f'metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')
    else:
        logger.info(f'Test Data: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}\t'
                    f'metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    return f1, loss
