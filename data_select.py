import math

import numpy as np
import random
import torch
import logging
import heapq

from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from torch import nn
from sklearn.cluster import KMeans, DBSCAN
from dataset import collate_fn
from models.LM import LMFcExtractor
from models.PCNN import CNNFcExtractor
from models.BiLSTM import BiLSTMFcExtractor
from utils import load_pkl
from text2vec import Similarity
from text2vec.algorithm.distance import cosine_distance
from itertools import groupby
from sklearn.decomposition import PCA
from collections import defaultdict

logger = logging.getLogger(__name__)


# random_seed = 5
# np.random.seed(random_seed)
# random.seed(random_seed)

def get_divided_by_select(cur_labeled_ds, unlabeled_ds, select):
    logger.info(f'select index:{select}')
    new_unlabeled_ds = []
    for index, sen in enumerate(unlabeled_ds):
        if index in select:
            cur_labeled_ds.append(sen)
        else:
            new_unlabeled_ds.append(sen)
    return cur_labeled_ds, new_unlabeled_ds


def random_sample(cur_labeled_ds, unlabeled_ds, per_select_size, model, device, cfg):
    # 不重复地选size个样本做标记
    select = random.sample(range(0, len(unlabeled_ds)), per_select_size)
    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)


def iter_diff_sample(cur_labeled_ds, unlabeled_ds, per_select_size, pre_labels, model, device, cfg):
    model.eval()
    all_y_pred = np.empty((0, cfg.num_relations))
    # TODO 这边是不是能全部加载处理，而不是一个一个构造dataloader
    for index, one in enumerate(unlabeled_ds):
        one_dataloader = DataLoader([one], collate_fn=collate_fn(cfg))
        (x, y) = next(iter(one_dataloader))
        for key, value in x.items():
            x[key] = value.to(device)
        with torch.no_grad():
            y_pred = model(x)
        y_pred = y_pred.cpu().detach().numpy()
        all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)

    all_y_pred = torch.from_numpy(all_y_pred)
    all_y_pred_probability = nn.functional.softmax(all_y_pred, dim=1).numpy()
    cur_labels = np.argmax(all_y_pred_probability, axis=1).tolist()
    select = None
    if not pre_labels:
        select = random.sample(range(0, len(unlabeled_ds)), per_select_size)
    else:
        select = set()
        print('select num:', end='\t')
        n = len(unlabeled_ds)
        for i in range(n):
            if cur_labels[i] != pre_labels[i]:
                select.add(i)
            if len(select) >= per_select_size:
                break
        else:
            print('只有' + str(len(select)) + '个先后不一致的样本')
            for j in range(n):
                if len(select) < per_select_size:
                    select.add(j)
                else:
                    break
            else:
                logger.info('数组添加个数出错了')
        print('成功挑选300个样本')

        # while len(select) < per_select_size:
        #     tmp = random.sample(range(0, len(unlabeled_ds)), per_select_size - len(select))
        #     for i in tmp:
        #         if cur_labels[i] != pre_labels[i]:
        #             select.add(i)
        #     print(len(select), end='\t')
        # print('select all num:', len(select))

    logger.info(f'select index:{select}')
    new_unlabeled_ds, new_pre_label = [], []
    for index, sen in enumerate(unlabeled_ds):
        if index in select:
            cur_labeled_ds.append(sen)
        else:
            new_unlabeled_ds.append(sen)
            new_pre_label.append(cur_labels[index])

    return cur_labeled_ds, new_unlabeled_ds, new_pre_label


def uncertainty_sample(cur_labeled_ds, unlabeled_ds, per_select_size, model, device, cfg):
    model.eval()
    all_y_pred = np.empty((0, cfg.num_relations))
    for index, one in enumerate(unlabeled_ds):
        one_dataloader = DataLoader([one], collate_fn=collate_fn(cfg))
        (x, y) = next(iter(one_dataloader))
        for key, value in x.items():
            x[key] = value.to(device)
        with torch.no_grad():
            y_pred = model(x)
        y_pred = y_pred.cpu().detach().numpy()
        all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)

    all_y_pred = torch.from_numpy(all_y_pred)
    all_y_pred_probability = nn.functional.softmax(all_y_pred, dim=1).numpy()
    type = cfg.concrete
    select = None
    if type == 'least_confident':
        # 当前模型预测的最有可能的标签概率是最小的
        tmp = all_y_pred_probability.max(axis=1)
        select = heapq.nsmallest(per_select_size, range(len(tmp)), tmp.take)
    elif type == 'margin_sampling':
        res = np.empty(0)
        ttmp = np.vsplit(all_y_pred_probability, all_y_pred_probability.shape[0])
        for tmp in ttmp:
            tmp = np.squeeze(tmp)
            # 取最大的两个数
            first_two = tmp[np.argpartition(tmp, -2)[-2:]]
            # 收集最大两个数的差值
            res = np.concatenate((res, np.array([abs(first_two[0] - first_two[1])])), axis=0)
        select = heapq.nsmallest(per_select_size, range(len(res)), res.take)
    elif type == 'entropy_sampling':
        res = np.empty(0)
        ttmp = np.vsplit(all_y_pred_probability, all_y_pred_probability.shape[0])
        for tmp in ttmp:
            tmp = np.squeeze(tmp)
            # .dot方法好像可以自动转置再求点积
            res = np.concatenate((res, np.array([tmp.dot(np.log2(tmp))])), axis=0)
        select = heapq.nsmallest(per_select_size, range(len(res)), res.take)
    else:
        assert ('uncertainty concrete choose error')

    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)


def diversity_sample(cur_labeled_ds, unlabeled_ds, per_select_size, model, device, cfg):
    # 这个方法根据bert重写
    if cfg.model_name == 'cnn':
        extractor = CNNFcExtractor(model)
        features = np.empty((0, 80))
    elif cfg.model_name == 'lm':
        extractor = LMFcExtractor(model)
        features = np.empty((0, 100))
    else:
        assert ('diversity model choose error')

    for index, one in enumerate(unlabeled_ds):
        one_dataloader = DataLoader([one], collate_fn=collate_fn(cfg))
        (x, y) = next(iter(one_dataloader))
        for key, value in x.items():
            x[key] = value.to(device)
        with torch.no_grad():
            feature = extractor(x).cpu().detach()
            features = np.concatenate((features, np.array(feature)), axis=0)
            # features = np.concatenate((features, np.array([feature])), axis=0)
    # 按除最大值
    # min = np.amin(features)
    # max = np.amax(features)
    # features = (features - min) / (max - min)
    # 按均值和标准差
    mu = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mu) / std
    type = cfg.concrete
    select = set()

    if type == 'per_select_num':
        if len(unlabeled_ds) <= per_select_size:
            select.update([i for i in range(per_select_size)])
        else:
            estimator = KMeans(n_clusters=per_select_size)
            estimator.fit(features)
            centers = estimator.cluster_centers_
            label_pred = estimator.labels_

            logger.info(f'kmeans 迭代次数与总距离:{estimator.n_iter_, estimator.inertia_}')
            logger.info("轮廓系数: %0.3f" % silhouette_score(features, estimator.labels_))  # 轮廓系数评价聚类的好坏
            lst = list(zip(features, label_pred, range(len(unlabeled_ds))))
            lst.sort(key=lambda x: x[1])

            # 先按label_pred分组，再在不同组里选择最靠近聚类中心的点
            for pred_label, items in groupby(lst, key=lambda x: x[1]):
                min_dist, min_index = 1e9, -1
                for item in items:
                    dist = np.sum((np.array(item[0]) - centers[pred_label]) ** 2)
                    if dist < min_dist:
                        min_dist, min_index = dist, item[2]
                select.add(min_index)
    elif type == 'closest_large_cluster':
        # 先做特征降维吗，考虑到DBCSAN的eps难以确定,特征减少利于调参

        pca = PCA(n_components=10)
        features = pca.fit_transform(features)
        # print('各特征权重占比：',pca.explained_variance_ratio_)
        estimator = DBSCAN(eps=1.5, min_samples=3).fit(features)
        label_pred = estimator.labels_
        # 核心样本数不一定能挑满270（存在某类簇只有3个样本，但是要采4个的情况），剩下不足的都用噪声样本来补充
        core_nums = per_select_size * 9 // 10
        n_clusters_ = len(set(label_pred)) - (1 if -1 in label_pred else 0)  # 获取分簇的数目
        dic = {i: len(label_pred[label_pred == i]) for i in range(n_clusters_)}
        dic = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
        # print(dic)
        a, b = divmod(core_nums, n_clusters_)
        # print(n_clusters_, a, b)
        # 给出核心样本中每个类簇所需采样个数
        dic2 = {i: a + (1 if idx < b else 0) for idx, i in enumerate(dic.keys())}
        # print(dic2)
        lst = list(zip(features, label_pred, range(len(unlabeled_ds))))
        lst.sort(key=lambda x: x[1])
        for pre_label, items in groupby(lst, key=lambda x: x[1]):
            items = list(items)
            if pre_label != -1:
                estimator2 = KMeans(n_clusters=1).fit([item[0] for item in items])
                center = estimator2.cluster_centers_[0]
                tmp = ((np.sum((item[0] - center) ** 2), item[2]) for item in items)
                # print(heapq.nsmallest(dic2[pre_label], tmp, key=lambda x: x[0]))
                select.extend((index for _, index in heapq.nsmallest(dic2[pre_label], tmp, key=lambda x: x[0])))
                # select.extend([index for _, index in heapq.nsmallest(dic2[pre_label], tmp, key=lambda x: x[0])])
            else:
                ratio_data = items
                # all_ratio_cnts = len(label_pred[label_pred == -1])
                # select.extend([items[i][2] for i in random.sample(range(0, all_ratio_cnts), ratio_nums)])
            # print(len(select), end='\t')
        print(len(select))
        if len(select) < per_select_size:
            all_ratio_cnts = len(label_pred[label_pred == -1])
            select.extend(
                (ratio_data[i][2] for i in random.sample(range(0, all_ratio_cnts), per_select_size - len(select))))
        print('样本数：', len(select))
        # 按9：1的比例分配，正常样本点（270）和噪音（30）,先按类簇数排序
        # 类簇数（100） < 正常点个数（270），每个类簇取3个点，截断超出的部分
        # 类簇数（300） > 正常点个数（270），取前270个就好
        # 噪音点中随机取30个
        # print("轮廓系数: %0.3f" % silhouette_score(features, labels))  # 轮廓系数评价聚类的好坏

    elif type == 'relations_num':
        size = cfg.num_relations - 1
        estimator = KMeans(n_clusters=size).fit(features)
        centers = estimator.cluster_centers_
        logger.info(f'kmeans 迭代次数与总距离:{estimator.n_iter_, estimator.inertia_}')
        logger.info("轮廓系数: %0.3f" % silhouette_score(features, estimator.labels_))  # 轮廓系数评价聚类的好坏

        lst = list(zip(features, estimator.labels_, range(len(unlabeled_ds))))
        lst.sort(key=lambda x: x[1])
        assert per_select_size % size == 0, '每次选取样本个数应为关系种类的整数倍'

        # 先按label_pred分组，再在不同组里选择最靠近聚类中心的点
        for pred_label, items in groupby(lst, key=lambda x: x[1]):
            tmp = ((np.sum((np.array(item[0]) - centers[pred_label]) ** 2), item[2]) for item in items)
            select.update([index for _, index in heapq.nsmallest(per_select_size // size, tmp, key=lambda x: x[1])])
            # select.extend((index for _, index in heapq.nsmallest(per_select_size // size, tmp, key=lambda x: x[1])))

    else:
        assert ('uncertainty concrete choose error')

    n = len(unlabeled_ds)
    while len(select) < per_select_size:
        idx = random.randint(0, n - 1)
        select.add(idx)
    if len(select) == per_select_size:
        print(f'当前挑选满{per_select_size}个')

    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)


sim = Similarity()


def similarity_sample(cur_labeled_ds, unlabeled_ds, per_select_size, model, device, cfg):
    # 这个200是tecent预训练集中的维度
    all_emb = np.empty((0, 200))
    for one in cur_labeled_ds:
        emb = sim.encode(one['sentence'])
        all_emb = np.concatenate((all_emb, emb[np.newaxis, :]), axis=0)
    all_emb = np.mean(all_emb, axis=0)
    scores = ((idx, cosine_distance(all_emb, sim.encode(one['sentence']))) for idx, one in enumerate(unlabeled_ds))
    select = [idx for idx, _ in heapq.nsmallest(per_select_size, scores, key=lambda x: x[1])]
    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)


def ensemble_sample_by_weight(cur_labeled_ds, unlabeled_ds, per_select_size, model, device, cfg):
    dic = {}
    all_emb = np.empty((0, 200))
    for one in cur_labeled_ds:
        emb = sim.encode(one['sentence'])
        all_emb = np.concatenate((all_emb, emb[np.newaxis, :]), axis=0)
    all_emb = np.mean(all_emb, axis=0)
    scores = ((idx, cosine_distance(all_emb, sim.encode(one['sentence']))) for idx, one in enumerate(unlabeled_ds))
    for index, (idx, _) in enumerate(heapq.nsmallest(per_select_size, scores, key=lambda x: x[1])):
        # 考虑到序号也有一定的信息
        # dic[idx] = 6 * (300 - index)
        dic[idx] = 300 - index
    logger.info(f'after select1:{dic}')
    # --------

    # extractor = LMFcExtractor(model)
    extractor = CNNFcExtractor(model)
    extractor = BiLSTMFcExtractor(model)
    extractor.eval()
    model.eval()
    # features, all_y_pred = np.empty((0, 100)), np.empty((0, cfg.num_relations))
    # features, all_y_pred = np.empty((0, 80)), np.empty((0, cfg.num_relations))
    features, all_y_pred = np.empty((0, 150)), np.empty((0, cfg.num_relations))
    for one in unlabeled_ds:
        one_dataloader = DataLoader([one], collate_fn=collate_fn(cfg))
        (x, y) = next(iter(one_dataloader))
        for key, value in x.items():
            x[key] = value.to(device)
        with torch.no_grad():
            feature = extractor(x).cpu().detach().numpy()
            features = np.concatenate((features, feature), axis=0)
            y_pred = model(x).cpu().detach().numpy()
            all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)

    # 采用entropy的方法
    all_y_pred = torch.from_numpy(all_y_pred)
    all_y_pred = nn.functional.softmax(all_y_pred, dim=1).numpy()
    res = np.empty(0)
    ttmp = np.vsplit(all_y_pred, all_y_pred.shape[0])
    for tmp in ttmp:
        tmp = np.squeeze(tmp)
        # .dot方法好像可以自动转置再求点积
        res = np.concatenate((res, np.array([tmp.dot(np.log2(tmp))])), axis=0)

    for index, idx in enumerate(heapq.nsmallest(per_select_size, range(len(res)), res.take)):
        # dic[idx] = 12 * (300 - index) if idx not in dic else dic[idx] + 12 * (300 - index)
        dic[idx] = 3 * (300 - index) if idx not in dic else dic[idx] + 3 * (300 - index)
        # dic[idx] = 300 - index if idx not in dic else dic[idx] + 300 - index
    logger.info(f'after select2:{dic}')
    # ---------
    size = cfg.num_relations - 1
    estimator = KMeans(n_clusters=size).fit(features)
    centers = estimator.cluster_centers_

    lst = list(zip(features, estimator.labels_, range(len(unlabeled_ds))))
    lst.sort(key=lambda x: x[1])
    assert per_select_size % size == 0, '每次选取样本个数应为关系种类的整数倍'
    # 先按label_pred分组，再在不同组里选择最靠近聚类中心的点
    # i = 0
    for pred_label, items in groupby(lst, key=lambda x: x[1]):
        tmp = ((np.sum((np.array(item[0]) - centers[pred_label]) ** 2), item[2]) for item in items)
        for index, (_, idx) in enumerate(heapq.nsmallest(per_select_size // size, tmp, key=lambda x: x[1])):
            # dic[idx] = 5 * (300 - index * 50) if idx not in dic else dic[idx] + 5 * (300 - index * 50)
            # dic[idx] = 5 * (300 - i - index * 50) if idx not in dic else dic[idx] + 5 * (300 - i - index * 50)
            dic[idx] = 2 * (300 - index * 50) if idx not in dic else dic[idx] + 2 * (300 - index * 50)
        # i += 1
    logger.info(f'after select3:{dic}')
    # print(dic)
    # logger.info(heapq.nlargest(per_select_size, dic.items(), key=lambda x: x[1]))
    # print(heapq.nlargest(per_select_size, dic.items(), key=lambda x: x[1]))
    select = [index for index, _ in heapq.nlargest(per_select_size, dic.items(), key=lambda x: x[1])]

    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)


def all_uncertainty_sampling(unlabeled_ds, per_select_size, model, device, cfg):
    model.eval()
    all_y_pred = np.empty((0, cfg.num_relations))
    for one in unlabeled_ds:
        one_dataloader = DataLoader([one], collate_fn=collate_fn(cfg))
        (x, y) = next(iter(one_dataloader))
        for key, value in x.items():
            x[key] = value.to(device)
        with torch.no_grad():
            y_pred = model(x).cpu().detach().numpy()
            all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)

    all_y_pred = torch.from_numpy(all_y_pred)
    all_y_pred = nn.functional.softmax(all_y_pred, dim=1).numpy()

    res1 = all_y_pred.max(axis=1)
    select = set((idx for idx in heapq.nsmallest(per_select_size, range(len(res1)), res1.take)))

    res3, res2 = np.empty(0), np.empty(0)
    ttmp = np.vsplit(all_y_pred, all_y_pred.shape[0])
    for tmp in ttmp:
        tmp = np.squeeze(tmp)
        res2 = np.concatenate((res2, np.array([tmp.dot(np.log2(tmp))])), axis=0)
        first_two = tmp[np.argpartition(tmp, -2)[-2:]]
        res3 = np.concatenate((res3, np.array([abs(first_two[0] - first_two[1])])), axis=0)

    select.update((idx for idx in heapq.nsmallest(per_select_size, range(len(res3)), res3.take)))
    select.update((idx for idx in heapq.nsmallest(per_select_size, range(len(res2)), res2.take)))
    return select


def multi_criterion_sampling(cur_labeled_ds, unlabeled_ds, per_select_size, model, device, cfg):
    pre_select = all_uncertainty_sampling(unlabeled_ds, per_select_size * 3, model, device, cfg)
    pre_select = list(pre_select)
    n = len(pre_select)
    # print(pre_select)
    logger.info(f'all_uncertainty挑选的样本数：{len(pre_select)}')

    extractor = LMFcExtractor(model)
    extractor.eval()
    features = np.empty((0, 100))
    for i in pre_select:
        one_dataloader = DataLoader([unlabeled_ds[i]], collate_fn=collate_fn(cfg))
        (x, y) = next(iter(one_dataloader))
        for key, value in x.items():
            x[key] = value.to(device)
        with torch.no_grad():
            feature = extractor(x).cpu().detach().numpy()
            features = np.concatenate((features, feature), axis=0)

    size = cfg.num_relations - 1
    estimator = KMeans(n_clusters=size).fit(features)
    # estimator = KMeans(n_clusters=per_select_size).fit(features)
    centers = estimator.cluster_centers_

    # lst = list(zip(features, estimator.labels_, range(n)))
    lst = list(zip(estimator.labels_, pre_select))
    lst.sort(key=lambda x: x[0])
    assert per_select_size % size == 0, '每次选取样本个数应为关系种类的整数倍'
    # 改成relation_nums 的聚类方法
    select = list()
    candidate = []
    for pred_label, items in groupby(lst, key=lambda x: x[0]):
        items = list(items)
        cnt = len(items)
        cosine_dis = [[0 for _ in range(cnt)] for _ in range(cnt)]
        for i in range(cnt):
            for j in range(i + 1, cnt):
                cosine_dis[i][j] = cosine_dis[j][i] = cosine_distance(sim.encode(unlabeled_ds[items[i][1]]['sentence']),
                                                                      sim.encode(unlabeled_ds[items[j][1]]['sentence']))

        tmp = []
        for i in range(cnt):
            entropy = 0
            for j in range(cnt):
                entropy += cosine_dis[i][j] * math.log(cosine_dis[i][j] + 1)
            tmp.append((entropy, items[i][1]))
        ttmp = [index for _, index in heapq.nlargest(round(per_select_size * cnt / n) + 1, tmp, key=lambda x: x[0])]
        # tmp = ((cosine_dis[i][j] * math.log(cosine_dis[i][j] + 1), items[i][1]) for j in range(cnt) for i in range(cnt))
        # ac_num = math.ceil(per_select_size * cnt / n)
        # ttmp = [index for _, index in heapq.nlargest(cnt, tmp, key=lambda x: x[0])]
        select.extend(ttmp[:-1])
        candidate.append(ttmp[-1])
    print(candidate)

    # for i in range(len(candidate)):
    #     ac_num = math.ceil(per_select_size * len(candidate[i]) / n)
    #     select.extend(candidate[i][:ac_num])
    #     candidate[i] = candidate[i][ac_num:]
    # candidate.append(ttmp[ac_num:])
    # select.update(ttmp[:ac_num+1])
    # print(cnt, math.ceil(per_select_size * cnt / n))
    # print(candidate)

    # print(f'当前候选集样本数：len(candidate)')
    print(f'当前挑选{len(select)}个')
    # i = 0
    while len(select) < per_select_size:
        i = random.randint(0, len(candidate) - 1)
        select.append(candidate[i])
        candidate.pop(i)
        # if candidate[i]:
        #     select.add(candidate[i][0])
        #     candidate[i].pop(0)

        # select.add(candidate[idx])
    print(candidate)
    select = select[:200]
    if len(select) == per_select_size:
        print(f'当前挑选满{per_select_size}个')

    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)


def ensemble_sample_by_uncertain_similar(cur_labeled_ds, unlabeled_ds, per_select_size, model, device, cfg):
    pre_select = all_uncertainty_sampling(unlabeled_ds, per_select_size, model, device, cfg)
    logger.info(f'all_uncertainty挑选的样本数：{len(pre_select)}')
    # 这个200是tecent预训练集中的维度
    all_emb = np.empty((0, 200))
    for one in cur_labeled_ds:
        emb = sim.encode(one['sentence'])
        all_emb = np.concatenate((all_emb, emb[np.newaxis, :]), axis=0)
    all_emb = np.mean(all_emb, axis=0)

    scores = ((i, cosine_distance(all_emb, sim.encode(unlabeled_ds[i]['sentence']))) for i in pre_select)
    select = [idx for idx, _ in heapq.nsmallest(per_select_size, scores, key=lambda x: x[1])]
    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)


def ensemble_sample_by_uncertain_diverse(cur_labeled_ds, unlabeled_ds, per_select_size, model, device, cfg):
    pre_select = all_uncertainty_sampling(unlabeled_ds, per_select_size, model, device, cfg)
    pre_select = list(pre_select)
    n = len(pre_select)
    select = set()
    extractor = LMFcExtractor(model)
    extractor.eval()
    features = np.empty((0, 100))
    for i in pre_select:
        one_dataloader = DataLoader([unlabeled_ds[i]], collate_fn=collate_fn(cfg))
        (x, y) = next(iter(one_dataloader))
        for key, value in x.items():
            x[key] = value.to(device)
        with torch.no_grad():
            feature = extractor(x).cpu().detach().numpy()
            features = np.concatenate((features, feature), axis=0)

    size = cfg.num_relations - 1
    estimator = KMeans(n_clusters=size).fit(features)
    # estimator = KMeans(n_clusters=per_select_size).fit(features)
    centers = estimator.cluster_centers_

    lst = list(zip(features, estimator.labels_, pre_select))
    lst.sort(key=lambda x: x[1])
    assert per_select_size % size == 0, '每次选取样本个数应为关系种类的整数倍'
    # 改成relation_nums 的聚类方法
    for pred_label, items in groupby(lst, key=lambda x: x[1]):
        tmp = ((np.sum((np.array(item[0]) - centers[pred_label]) ** 2), item[2]) for item in items)
        select.update([index for _, index in heapq.nsmallest(per_select_size // size, tmp, key=lambda x: x[1])])
        # select.update([index for _, index in heapq.nsmallest(1, tmp, key=lambda x: x[1])])
    # pre_select = list(pre_select)
    while len(select) < per_select_size:
        idx = random.randint(0, n - 1)
        select.add(pre_select[idx])
    if len(select) == per_select_size:
        print(f'当前挑选满{per_select_size}个')

    return get_divided_by_select(cur_labeled_ds, unlabeled_ds, select)


if __name__ == '__main__':
    import time

    train_data_path = 'data/out/train.pkl'
    all_train_ds = load_pkl(train_data_path)
    random.shuffle(all_train_ds)
    cur_labeled_ds = all_train_ds[:500]
    unlabeled_ds = all_train_ds[500:]
    cur = time.time()
    # cur_labeled_ds, unlabeled_ds = random_sample(cur_labeled_ds, unlabeled_ds, 2000)
    # print(time.time() - cur)
    # cur_labeled_ds = all_train_ds[:500]
    # unlabeled_ds = all_train_ds[500:]
    # cur = time.time()
    # cur_labeled_ds, unlabeled_ds = random_sample2(cur_labeled_ds, unlabeled_ds, 2000)
    # print(time.time() - cur)
