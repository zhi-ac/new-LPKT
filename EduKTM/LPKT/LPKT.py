# coding: utf-8
# 2021/8/17 @ sone

import math
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import tqdm
import matplotlib.pyplot as plt

from EduKTM import KTM
from .LPKTNet import LPKTNet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train_one_epoch(net, optimizer, criterion, batch_size, a_data, e_data, it_data, at_data, _):
    net.train()  # 将模型设置成训练模式
    n = int(math.ceil(len(e_data) / batch_size))
    shuffled_ind = np.arange(e_data.shape[0])
    np.random.shuffle(shuffled_ind)
    e_data = e_data[shuffled_ind]
    at_data = at_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    it_data = it_data[shuffled_ind]

    pred_list = []
    target_list = []

    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()  # 清除之前批次的梯度

        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)

        pred = net(input_e, input_at, target, input_it)

        mask = input_e[:, 1:] > 0
        masked_pred = pred[:, 1:][mask]
        masked_truth = target[:, 1:][mask]

        loss = criterion(masked_pred, masked_truth).sum()

        loss.backward()
        optimizer.step()

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()

        pred_list.append(masked_pred)
        target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)


    return loss, auc, accuracy


def test_one_epoch(net, batch_size, a_data, e_data, it_data, at_data, student_ids):
    net.eval()  # 将模型设置为评估模式
    n = int(math.ceil(len(e_data) / batch_size))

    pred_list = []
    target_list = []

    student_mastery = {student_id: {skill_id: [] for skill_id in range(net.q_matrix.shape[1])} for student_id in
                       student_ids}

    for idx in tqdm.tqdm(range(n), 'Testing'):
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)
        batch_student_ids = student_ids[idx * batch_size: (idx + 1) * batch_size]

        with torch.no_grad():
            pred = net(input_e, input_at, target, input_it)
            mask = input_e[:, 1:] > 0
            masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
            masked_truth = target[:, 1:][mask].detach().cpu().numpy()

            pred_list.append(masked_pred)
            target_list.append(masked_truth)
            # 只处理有效部分的数据
            for i in range(input_e.shape[0]):  # 遍历批次中的每一个学生
                for j in range(1, input_e.shape[1]):  # 遍历每个时间步
                    if e_one_seq[i, j] == 0:  # 遇到填充数据则退出循环
                        break
                    exercise_id = int(e_one_seq[i, j])
                    # 获取技能 ID
                    skills = (net.q_matrix[exercise_id] == 1).nonzero(as_tuple=True)[0].tolist()
                    student_id = batch_student_ids[i]
                    for skill_id in skills:
                        if skill_id not in student_mastery[student_id]:
                            student_mastery[student_id][skill_id] = []
                        student_mastery[student_id][skill_id].append(pred[i, j].item())

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    # 计算每个学生每个技能的平均预测值，作为掌握情况
    for student_id in student_mastery:
        for skill_id in student_mastery[student_id]:
            if student_mastery[student_id][skill_id]:  # 如果列表不为空
                student_mastery[student_id][skill_id] = np.mean(student_mastery[student_id][skill_id])
            else:
                student_mastery[student_id][skill_id] = 0.03  # 或设置一个默认值，如0或NaN
    import json
    # 将所有的键转换为字符串
    def convert_keys_to_str(d):
        if isinstance(d, dict):
            return {str(k): convert_keys_to_str(v) for k, v in d.items()}
        return d

    student_mastery_str_keys = convert_keys_to_str(student_mastery)

    # 保存到 JSON 文件
    with open('student_mastery.json', 'w') as json_file:
        json.dump(student_mastery_str_keys, json_file)
    return loss, auc, accuracy, student_mastery


class LPKT(KTM):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout=0.2):
        super(LPKT, self).__init__()
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        self.lpkt_net = LPKTNet(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, dropout).to(device)
        self.batch_size = batch_size

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002, lr_decay_step=15, lr_decay_rate=0.5) -> ...:
        optimizer = torch.optim.Adam(self.lpkt_net.parameters(), lr=lr, eps=1e-8, betas=(0.1, 0.999), weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)
        criterion = nn.BCELoss(reduction='none')
        best_train_auc, best_test_auc = .0, .0
        for idx in range(epoch):
            train_loss, train_auc, train_accuracy  = train_one_epoch(self.lpkt_net, optimizer, criterion,
                                                                    self.batch_size, *train_data)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))
            if train_auc > best_train_auc:
                best_train_auc = train_auc

            scheduler.step()

            if test_data is not None:
                test_loss, test_auc, test_accuracy, _  = self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (idx, test_auc, test_accuracy))
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                # 可视化测试数据的技能掌握情况
                # if idx == epoch - 1 or True:  # 仅在最后一个 Epoch 可视化
                #     self.visualize_skill_mastery(test_skill_mastery)

        return best_train_auc, best_test_auc

    def eval(self, test_data) -> ...:
        self.lpkt_net.eval()
        return test_one_epoch(self.lpkt_net, self.batch_size, *test_data)

    def save(self, filepath) -> ...:
        torch.save(self.lpkt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.lpkt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
