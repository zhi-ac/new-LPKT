# coding: utf-8
# 2021/8/17 @ sone

import torch
from torch import nn

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# 总结
# LPKTNet 是一个用于学生知识追踪的神经网络模型。
# 它通过嵌入层将回答时间、间隔时间、练习题等特征转换为嵌入向量，
# 然后通过多层线性变换和激活函数计算学生的学习增益、遗忘参数，并更新学生的知识掌握状态。
# 最终，它返回学生在每个时间步的预测结果和知识掌握情况。
class LPKTNet(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, dropout=0.2):
        super(LPKTNet, self).__init__()
        # n_at, n_it, n_exercise, n_question：分别表示回答时间的种类数、间隔时间的种类数、练习题的数量、技能的数量。
        self.d_k = d_k  # 隐藏层特征维度
        self.d_a = d_a  # 回答特征维度
        self.d_e = d_e  # 练习题特征维度
        self.q_matrix = q_matrix  # 题目-技能矩阵
        self.n_question = n_question  # 技能的数量

        # 这里定义了三个嵌入层，并初始化它们的权重：
        #
        # at_embed: 回答时间嵌入层
        # it_embed: 间隔时间嵌入层
        # e_embed: 练习题嵌入层
        self.at_embed = nn.Embedding(n_at + 10, d_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(n_it + 10, d_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(n_exercise + 10, d_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)


        # 这些线性层用于对输入特征进行变换：
        #
        # linear_1 到 linear_5: 分别用于不同的计算过程中的线性变换。
        self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)

        # 定义了激活函数 Tanh 和 Sigmoid，以及 Dropout 层。
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)


    def forward(self, e_data, at_data, a_data, it_data, isTest=True):  # 在前向传播过程中，模型会依次处理每个时间步的数据，计算预测结果和学生的技能掌握情况。

        # 嵌入层转换： 将输入数据 e_data, at_data, it_data 分别通过嵌入层转换为嵌入向量。  e_data 题目序列, at_data 回答时间序列, a_data 答案序列, it_data 间隔时间序列
        # 初始化 h_pre： 初始化隐藏状态 h_pre。
        # 特征组合： 将练习题嵌入、回答时间嵌入、答案数据组合，通过 linear_1 线性变换。
        batch_size, seq_len = e_data.size(0), e_data.size(1)
        e_embed_data = self.e_embed(e_data)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question + 1, self.d_k)).repeat(batch_size, 1, 1).to(device)
        h_tilde_pre = None
        all_learning = self.linear_1(torch.cat((e_embed_data, at_embed_data, a_data), 2))
        learning_pre = torch.zeros(batch_size, self.d_k).to(device)

        pred = torch.zeros(batch_size, seq_len).to(device)


        # 特征提取和变换： 提取当前时间步的练习题、回答时间、答案等嵌入特征，并进行线性变换和激活。
        # 计算学习增益和遗忘参数： 计算学习增益 LG 和遗忘参数 gamma_f。
        # 更新隐藏状态 h： 结合当前隐藏状态和学习增益更新隐藏状态 h。
        # 计算预测结果 y： 计算下一时间步的预测结果 y。
        # 存储知识掌握信息： 将当前隐藏状态存储到 skill_mastery 列表中。
        for t in range(0, seq_len - 1):
            e = e_data[:, t]
            q_e = self.q_matrix[e].view(batch_size, 1, -1)  # q_e表示问题e所对应的技能矩阵
            it = it_embed_data[:, t]

            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)
            learning = all_learning[:, t]
            learning_gain = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            learning_gain = self.tanh(learning_gain)
            gamma_l = self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            n_skill = LG_tilde.size(1)
            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                it.repeat(1, n_skill).view(batch_size, -1, self.d_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre

            h_tilde = self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k)
            y = self.sig(self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y


            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde


        return pred
