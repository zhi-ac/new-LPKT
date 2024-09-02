import logging
import numpy as np
from load_data import DATA
from EduKTM import LPKT

pathname = 'anonymized_full_release_competition_dataset'
# pathname = 'lianxi'

# def generate_q_matrix(path, n_skill, n_problem, gamma=0.0):
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f:
#             problem2skill = eval(line)
#     q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
#     for p in problem2skill.keys():
#         q_matrix[p][problem2skill[p]] = 1
#     return q_matrix

# 修改代码能够适应一个问题对应多个技能
def generate_q_matrix(path, n_skill, n_problem, gamma=0.0):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            problem2skill = eval(line)
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p, skills in problem2skill.items():
        for skill in skills:
            q_matrix[p][skill] = 1
    return q_matrix

batch_size = 32
n_at = 1326
n_it = 2839
n_question = 102
n_exercise = 3162
seqlen = 1000  # 可以通过调整这一个的大小，让每个学生的做题序列被拆分成更少的batch
# 数据集的最大序列长度
# 2742  train
# 2054  valid
# 3057  test
d_k = 128
d_a = 50
d_e = 128
q_gamma = 0.03
dropout = 0.2

# batch_size = 32
# n_at = 334
# n_it = 291
# n_question = 10
# n_exercise = 50
# seqlen = 10
# d_k = 128
# d_a = 50
# d_e = 128
# q_gamma = 0.03
# dropout = 0.2

# 第0行和第0列没有元素
q_matrix = generate_q_matrix(
    '../../data/' + pathname + '/problem2skill',
    n_question, n_exercise,
    q_gamma
)


dat = DATA(seqlen=seqlen, separate_char=',')

logging.getLogger().setLevel(logging.INFO)

# k-fold cross validation
k, train_auc_sum, valid_auc_sum = 5, .0, .0
# for i in range(k):
#     lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout)
#     train_data = dat.load_data('../../data/anonymized_full_release_competition_dataset/train' + str(i) + '.txt')
#     valid_data = dat.load_data('../../data/anonymized_full_release_competition_dataset/valid' + str(i) + '.txt')
#     best_train_auc, best_valid_auc = lpkt.train(train_data, valid_data, epoch=30, lr=0.003, lr_decay_step=10)
#     print('fold %d, train auc %f, valid auc %f' % (i, best_train_auc, best_valid_auc))
#     train_auc_sum += best_train_auc
#     valid_auc_sum += best_valid_auc
# print('%d-fold validation: avg of best train auc %f, avg of best valid auc %f' % (k, train_auc_sum / k, valid_auc_sum / k))

# train and pred
# train_data = dat.load_data('../../data/' + pathname + '/train.txt')
# valid_data = dat.load_data('../../data/' + pathname + '/valid.txt')
test_data = dat.load_data('../../data/' + pathname + '/test.txt')
lpkt = LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout)
# lpkt.train(train_data, valid_data, epoch=30, lr=0.003, lr_decay_step=10)
# lpkt.save("lpkt.params")

lpkt.load("lpkt.params")
# _, auc, accuracy, student_mastery = lpkt.eval(test_data)
# print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))

import json
with open('student_mastery.json', 'r') as json_file:
    loaded_student_mastery_str_keys = json.load(json_file)

# 将键转换回适当的类型 (这里假设需要转换为 int)
def convert_keys_to_int(d):
    if isinstance(d, dict):
        return {int(k): convert_keys_to_int(v) if v is not None else v for k, v in d.items()}
    return d

loaded_student_mastery = convert_keys_to_int(loaded_student_mastery_str_keys)
# 读取 JSON 文件
with open('knowledge2id.json', 'r') as f:
    knowledge2id = json.load(f)

with open('abilities.json', 'r') as f:
    abilities = json.load(f)

with open('competencies.json', 'r') as f:
    competencies = json.load(f)

# 定义一个函数来计算加权平均
def weighted_average(values, weights):
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / total_weight

# 计算能力掌握概率
def calculate_abilities(student_mastery, abilities):
    student_abilities = {}
    for student_id, mastery in student_mastery.items():
        student_abilities[student_id] = {}
        for ability_id, k_and_w in abilities.items():
            k_values = [mastery.get(knowledge2id.get(k), 0) for k in k_and_w.keys()]
            weights = list(k_and_w.values())
            ability_score = weighted_average(k_values, weights)
            student_abilities[student_id][ability_id] = ability_score
    return student_abilities

# 计算素养掌握概率
def calculate_competencies(student_abilities, competencies):
    student_competencies = {}
    for student_id, abilities in student_abilities.items():
        student_competencies[student_id] = {}
        for competency_id, a_and_w in competencies.items():
            a_values = [abilities.get(a, 0) for a in a_and_w.keys()]
            weights = list(a_and_w.values())
            competency_score = weighted_average(a_values, weights)
            student_competencies[student_id][competency_id] = competency_score
    return student_competencies

# 计算所有学生的能力掌握概率
student_abilities = calculate_abilities(loaded_student_mastery, abilities)

# 计算所有学生的素养掌握概率
student_competencies = calculate_competencies(student_abilities, competencies)

# 构建 id2knowledge 的反向映射
id2knowledge = {v: k for k, v in knowledge2id.items()}

# 替换 loaded_student_mastery 中的知识点ID为知识点的名字，并去掉键为0的列
def replace_ids_with_names_and_remove_key_zero(student_mastery, id2knowledge):
    student_mastery_with_names = {}
    for student_id, knowledge_dict in student_mastery.items():
        student_mastery_with_names[student_id] = {
            id2knowledge[k_id]: mastery for k_id, mastery in knowledge_dict.items() if k_id != 0
        }
    return student_mastery_with_names

student_mastery_with_names = replace_ids_with_names_and_remove_key_zero(loaded_student_mastery, id2knowledge)



print("Student Knowledge Mastery:", student_mastery_with_names)
print("Student Abilities Mastery:", student_abilities)
print("Student Competencies Mastery:", student_competencies)