import numpy as np

# 习题知识关联矩阵 (Q矩阵)
Q = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 0]
])

# 初始化学生对每个知识点的掌握概率
num_students = 1
num_skills = Q.shape[1]
mastery_prob = np.full((num_students, num_skills), 0.5)

# 学生做题记录 (例如，学生做对了第1和第3题，做错了第2题)
student_answers = np.array([1, 0, 1])

# 更新知识掌握状态 (简单示例)
for i in range(len(student_answers)):
    for j in range(num_skills):
        if Q[i, j] == 1:
            if student_answers[i] == 1:  # 做对了题目
                mastery_prob[0, j] = min(mastery_prob[0, j] + 0.1, 1.0)
            else:  # 做错了题目
                mastery_prob[0, j] = max(mastery_prob[0, j] - 0.1, 0.0)

print("学生的知识掌握概率：", mastery_prob)
