import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 设置随机种子以便结果可重复
np.random.seed(42)
random.seed(42)

# 定义学生、问题和技能的数量
num_students = 10
num_problems = 50
num_skills = 10

# 生成学生ID列表
student_ids = [i for i in range(1, num_students + 1)]

# 生成问题ID列表
problem_ids = [i for i in range(1, num_problems + 1)]

# 生成技能ID列表
skills = [f"skill_{i}" for i in range(1, num_skills + 1)]

# 生成每个问题对应的多个技能
problem2skills = {problem_id: random.sample(skills, random.randint(1, 3)) for problem_id in problem_ids}

# 生成数据
data = {
    "startTime": [],
    "timeTaken": [],
    "studentId": [],
    "skill": [],
    "problemId": [],
    "correct": []
}

# 随机生成数据
start_date = datetime.now()

for _ in range(500):  # 假设生成500条记录
    student_id = random.choice(student_ids)
    problem_id = random.choice(problem_ids)
    skills_for_problem = problem2skills[problem_id]
    correct = random.choice([0, 1])
    start_time = start_date + timedelta(minutes=random.randint(0, 10000))
    time_taken = random.randint(30, 600)  # 假设答题时间在30秒到10分钟之间

    data["startTime"].append(int(start_time.timestamp()))
    data["timeTaken"].append(time_taken)
    data["studentId"].append(student_id)
    data["skill"].append(','.join(skills_for_problem))
    data["problemId"].append(problem_id)
    data["correct"].append(correct)

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 保存到CSV文件
df.to_csv("lianxi.csv", index=False)
