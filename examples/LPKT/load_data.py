# Code reused from https://github.com/arghosh/AKT.git
import numpy as np
import math


class DATA(object):
    def __init__(self, seqlen, separate_char):
        self.separate_char = separate_char
        self.seqlen = seqlen

    '''
    data format:
    length
    KC sequence
    answer sequence
    exercise sequence
    it sequence
    at sequence
    '''

    def load_data(self, path):
        f_data = open(path, 'r')
        a_data = []
        e_data = []
        it_data = []
        at_data = []
        student_ids = []  # Add a list to store student IDs  这个学生Id只在测试集中使用，训练集和验证集没有用到，且是我自己添加的
        student_id = 0  # 学生ID我已经添加到数据集中了第二行表示的是学生ID
        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 7 != 0 and lineID % 7 != 1:
                line_data = line.split(self.separate_char)
                if len(line_data[len(line_data) - 1]) == 0:
                    line_data = line_data[:-1]
            if lineID % 7 == 1:
                student_id = int(line)
            elif lineID % 7 == 3:
                A = line_data
            elif lineID % 7 == 4:
                E = line_data
            elif lineID % 7 == 5:
                IT = line_data
            elif lineID % 7 == 6:
                AT = line_data
                # start split the data
                n_split = 1
                total_len = len(A)
                if total_len > self.seqlen:
                    n_split = math.floor(len(A) / self.seqlen)
                    if total_len % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    answer_sequence = []
                    exercise_sequence = []
                    it_sequence = []
                    at_sequence = []
                    if k == n_split - 1:
                        end_index = total_len
                    else:
                        end_index = (k + 1) * self.seqlen
                    # choose the sequence length is larger than 2
                    if end_index - k * self.seqlen > 2:
                        for i in range(k * self.seqlen, end_index):
                            answer_sequence.append(int(A[i]))
                            exercise_sequence.append(int(E[i]))
                            it_sequence.append(int(IT[i]))
                            at_sequence.append(int(AT[i]))

                        # print('instance:-->', len(instance),instance)
                        a_data.append(answer_sequence)
                        e_data.append(exercise_sequence)
                        it_data.append(it_sequence)
                        at_data.append(at_sequence)
                        student_ids.append(student_id)  # Append the student ID for each sequence
        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        a_dataArray = np.zeros((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        it_dataArray = np.zeros((len(it_data), self.seqlen))
        for j in range(len(it_data)):
            dat = it_data[j]
            it_dataArray[j, :len(dat)] = dat

        at_dataArray = np.zeros((len(at_data), self.seqlen))
        for j in range(len(at_data)):
            dat = at_data[j]
            at_dataArray[j, :len(dat)] = dat
        student_idsArray = np.array(student_ids)  # Convert student IDs list to a numpy array
        return a_dataArray, e_dataArray, it_dataArray, at_dataArray, student_idsArray
