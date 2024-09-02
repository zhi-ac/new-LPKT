编程语言：Python3.8； 

开发工具：pycharm；
库和框架：
torch=.2.1.2+cu118
numpy=.1.23.5
pandas=.2.0.3
tqdm=.4.66.1
EduData=.0.0.18
scikit-learn=.1.3.2

3、算法的输入及输出：
输入：学生id，习题id，知识（以,分隔），作答开始时间，作答花费时间，答题正确与否，能力-知识权重关系格式如下

{
"A1": {
"K1": 2,
"K2": 1
},
"A2": {
"K2": 1,
"K3": 2
}
}
，
素养-能力权重关系
{
"S1": {
"A1": 1,
"A2": 3
}
}

输出：学生所有知识、能力和素养的掌握概率


这是一个集成多个模型的代码库
我的代码在LPKT模型中
具体包含以下两个个文件路径中：
EduKTM/LPKT
examples/LPKT


启动项目具体流程如下：
1、先进行数据预处理过程
使用examples\prepare_dataset.ipynb进行数据预处理
2、运行
examples/LPKT/LPKT.py文件









