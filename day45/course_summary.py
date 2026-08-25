from collections import OrderedDict


LEARNING_PATH = OrderedDict([
    ("基础", ["张量与自动求导", "线性回归", "Softmax 回归", "多层感知机"]),
    ("卷积网络", ["卷积与池化", "经典 CNN", "残差网络", "图像增广"]),
    ("计算机视觉", ["目标检测", "语义分割", "样式迁移"]),
    ("序列模型", ["RNN", "GRU", "LSTM", "seq2seq"]),
    ("注意力", ["注意力机制", "Transformer", "BERT"]),
])


def count_topics(path):
    return sum(len(topics) for topics in path.values())


def next_steps():
    return [
        "选择一个公开数据集完成端到端训练",
        "记录训练曲线并比较不同超参数",
        "阅读模型论文并复现一个核心模块",
        "学习模型部署和推理优化",
    ]


if __name__ == "__main__":
    for module, topics in LEARNING_PATH.items():
        print(f"{module}: {'、'.join(topics)}")
    print("知识点数量:", count_topics(LEARNING_PATH))
    print("后续学习:")
    for item in next_steps():
        print("-", item)
