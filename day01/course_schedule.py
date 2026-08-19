"""Day 1：课程安排。"""

SCHEDULE = {
    "基础": ["数据操作", "线性代数", "线性回归", "Softmax 回归", "多层感知机"],
    "卷积神经网络": ["卷积基础", "LeNet", "AlexNet", "VGG", "NiN", "GoogLeNet", "ResNet"],
    "计算机视觉": ["多 GPU", "图像增广", "目标检测", "语义分割"],
    "循环神经网络": ["序列模型", "RNN", "GRU", "LSTM"],
}

if __name__ == "__main__":
    for module, topics in SCHEDULE.items():
        print(f"{module}: {'、'.join(topics)}")
