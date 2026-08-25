# Deep Learning Practice

《动手学深度学习》学习代码整理。内容按学习日划分，每个知识点对应一个独立代码文件，主要使用 PyTorch 实现。

## 学习目录

| 天数 | 学习内容 |
| --- | --- |
| Day 1 | [课程安排](day01/course_schedule.py)、[深度学习介绍](day01/deep_learning_intro.py)、[安装](day01/environment_check.py)、[数据操作](day01/tensor_operations.py)、[数据预处理](day01/data_preprocessing.py) |
| Day 2 | [线性代数](day02/linear_algebra.py)、[按特定轴求和](day02/axis_reduction.py)、[矩阵计算](day02/matrix_calculus.py)、[自动求导](day02/autograd.py) |
| Day 3 | [线性回归](day03/linear_regression.py)、[基础优化方法](day03/optimization_basics.py)、[线性回归的从零开始实现](day03/linear_regression_scratch.py)、[线性回归的简洁实现](day03/linear_regression_concise.py) |
| Day 4 | [Softmax 回归](day04/softmax_regression.py)、[损失函数](day04/loss_functions.py)、[图像分类数据集](day04/fashion_mnist.py)、[Softmax 回归的从零开始实现](day04/softmax_scratch.py)、[Softmax 回归的简洁实现](day04/softmax_concise.py) |
| Day 5 | [休课](day05/README.md) |
| Day 6 | [休课](day06/README.md) |
| Day 7 | [感知机](day07/perceptron.py)、[多层感知机](day07/mlp.py)、[多层感知机的从零开始实现](day07/mlp_scratch.py)、[多层感知机的简洁实现](day07/mlp_concise.py) |
| Day 8 | [模型选择](day08/model_selection.py)、[欠拟合和过拟合](day08/underfit_overfit.py) |
| Day 9 | [权重衰减](day09/weight_decay.py)、[Dropout](day09/dropout.py) |
| Day 10 | [数值稳定性](day10/numerical_stability.py)、[模型初始化和激活函数](day10/initialization_activations.py)、[实战 Kaggle 比赛：预测房价](day10/kaggle_house_prices.py)、[竞赛：预测房价](day10/house_price_competition.py) |
| Day 11 | [模型构造](day11/model_construction.py)、[参数管理](day11/parameter_management.py)、[自定义层](day11/custom_layers.py)、[读写文件](day11/save_load.py)、[GPU](day11/gpu.py) |
| Day 12 | [休课](day12/README.md) |
| Day 13 | [休课](day13/README.md) |
| Day 14 | [休课](day14/README.md) |
| Day 15 | [休课](day15/README.md) |
| Day 16 | [休课](day16/README.md) |
| Day 17 | [预测房价竞赛总结](day17/house_price_competition_summary.py)、[从全连接层到卷积](day17/fully_connected_to_convolution.py)、[图像卷积](day17/image_convolution.py) |
| Day 18 | [填充和步幅](day18/padding_stride.py)、[多输入多输出通道](day18/multi_channel.py) |
| Day 19 | [池化层](day19/pooling.py)、[卷积神经网络（LeNet）](day19/lenet.py) |
| Day 20 | [深度卷积神经网络（AlexNet）](day20/alexnet.py)、[使用块的网络（VGG）](day20/vgg.py) |
| Day 21 | [网络中的网络（NiN）](day21/nin.py)、[含并行连结的网络（GoogLeNet）](day21/googlenet.py) |
| Day 22 | [批量归一化](day22/batch_normalization.py)、[残差网络（ResNet）](day22/resnet.py)、[竞赛：图片分类](day22/image_classification_competition.py) |
| Day 23 | [硬件：CPU 和 GPU](day23/cpu_gpu_hardware.py) |
| Day 24 | [更多的专有硬件](day24/specialized_hardware.py)、[多 GPU 训练](day24/multi_gpu_training.py) |
| Day 25 | [休课](day25/README.md) |
| Day 26 | [休课](day26/README.md) |
| Day 27 | [多 GPU 训练的实现](day27/multi_gpu_implementation.py)、[分布式训练](day27/distributed_training.py) |
| Day 28 | [图像增广](day28/image_augmentation.py)、[微调](day28/fine_tuning.py) |
| Day 29 | [实战 Kaggle 比赛：图像分类](day29/kaggle_cifar10.py)、[实战 Kaggle 比赛：狗的品种识别](day29/dog_breed_identification.py) |
| Day 30 | [物体检测](day30/object_detection.py)、[边界框实现](day30/bounding_boxes.py)、[物体检测数据集](day30/detection_dataset.py)、[锚框](day30/anchor_boxes.py)、[竞赛：树叶分类竞赛总结](day30/leaf_classification_summary.py) |
| Day 31 | [区域卷积神经网络（R-CNNs）](day31/rcnn.py)、[单发多框检测（SSD）](day31/ssd.py)、[你只看一次（YOLO）](day31/yolo.py) |
| Day 32 | [多尺度物体检测实现](day32/multiscale_object_detection.py)、[SSD 实现](day32/tiny_ssd.py) |
| Day 33 | [语义分割](day33/semantic_segmentation.py)、[语义分割数据集](day33/semantic_segmentation_dataset.py)、[转置卷积](day33/transposed_convolution.py)、[转置卷积是一种卷积](day33/transposed_convolution_as_convolution.py) |
| Day 34 | [全连接卷积神经网络（FCN）](day34/fcn.py)、[样式迁移](day34/neural_style_transfer.py)、[竞赛：目标检测](day34/object_detection_competition.py) |
| Day 35 | [序列模型](day35/sequence_model.py)、[文本预处理](day35/text_preprocessing.py) |
| Day 36 | [语言模型和数据集](day36/language_model.py)、[循环神经网络](day36/rnn.py) |
| Day 37 | [循环神经网络的从零开始实现](day37/rnn_scratch.py)、[循环神经网络的简洁实现](day37/rnn_concise.py) |
| Day 38 | [门控循环单元（GRU）](day38/gru.py)、[长短期记忆网络（LSTM）](day38/lstm.py)、[深层循环神经网络](day38/deep_rnn.py)、[双向循环神经网络](day38/bidirectional_rnn.py) |
| Day 39 | [休课](day39/README.md) |
| Day 40 | [休课](day40/README.md) |
| Day 41 | [机器翻译与数据集](day41/machine_translation_dataset.py)、[编码器-解码器结构](day41/encoder_decoder.py)、[序列到序列学习（seq2seq）](day41/seq2seq.py)、[束搜索](day41/beam_search.py) |
| Day 42 | [注意力机制](day42/attention_mechanism.py)、[注意力分数](day42/attention_scoring.py)、[使用注意力机制的 seq2seq](day42/attention_seq2seq.py) |
| Day 43 | [自注意力和位置编码](day43/self_attention_positional_encoding.py)、[Transformer](day43/transformer.py) |
| Day 44 | [BERT](day44/bert.py)、[BERT 预训练数据集](day44/bert_pretraining_dataset.py)、[预训练 BERT](day44/bert_pretraining.py)、[微调 BERT](day44/bert_fine_tuning.py)、[自然语言推理和数据集](day44/natural_language_inference_dataset.py)、[自然语言推理：微调 BERT](day44/bert_nli_fine_tuning.py)、[竞赛：目标检测总结](day44/object_detection_competition_summary.py) |
| Day 45 | [优化算法](day45/optimization_algorithms.py)、[课程总结和进阶学习](day45/course_summary.py) |

## 环境

- Python 3.10+
- PyTorch 2.1+
- torchvision 0.16+
- pandas 2.0+

依赖记录在 `requirements.txt`。

## 资料

- [《动手学深度学习》中文教材](https://zh-v2.d2l.ai/)
- [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)
