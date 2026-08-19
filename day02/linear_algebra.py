"""Day 2：向量、矩阵运算与范数。"""
import torch


a = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
other = torch.ones(3, 2)
print("dot product:", torch.dot(a, a))
print("matrix-vector:", matrix @ a)
print("matrix-matrix:\n", matrix @ other)
print("L1 norm:", torch.linalg.vector_norm(a, ord=1))
print("L2 norm:", torch.linalg.vector_norm(a))
print("Frobenius norm:", torch.linalg.matrix_norm(matrix))
