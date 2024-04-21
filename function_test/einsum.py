import torch
import numpy as np
# x = torch.randn(1, 3, 2, 2)
#
# print(x)
#
# x = torch.einsum('ijkl->iklj', x)
#
# print(x.shape)
#
# print(x)

y = torch.rand(2, 2, 2)
x = torch.rand(1, 2, 2)
print(y)
print(x)
# y = torch.argsort(y, dim=1)
# print(y)
print(x+y)


