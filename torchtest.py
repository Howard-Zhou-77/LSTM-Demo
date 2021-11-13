import torch
import torch.nn as nn
a=nn.Linear(3,4,bias=False);
b=nn.Parameter(torch.ones([3]));
c=torch.zeros(1,3,3)
print(a);
print(b);
print(b.size())
print(a(b));
print(c);
print(a(c));