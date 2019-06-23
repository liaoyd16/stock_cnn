import torch
from torch import nn
from torch.autograd import Variable

data = Variable(torch.rand(1, 3, 540, 960))

pool = nn.MaxPool2d(2, 2, return_indices=True)
unpool = nn.MaxUnpool2d(2, 2)

size1 = data.size()
out, indices1 = pool(data)
size2 = out.size()
out, indices2 = pool(out)
size3 = out.size()
out, indices3 = pool(out)


print(out.shape, indices3.shape, size3)
out = unpool(out, indices3, output_size=size3)
out = unpool(out, indices2, output_size=size2)
out = unpool(out, indices1, output_size=size1)