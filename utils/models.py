
import mindspore.nn as nn
import mindspore.ops as ops

def sigmoid_activation(ts):
    p1 = ops.sigmoid(ts)
    p0 = 1-p1
    return ops.stack((p0,p1),dim=-1)

class Linear(nn.Cell):
    def __init__(self,dim):
        super(Linear,self).__init__()
        self.fc = nn.Linear(dim,1)
    def forward(self,x):
        return self.fc(x)
