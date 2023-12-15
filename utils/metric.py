import torch

import mindspore
from mindspore.ops as ops
from mindspore.ops import stop_gradient
class acc_check():
    def __init__(self,device,size_m):
        self.device = device
        self.size_m = size_m
    def acc(self,loader, model, device):
        total = 0
        correct = 0
        for x, label, index, mask in loader:
            x, label, mask = x.to(device), label.to(device), mask.to(device)
            x = x.reshape((x.size(0)*self.size_m,)+x.shape[2:])
            outputs = model(x)
            outputs = stop_gradient(outputs)
            outputs = outputs.reshape(-1,self.size_m)
            outputs[mask==0] = -1
            predict = (outputs>0).type(mindspore.LongTensor).to(device)
            predict = predict.max(dim=1)[0]
            correct += ops.sum(predict==label)
            total += outputs.size(0)
            return correct/total
                
            
