

import mindspore
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import stop_gradient

class confidence_weight():
    def __init__(self,device,data_size,size_m,K):
        if K == 2:
            K = 1
        self.device = device
        self.weight_tensor = ops.ones((data_size,size_m,K)).to(device)
        self.size_m = size_m
        self.K = K
    def update_weight(self,model,x,index):
        outputs = model(x)
        outputs = stop_gradient(outputs)
        outputs = outputs.reshape(-1,self.size_m,self.K)
        self.weight_tensor[index] = outputs
        
    def init_weight(self,loader,model,uniform):
        if uniform:
            if self.K != 1:
                self.weight_tensor = self.weight_tensor/self.K
            else: self.weight_tensor = self.weight_tensor * 0
        else:
            for i,(x,label,index,_) in enumerate(loader):
                x, label, index = x.to(self.device), label.to(self.device), index.to(self.device)
                x = x.reshape((x.size(0)*self.size_m,)+x.shape[2:])
                outputs = model(x)
                outputs = stop_gradient(outputs)
                outputs = outputs.reshape(-1,self.size_m,self.K)
                self.weight_tensor[index] = outputs
    def get_weight(self,index):
        return self.weight_tensor[index].split(1,1)

