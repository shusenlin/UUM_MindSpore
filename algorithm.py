import mindspore.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import mindspore
from mindspore.ops import operations as ops
zeros = ops.Zeros()
from utils.utils_weight import confidence_weight
from mindspore import Model
from mindspore.train.callback import Callback, LossMonitor

class CustomWithLossCell(nn.Cell):
    """Connect the forward network and the loss function"""

    def __init__(self, backbone, loss_fn, alpha, size_m, device, weight, args):
        """There are two inputs, the forward network backbone and the loss function"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._alpha = alpha
        self._size_m = size_m
        self._args = args
        self._weight = weight
        self._device = device
        if self._args.init_ep>0:
            self._alpha = 1
        else:
            self._alpha = 0
        
    def construct(self, batchX, batchY, index, mask):
        total_loss = 0
        x, label, index, mask = batchX.to(self._device), batchY.to(self._device), index.to(self._device),mask.to(self._device)
        x = x.reshape((x.size(0)*self._size_m,)+x.shape[2:])
        outputs = self._backbone(x)
        outputs = outputs.reshape(-1,self._size_m,K)
        outputs = outputs.split(1,1)
        loss = self._loss_fn(outputs,label,self._weight.get_weight(index),mask,self._alpha)
        total_loss += loss*x.size(0)
        self._weight.update_weight(self._backbone,x,index)

        return loss
    
def UUM(aggreate_train_loader, model, loss_fn, alpha, size_m, device, data_size, K, args):
    weight = confidence_weight(device,data_size,size_m,K)
    weight.init_weight(aggreate_train_loader,model,args.init_ep==0)
    custom_model = CustomWithLossCell(model, loss_fn, alpha, size_m, device, weight, args)
    optimizer = mindspore.nn.Adam(model.parameters(), learning_rate = args.lr, weight_decay = args.wd)
    model = Model(custom_model, optimizer=optimizer)
    model.train(args.ep, aggreate_train_loader, callbacks=[LossMonitor(1), ])