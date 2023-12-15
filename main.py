import torch
import numpy as np
import argparse
import random
import os
from utils.utils_data import prepare_datasets
from utils.utils_loss import rc_mil_loss
from utils.utils_weight import confidence_weight
from utils.models import Linear
from utils.metric import acc_check

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.callback import Callback, LossMonitor
from mindspore import Tensor, set_context, PYNATIVE_MODE, dtype as mstype
from algorithm import UUM

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0',help = "used gpu id",type = str )
parser.add_argument('-lr', default=2e-1,help="learning rate for training",type = float )
parser.add_argument('-batch_size', type=int,help = "used batch size for training", default=128)
parser.add_argument('-dataset',type=str,help = "choose the used dataset", choices=['musk1','musk2','elephant','fox','tiger'],default='fox')
parser.add_argument('-seed',type=int,default=0,help = "used random state")
parser.add_argument('-ep',type=int,help = "number of epoch",default=3500)
parser.add_argument('-wd',default=0,help="weight decay",type=float)
parser.add_argument('-init_ep',default=0,help = "initialize epoch",type=int)
parser.add_argument('-alpha',type=float,help="whether to use matrix to stor \eta(x),\
                    0 denotes use matrix, 1 denotes use current model.",default=0)


args = parser.parse_args()
loss_fn = rc_mil_loss()



batch_size = args.batch_size


random.seed(args.seed)
np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#torch.cuda.manual_seed_all(args.seed)
#torch.backends.cudnn.deterministic = True

device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
print('used device: ' + str(device))
print('seed: '+str(args.seed))

print('generate aggreate dataset')
aggreate_train_loader, val_loader, test_loader, instance_dim, size_m = prepare_datasets(args.dataset,batch_size,args.seed)
K = 2
print('finish generate aggreate dataset')
data_size = aggreate_train_loader.dataset.__len__() + val_loader.dataset.__len__() + test_loader.dataset.__len__()

# set activation function for case k=2
model = Linear(instance_dim).to(device)

acc = acc_check(device,size_m)

if K == 2:
    K = 1



print('start training')
best_loss = 0
mo_dict = model.state_dict()
UUM(aggreate_train_loader, model, loss_fn, alpha, size_m, device, data_size, K, args)
model.load_state_dict(torch.load('data/model/{}dict.pth'.format(os.getpid())))
print('best Val acc:{}. Te acc:{}'.format(acc.acc(val_loader,model,device),acc.acc(test_loader,model,device)))