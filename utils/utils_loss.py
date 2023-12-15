from typing import Callable, List

import mindspore
import mindspore.ops as ops




def rc_mil_loss(activation: Callable = None) -> Callable:
    if activation is None:
        activation = lambda t: ops.sigmoid(t).squeeze()
    def loss(ts: List[mindspore.Tensor], y: mindspore.Tensor, weight: List[mindspore.Tensor], mask, alpha:float) -> mindspore.Tensor:
        ts = ops.cat(ts,dim=1)
        p_z1 = activation(ts)
        p_z0 = 1- p_z1
        lossz0 = ops.log(p_z0 + 1e-32)
        lossz1 = ops.log(p_z1 + 1e-32)
        lossz0[mask==0] = 0
        lossz1[mask==0] = 0
        
        cp_z0 = p_z0.detach()
        cp_z1 = p_z1.detach()
        
        ts = ops.cat(weight,dim=1)
        p_z1 = activation(ts)
        p_z0 = 1- p_z1
        
        p_z0 = alpha * cp_z0 + (1-alpha) * p_z0
        p_z1 = alpha * cp_z1 + (1-alpha) * p_z1
        
        p_z0[mask==0] = 1
        p_z1[mask==0] = 1
        
        p_y0 = p_z0.prod(dim=-1)
        p_y1 = 1 - p_y0
        
        weight1 = p_z1
        weight0 = (1 - p_y0.unsqueeze(-1)/(p_z0 + 1e-32)) * p_z0
        
        
        lossc0 = weight0 * lossz0 / (p_y1.unsqueeze(-1) + 1e-32)
        lossc1 = weight1 * lossz1 / (p_y1.unsqueeze(-1) + 1e-32)
        
        
        loss1 = lossc0.sum(dim = -1)/mask.sum(dim = -1) + lossc1.sum(dim = -1)/mask.sum(dim=-1)
        loss0 = lossz0.sum(dim=-1)/mask.sum(dim = -1)
        
        return ops.nll_loss(ops.stack((loss0,loss1),dim=-1),y)
    return loss
        