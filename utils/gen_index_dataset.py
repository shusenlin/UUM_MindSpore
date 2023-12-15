
from mindspore.dataset as Dataset
import mindspore

class gen_index_dataset(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        each_x = self.x[index]
        each_label = self.labels[index]
        
        return each_x, each_label, index

class normal_Dataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self,index):
        return self.data[index], self.y[index]
    
class gen_mask_dataset(Dataset):
    def __init__(self, dataset):
        x, y, bag_label = dataset
        dim = x[0][0].size(0)
        max_bag_size = max([len(i) for i in x])
        self.size_m = max_bag_size
        self.data = ops.zeros(len(x),max_bag_size,dim)
        self.mask = ops.zeros(len(x),max_bag_size).type(mindspore.LongTensor)
        self.bag_label = mindspore.tensor(bag_label)
        for i in range(len(x)):
            self.data[i][:len(x[i])] = ops.stack(x[i],dim=0)
            self.mask[i][:len(x[i])] = 1
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index],self.bag_label[index],index,self.mask[index]
    def get_size_m(self):
        return self.size_m