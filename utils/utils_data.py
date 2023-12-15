import numpy as np
from utils.gen_index_dataset import gen_index_dataset, normal_Dataset, gen_mask_dataset
import mindspore
import mindspore.ops as ops


def prepare_datasets(dataname, batchsize,seed):
    dataset, dim = data_processing(dataname)
    dataset = gen_mask_dataset(dataset)
    size_m = dataset.get_size_m()
    len_train = int(0.6*dataset.__len__())
    len_temp = dataset.__len__() - len_train
    len_val = int(0.5 * len_temp)
    len_test = len_temp - len_val
    train_dataset, temp_dataset = dataset[:len_train,:], dataset[len_train:len_train+len_temp,:]
    val_dataset, test_dataset = temp_dataset[:len_val,:], temp_dataset[len_val:len_val+len_test,:]
    train_data = mindspore.dataset.GeneratorDataset(dataset=train_dataset,shuffle=True,num_parallel_workers=0)
    val_data = mindspore.dataset.GeneratorDataset(dataset=val_dataset,shuffle=True,num_parallel_workers=0)
    test_data = mindspore.dataset.GeneratorDataset(dataset=test_dataset,shuffle=True,num_parallel_workers=0)
    train_loader = train_data.batch(batchsize)
    val_loader = val_data.batch(batchsize)
    test_loader = test_data.batch(batchsize)
    return train_loader, val_loader, test_loader,dim,size_m


 

def file_processing(data_file, dim, num_bags):
    ordinary_data = []
    ordinary_label = []
    ordinary_bag_label = []
    bag_label = []
    temp_bag_id = '0'
    bag_instance = []
    
    with open(data_file) as f:
        for l in f.readlines():
            if l[0] == '#':
                continue
            ss = l.strip().split(' ') # ss = ['4776:919:-1', '0:-1.7513811627811062', ...]
# where 4776 means the 4776-th instance, 919 means the 919-th bag, -1 means the label of the instance
            x = torch.zeros(dim)
            for s in ss[1:]:
                i, xi = s.split(':') # each index and feature for each instance
                #i     = int(i) - 1
                xi    = float(xi) # each feature value
                x[int(i)]  = xi
            _, bag_id, y = ss[0].split(':') # get bid_id and label
            y = (int(y)+1)//2
            if bag_id != temp_bag_id:
                temp_bag_id = bag_id
                ordinary_data.append(bag_instance) # 4777 datum: x, y, bag_id
                ordinary_label.append(bag_label)
                ordinary_bag_label.append(max(bag_label))
                bag_label = []
                bag_instance = []
                
            bag_instance.append(x)
            bag_label.append(y)
        ordinary_data.append(bag_instance) # 4777 datum: x, y, bag_id
        ordinary_label.append(bag_label)
        ordinary_bag_label.append(max(bag_label))
    return ordinary_data,ordinary_label,ordinary_bag_label

def data_processing(dataset_name):
    if dataset_name == 'musk1':
        ordinary_data = file_processing('data/musk1.data', 166, 92*10) #920 bags
        dim = 166
    elif dataset_name == 'musk2':
        ordinary_data = file_processing('data/musk2.data', 166, 102*10)
        dim = 166
    elif dataset_name == 'elephant':
        ordinary_data = file_processing('data/elephant.data', 230, 200*10)
        dim = 230
    elif dataset_name == 'fox':
        ordinary_data = file_processing('data/fox.data', 230, 200*10)
        dim = 230
    elif dataset_name == 'tiger':
        ordinary_data = file_processing('data/tiger.data', 230, 200*10)
        dim = 230
    return ordinary_data, dim

def split_dataset(dataset,fraction):
    data,label,bag_label = dataset
    length = len(data)
    index = list(np.random.choice(length ,int(fraction*length), replace=False))
    a_data = [data[i] for i in index]
    a_label = [label[i] for i in index]
    a_bag_label = [bag_label[i] for i in index]
    b_data = [data[i] for i in range(length) if i not in index]
    b_label = [label[i] for i in range(length) if i not in index]
    b_bag_label = [bag_label[i] for i in range(length) if i not in index]
    return (a_data,a_label,a_bag_label),(b_data,b_label,b_bag_label)

def generate_normaldataset(dataset):
    data,label,_ = dataset
    data = [j for i in data for j in i]
    label = [j for i in label for j in i]
    data = ops.stack(data,dim=0)
    label = mindspore.LongTensor(label)
    return normal_Dataset(data,label)
    
    
    