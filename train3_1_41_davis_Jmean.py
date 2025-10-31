import sys
import os
import time

import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"






# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.dirname(sys.path[0]))

import torch.nn as nn
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from models.graphTran_cnnpro3d_41_Jmean import GraphTransformer

from utils6_78_3 import *
# import ci_cython


#在 PyTorch 中，DataLoader 是用于批处理数据的工具，它默认情况下只能处理一个输入数据（通常是特征）和一个标签数据。如果你希望处理多个输入数据（例如，有两个输入特征），则需要将这些数据组合成一个数据结构，通常是一个元组或字典。然后，你可以自定义数据集类以及相应的 __getitem__ 方法来返回这样的数据结构

# training function at each epoch#训练模型
def train(model, device, train_loader, optimizer, epoch):#(GCN,cuda:0,训练的数据，优化器，轮次增加)
    print('Training on {} samples...'.format(len(train_loader.dataset)))#一共有25046个样本
    model.train()#作用是 启用 batch normalization 和 dropout 。dropout 常常用于抑制过拟合。保证 BN 层能够用到 每一批数据 的均值和方差

    for batch_idx,(data,datapro,datapro_3d) in enumerate(train_loader):
        #python的内置函数，用于遍历序列中的元素并将其索引和值分别作为元组（index,value）返回
        data = data.to(device)
        datapro = datapro.to(device)
        datapro_3d = datapro_3d.to(device)
        optimizer.zero_grad()  # 通常在每个训练步之前调用这个函数。这是因为，在反向传播计算梯度之前，需要将之前计算的梯度清零，以免对当前计算造成影响
        # forward
        output = model(data, datapro,datapro_3d)
        # 模型输入特征

        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))










def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx,(data,datapro,datapro_3d) in enumerate(loader):
            data = data.to(device)
            datapro = datapro.to(device)
            datapro_3d = datapro_3d.to(device)
            output = model(data, datapro,datapro_3d)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)


    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [['davis', 'kiba'][0]]
modeling = [GraphTransformer][0]
# GCNNet:GraphDTA的模型代码
model_st = modeling.__name__
loss_fn = nn.MSELoss()


TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512

LR = 0.0002
LOG_INTERVAL = 20
NUM_EPOCHS = 2000

def main():

    cuda_name = "cuda:0"
    print(torch.cuda.current_device())
    if len(sys.argv) > 3:
        cuda_name = "cuda:"  + str([3])





    for dataset in datasets:#数据准备
        print('\nrunning on ', model_st + '_' + dataset)
        processed_data_file_train_drug = 'datasets/datasets/datasets/processed/' + dataset + '_train_data_mol_78_1.pt'
        processed_data_file_test_drug = 'datasets/datasets/datasets/processed/' + dataset + '_test_data_mol_78_1.pt'
        processed_data_file_train_pro = 'datasets/datasets/datasets/processed/' + dataset + '_train_data_pro_graph_3.pt'
        processed_data_file_test_pro = 'datasets/datasets/datasets/processed/' + dataset + '_test_data_pro_graph_3.pt'
        if ((not os.path.isfile(processed_data_file_train_drug)) or (not os.path.isfile(processed_data_file_test_drug))):
            print('please run create_data.py to prepare data in pytorch format!')
        else:
            train_data_drug = DTADataset(root='datasets/datasets/datasets', dataset=dataset+'_train' )
            test_data_drug = DTADataset(root='datasets/datasets/datasets', dataset=dataset+ '_test')
            train_data_pro = DTADataset_pro(root='datasets/datasets/datasets', dataset=dataset +'_train')
            test_data_pro = DTADataset_pro(root='datasets/datasets/datasets', dataset=dataset+ '_test' )
            train_data_pro_3d = DTADataset_pro_3d(root='datasets/datasets/datasets', dataset=dataset + '_train')
            test_data_pro_3d = DTADataset_pro_3d(root='datasets/datasets/datasets', dataset=dataset + '_test')

            trains = MyDataset(train_data_drug, train_data_pro,train_data_pro_3d )
            tests = MyDataset(test_data_drug, test_data_pro,test_data_pro_3d)





            train_loader= DataLoader(trains, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                               pin_memory=False,drop_last=False)
            test_loader = DataLoader(tests, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                      pin_memory=False,drop_last=False)

            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")#device表示我用的是gpu还是cpu

            model = modeling().to(device)#这里开始进入gcn。py
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            #print("optimizer:",optimizer)#优化算法
            best_mse = 1000
            best_ci = 0
            best_epoch = -1
            model_file_name = 'model_' + model_st + '_' + dataset + '.model'
            result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
            for epoch in range(NUM_EPOCHS):
                time_s = time.time()
                train(model, device, train_loader, optimizer, epoch + 1)#(GCN,cuda:0,训练的数据，优化器，轮次增加)
                print('train cost time ', time.time() - time_s)
                time_s = time.time()
                G, P = predicting(model, device, test_loader)
                print('predict cost time ', time.time() - time_s)
                # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), get_rm2(G, P), best_epoch, ci(G, P)]
                ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), get_rm2(G, P), best_epoch]

                time_s = time.time()
                # ret.append(ci_cython.boost_ci(G, P))
                # 这里优化了她原来的ci函数
                ret.append(boost_ci_v3(G, P))
                print('ci cost time ', time.time() - time_s)

                # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]

                if ret[1] < best_mse:

                    torch.save(model.state_dict(), model_file_name)

                    best_epoch = epoch + 1
                    # 调用两次
                    # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), get_rm2(G, P), best_epoch, ci(G, P)]
                    with open(result_file_name, 'w') as f:
                        f.write(','.join(map(str, ret)))

                    best_mse = ret[1]
                    best_ci = ret[-1]
                    best_rm2 = ret[4]

                    print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci,
                          'best_rm2:', best_rm2,
                          model_st, dataset)
                    print('')


                else:
                    print('')
                    print('MSE:', ret[1], 'CI:', ret[-1], 'RM2:', ret[-3], 'No improvement since epoch ', best_epoch,
                          '; best_mse,best_ci:', best_mse, best_ci,
                          'best_rm2:', best_rm2,
                          model_st, dataset)
                    print('')



if __name__ == '__main__':
    main()