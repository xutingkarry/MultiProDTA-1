import os
import torch
import numpy as np
from tqdm import tqdm
from math import sqrt
from scipy import stats
from numba import jit
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset, Dataset

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count









class MyDataset(Dataset):
    def __init__(self, datasetA, datasetB,datasetC):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC

    def __getitem__(self, index):
        # index  = index % len(self.datasetA)
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        xC = self.datasetC[index]
        return xA, xB,xC
    def __len__(self):
        return len(self.datasetB)




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)

def load_checkpoint(model_path):
    return torch.load(model_path)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x


VOCAB_PROTEIN = { "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, 
				"H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12, 
				"P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, 
				"W": 19, "Y": 20, "X": 21}

VOCAB_LIGAND = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def PROTEIN2INT(target):
    return [VOCAB_PROTEIN[s] for s in target] 

def MOL2INT(smi):
    return [VOCAB_LIGAND[s] for s in smi]






class DTADataset_pro_3d(InMemoryDataset):
    def __init__(self, root='datasets/datasets/datasets', dataset='davis',
                 drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None,
                 transform=None,
                 pre_transform=None, smile_graph=None, target_graph=None):
        super(DTADataset_pro_3d, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drug_smiles = drug_smiles
        self.target_sequence = target_sequence
        self.y = y
        self.smile_graph = smile_graph
        self.target_graph = target_graph
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))

            self.data,self.slices= torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(drug_smiles, target_sequence, x, x_mask, xt, xt_mask, y, smile_graph, target_graph)
            self.data,self.slices= torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        # return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']
        # return self.dataset + '_data_mol.pt'
        return self.dataset + '_data_pro_3d_3_9_3.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None,
                smile_graph=None, target_graph=None):
        assert (len(drug_smiles) == len(target_sequence) and len(drug_smiles) == len(
            y)), 'The three lists must have the same length!'
        data_list_mol = []
        data_list_pro = []

        data_len = len(drug_smiles)
        print('loading tensors ...')
        for i in tqdm(range(data_len)):

            smiles = drug_smiles[i]
            tar_seq = target_sequence[i]
            if x is not None:
                drug = x[i]
            if x_mask is not None:
                drug_mask = x_mask[i]
            if xt is not None:
                target = xt[i]
            if xt_mask is not None:
                target_mask = xt_mask[i]
            labels = y[i]
            mol_size, mol_features, mol_edge_index, mol_edge_attr = smile_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight = target_graph[tar_seq]

            drug_seq = MOL2INT(smiles)
            drug_seq_len = 220
            if len(drug_seq) < drug_seq_len:
                mol_seq_emb = np.pad(drug_seq, (0, drug_seq_len - len(drug_seq)))
            else:
                mol_seq_emb = drug_seq[:drug_seq_len]
            GCNData_mol = DATA.Data(x=mol_features,
                                    edge_index=mol_edge_index,
                                    edge_attr=mol_edge_attr,
                                    mol_emb=torch.LongTensor([mol_seq_emb]),
                                    y=torch.FloatTensor([labels]))
            if x is not None:
                GCNData_mol.drug = torch.LongTensor([drug])
            if x_mask is not None:
                GCNData_mol.drug_mask = torch.LongTensor([drug_mask])
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            target_seq = PROTEIN2INT(tar_seq)
            target_seq_len = 1200
            if len(target_seq) < target_seq_len:
                pro_seq_emb = np.pad(target_seq, (0, target_seq_len - len(target_seq)))
            else:
                pro_seq_emb = target_seq[:target_seq_len]

            GCNData_pro = DATA.Data(x=target_features,
                                    edge_index=target_edge_index,
                                    edge_attr=target_edge_weight,
                                    pro_emb=torch.LongTensor([pro_seq_emb]),
                                    y=torch.FloatTensor([labels]))

            if xt is not None:
                GCNData_pro.target = torch.LongTensor([target])
            if xt_mask is not None:
                GCNData_pro.target_mask = torch.LongTensor([target_mask])
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)
            # data, slices = self.collate(data_list_mol)
            data, slices = self.collate(GCNData_pro)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        torch.save((data, slices), self.processed_paths[0])



class DTADataset_pro_3d_11(InMemoryDataset):
    def __init__(self, root='datasets/datasets/datasets', dataset='davis',
                 drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None,
                 transform=None,
                 pre_transform=None, smile_graph=None, target_graph=None):
        super(DTADataset_pro_3d_11, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drug_smiles = drug_smiles
        self.target_sequence = target_sequence
        self.y = y
        self.smile_graph = smile_graph
        self.target_graph = target_graph
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))

            self.data,self.slices= torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(drug_smiles, target_sequence, x, x_mask, xt, xt_mask, y, smile_graph, target_graph)
            self.data,self.slices= torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        # return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']
        # return self.dataset + '_data_mol.pt'
        return self.dataset + '_data_cold_target_pro_3d_3.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None,
                smile_graph=None, target_graph=None):
        assert (len(drug_smiles) == len(target_sequence) and len(drug_smiles) == len(
            y)), 'The three lists must have the same length!'
        data_list_mol = []
        data_list_pro = []

        data_len = len(drug_smiles)
        print('loading tensors ...')
        for i in tqdm(range(data_len)):

            smiles = drug_smiles[i]
            tar_seq = target_sequence[i]
            if x is not None:
                drug = x[i]
            if x_mask is not None:
                drug_mask = x_mask[i]
            if xt is not None:
                target = xt[i]
            if xt_mask is not None:
                target_mask = xt_mask[i]
            labels = y[i]
            mol_size, mol_features, mol_edge_index, mol_edge_attr = smile_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight = target_graph[tar_seq]

            drug_seq = MOL2INT(smiles)
            drug_seq_len = 220
            if len(drug_seq) < drug_seq_len:
                mol_seq_emb = np.pad(drug_seq, (0, drug_seq_len - len(drug_seq)))
            else:
                mol_seq_emb = drug_seq[:drug_seq_len]
            GCNData_mol = DATA.Data(x=mol_features,
                                    edge_index=mol_edge_index,
                                    edge_attr=mol_edge_attr,
                                    mol_emb=torch.LongTensor([mol_seq_emb]),
                                    y=torch.FloatTensor([labels]))
            if x is not None:
                GCNData_mol.drug = torch.LongTensor([drug])
            if x_mask is not None:
                GCNData_mol.drug_mask = torch.LongTensor([drug_mask])
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            target_seq = PROTEIN2INT(tar_seq)
            target_seq_len = 1200
            if len(target_seq) < target_seq_len:
                pro_seq_emb = np.pad(target_seq, (0, target_seq_len - len(target_seq)))
            else:
                pro_seq_emb = target_seq[:target_seq_len]

            GCNData_pro = DATA.Data(x=target_features,
                                    edge_index=target_edge_index,
                                    edge_attr=target_edge_weight,
                                    pro_emb=torch.LongTensor([pro_seq_emb]),
                                    y=torch.FloatTensor([labels]))

            if xt is not None:
                GCNData_pro.target = torch.LongTensor([target])
            if xt_mask is not None:
                GCNData_pro.target_mask = torch.LongTensor([target_mask])
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)
            # data, slices = self.collate(data_list_mol)
            data, slices = self.collate(GCNData_pro)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        torch.save((data, slices), self.processed_paths[0])

















class DTADataset_pro(InMemoryDataset):
    def __init__(self, root='datasets/datasets/datasets', dataset='davis',
                 drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None,
                 transform=None,
                 pre_transform=None, smile_graph=None, target_graph=None):
        super(DTADataset_pro, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drug_smiles = drug_smiles
        self.target_sequence = target_sequence
        self.y = y
        self.smile_graph = smile_graph
        self.target_graph = target_graph
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))

            self.data,self.slices= torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(drug_smiles, target_sequence, x, x_mask, xt, xt_mask, y, smile_graph, target_graph)
            self.data,self.slices= torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        # return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']
        # return self.dataset + '_data_mol.pt'
        return self.dataset + '_data_cold_target_pro_graph.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None,
                smile_graph=None, target_graph=None):
        assert (len(drug_smiles) == len(target_sequence) and len(drug_smiles) == len(
            y)), 'The three lists must have the same length!'
        data_list_mol = []
        data_list_pro = []

        data_len = len(drug_smiles)
        print('loading tensors ...')
        for i in tqdm(range(data_len)):

            smiles = drug_smiles[i]
            tar_seq = target_sequence[i]
            if x is not None:
                drug = x[i]
            if x_mask is not None:
                drug_mask = x_mask[i]
            if xt is not None:
                target = xt[i]
            if xt_mask is not None:
                target_mask = xt_mask[i]
            labels = y[i]
            mol_size, mol_features, mol_edge_index, mol_edge_attr = smile_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight = target_graph[tar_seq]

            drug_seq = MOL2INT(smiles)
            drug_seq_len = 220
            if len(drug_seq) < drug_seq_len:
                mol_seq_emb = np.pad(drug_seq, (0, drug_seq_len - len(drug_seq)))
            else:
                mol_seq_emb = drug_seq[:drug_seq_len]
            GCNData_mol = DATA.Data(x=mol_features,
                                    edge_index=mol_edge_index,
                                    edge_attr=mol_edge_attr,
                                    mol_emb=torch.LongTensor([mol_seq_emb]),
                                    y=torch.FloatTensor([labels]))
            if x is not None:
                GCNData_mol.drug = torch.LongTensor([drug])
            if x_mask is not None:
                GCNData_mol.drug_mask = torch.LongTensor([drug_mask])
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            target_seq = PROTEIN2INT(tar_seq)
            target_seq_len = 1200
            if len(target_seq) < target_seq_len:
                pro_seq_emb = np.pad(target_seq, (0, target_seq_len - len(target_seq)))
            else:
                pro_seq_emb = target_seq[:target_seq_len]

            GCNData_pro = DATA.Data(x=target_features,
                                    edge_index=target_edge_index,
                                    edge_attr=target_edge_weight,
                                    pro_emb=torch.LongTensor([pro_seq_emb]),
                                    y=torch.FloatTensor([labels]))

            if xt is not None:
                GCNData_pro.target = torch.LongTensor([target])
            if xt_mask is not None:
                GCNData_pro.target_mask = torch.LongTensor([target_mask])
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)
            # data, slices = self.collate(data_list_mol)
            data, slices = self.collate(GCNData_pro)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        torch.save((data, slices), self.processed_paths[0])





# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='datasets/datasets/datasets', dataset='davis',
                drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None, transform=None,
                pre_transform=None, smile_graph=None,  target_graph=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.drug_smiles = drug_smiles
        self.target_sequence = target_sequence
        self.y = y
        self.smile_graph = smile_graph
        self.target_graph = target_graph
        # self.process(drug_smiles, target_sequence, x, x_mask, xt, xt_mask, y, smile_graph, target_graph)
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices= torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(drug_smiles, target_sequence, x, x_mask, xt, xt_mask, y, smile_graph, target_graph)
            self.data, self.slices= torch.load(self.processed_paths[0])



    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return self.dataset + '_data_cold_target_mol.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug_smiles=None, target_sequence=None, x=None, x_mask=None, xt=None, xt_mask=None, y=None, smile_graph=None, target_graph=None):
        assert (len(drug_smiles) == len(target_sequence) and len(drug_smiles) == len(y)), 'The three lists must have the same length!'
        data_list_mol = []
        data_list_pro = []

        data_len = len(drug_smiles)
        print('loading tensors ...')
        for i in tqdm(range(data_len)):

            smiles = drug_smiles[i]
            tar_seq = target_sequence[i]
            if x is not None:
                drug = x[i]
            if x_mask is not None:
                drug_mask = x_mask[i]
            if xt is not None:
                target = xt[i]
            if xt_mask is not None:
                target_mask = xt_mask[i]
            labels = y[i]
            mol_size, mol_features, mol_edge_index, mol_edge_attr= smile_graph[smiles]
            target_size, target_features, target_edge_index, target_edge_weight = target_graph[tar_seq]

            drug_seq = MOL2INT(smiles)
            drug_seq_len = 220
            if len(drug_seq) < drug_seq_len:
                mol_seq_emb = np.pad(drug_seq, (0, drug_seq_len- len(drug_seq)))
            else:
                mol_seq_emb = drug_seq[:drug_seq_len]
            GCNData_mol = DATA.Data(x=mol_features,
                                    edge_index=mol_edge_index,
                                    edge_attr = mol_edge_attr,
                                    mol_emb = torch.LongTensor([mol_seq_emb]),
                                    y=torch.FloatTensor([labels]))
            if x is not None:
                GCNData_mol.drug = torch.LongTensor([drug])
            if x_mask is not None:
                GCNData_mol.drug_mask = torch.LongTensor([drug_mask])
            GCNData_mol.__setitem__('c_size', torch.LongTensor([mol_size]))

            target_seq = PROTEIN2INT(tar_seq)
            target_seq_len = 1000
            if len(target_seq) < target_seq_len:
                pro_seq_emb = np.pad(target_seq, (0, target_seq_len- len(target_seq)))
            else:
                pro_seq_emb = target_seq[:target_seq_len]

            GCNData_pro = DATA.Data(x=target_features,
                                    edge_index=target_edge_index,
                                    edge_attr = target_edge_weight,
                                    pro_emb = torch.LongTensor([pro_seq_emb]),
                                    y=torch.FloatTensor([labels]))
            
            if xt is not None:
                GCNData_pro.target = torch.LongTensor([target])
            if xt_mask is not None:
                GCNData_pro.target_mask = torch.LongTensor([target_mask])
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

            # data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)
            # data, slices = self.collate(data_list_mol)
            data, slices = self.collate(data_list_pro)




        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]


        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        torch.save((data, slices), self.processed_paths[0])




#     def __getitem__(self, idx):
#         # return GNNData_mol, GNNData_pro
#         return self.data_mol[idx], self.data_pro[idx]
#
# def collate(data_list):
#     batchA = Batch.from_data_list([data[0] for data in data_list])
#     batchB = Batch.from_data_list([data[1] for data in data_list])
#     return batchA, batchB










def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

def ci_v3(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)
    j = i - 1
    i_len = i - 1
    j_len = j - 1
    z = 0.0
    S = 0.0
    for l1 in range(i):
        for l2 in range(j):
            if y[i_len - l1] > y[j_len - l2]:
                z = z + 1
                u = f[i_len - l1] - f[j_len - l2]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
        j = j - 1

    ci = S / z
    return ci

# 使用numba加速CI运算速度
@jit(nopython=True)
def boost_ci_v3(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)
    j = i - 1
    i_len = i - 1
    j_len = j - 1
    z = 0.0
    S = 0.0
    for l1 in range(i):
        for l2 in range(j):
            if y[i_len - l1] > y[j_len - l2]:
                z = z + 1
                u = f[i_len - l1] - f[j_len - l2]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
        j = j - 1
    ci = S / z
    return ci


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))
