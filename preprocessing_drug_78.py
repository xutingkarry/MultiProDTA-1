import os
import torch
import os.path as osp
import json, pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from protein_to_graph import coord_to_graph

from utils6 import *
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDConfig
from collections import OrderedDict
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


def read_data(filename):
    df = pd.read_csv(filename)
    drugs, prots, Y = list(df['smiles']),list(df['protein_seqence']),list(df['affinity'])
    return drugs, prots, Y

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile2graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    num_atoms = mol.GetNumAtoms()

    # 初始化邻接矩阵
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    # 遍历分子的每条化学键，填充邻接矩阵
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        adjacency_matrix[begin_atom_idx, end_atom_idx] = 1
        adjacency_matrix[end_atom_idx, begin_atom_idx] = 1


    return c_size, features, edge_index

def get_nodes(g):
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                                'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                                'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
                                                'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                                'Pt', 'Hg', 'Pb', 'X']]
        h_t.append(d['a_num'])
        h_t.append(d['acceptor'])
        h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['degree'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['ImplicitValence'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['num_h'] == x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        h_t += [int(d['hybridization'] == x) for x in (Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3)]
        # 5 more
        h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge'])
        h_t.append(d['NumExplicitHs'])
        h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])
    return node_attr

def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]

        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))


    return edge_index, edge_attr

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(VOCAB_PROTEIN)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_hot_encoding(pro_seq[i], VOCAB_PROTEIN)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)

    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

def target2feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)

    return feature


def sequence2graph(target_key, target_sequence, distance_dir):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)

    target_feature = seq_feature(target_sequence)
    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    distance_map = np.load(contact_map_file)

    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  # dege
        target_edge_distance.append(distance_map[i, j])  # edge weight

    target_feature = torch.Tensor(target_feature)
    target_edge_index = torch.LongTensor(target_edge_index).transpose(1, 0)
    target_edge_distance = torch.FloatTensor(target_edge_distance)

    # pro_graph=coord_to_graph(target_sequence,1000,'cuda:0')


    return target_size, target_feature, target_edge_index, target_edge_distance

def create_dataset(dataset):
    dataset_dir = os.path.join('datasets/datasets/datasets/',dataset)
    # drug smiles
    ligands = json.load(open(os.path.join(dataset_dir, 'ligands_can.txt')), object_pairs_hook=OrderedDict)
    # protein sequences
    proteins = json.load(open(os.path.join(dataset_dir, 'proteins.txt')), object_pairs_hook=OrderedDict)

    # load protein feature and predicted distance map
    process_dir = os.path.join('datasets/datasets/datasets/')
    pro_distance_dir = os.path.join(process_dir, dataset, 'pconsc4')  # numpy .npy file
    pro_msa_path = os.path.join(process_dir, dataset, 'aln')  # numpy .npy file

    # dataset process
    drugs = []  # rdkit entity
    prots = []  # sequences
    prot_keys = []  # protein id (or name)
    drug_smiles = []  # smiles
    # create molecule graph
    print("create molecule graph ...")
    # smiles
    for d in ligands.keys():
        if dataset == 'metz':
            lg = ligands[d]
        else:
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        #lg = ligands[d]
        drugs.append(lg)
        drug_smiles.append(ligands[d])
        smile_graph = {}

    for i in tqdm(range(len(drugs))):
        smile = drugs[i]
        g_d = smile2graph(smile)
        smile_graph[smile] = g_d



    print("create protein graph ...")
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)

    target_graph = {}
    for i in tqdm(range(len(prot_keys))):
        key = prot_keys[i]
        protein = prots[i]
        g_t = sequence2graph(key, protein, pro_distance_dir)
        target_graph[protein] = g_t
    
    # read files(train and test)
    train_csv = process_dir+dataset + '/raw/data_train3.csv'
    test_csv = process_dir+dataset + '/raw/data_test3.csv'

    train_drugs, train_prots, train_Y = read_data(train_csv)
    test_drugs, test_prots,test_Y = read_data(test_csv)

    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

    train_data = DTADataset(root='datasets/datasets/datasets', dataset=dataset + '_' + 'train', drug_smiles=train_drugs,
                            target_sequence=train_prots, y=train_Y,
                            smile_graph=smile_graph, target_graph=target_graph)

    test_data = DTADataset(root='datasets/datasets/datasets', dataset=dataset + '_' + 'test', drug_smiles=test_drugs,
                           target_sequence=test_prots, y=test_Y,
                           smile_graph=smile_graph, target_graph=target_graph)
    train_data_pro = DTADataset1(root='datasets/datasets/datasets', dataset=dataset + '_' + 'train',
                                 drug_smiles=train_drugs,
                                 target_sequence=train_prots, y=train_Y,
                                 smile_graph=smile_graph, target_graph=target_graph)

    test_data_pro = DTADataset1(root='datasets/datasets/datasets', dataset=dataset + '_' + 'test',
                                drug_smiles=test_drugs,
                                target_sequence=test_prots, y=test_Y,
                                smile_graph=smile_graph, target_graph=target_graph)

    return train_data,test_data
if __name__ == '__main__':
    # df = pd.read_csv("datasets/datasets/datasets/" + 'davis/raw/' + "data.csv")
    # X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    # X_train.to_csv("datasets/datasets/datasets/" + 'davis/raw/' + "data_train.csv", index=False)
    # X_test.to_csv("datasets/datasets/datasets/" + 'davis/raw/' + "data_test.csv", index=False)
    # print("数据集分割成功")

    create_dataset('davis')
