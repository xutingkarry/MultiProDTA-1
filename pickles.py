import torch
import torch_geometric
from torch_geometric.data import Data
from Bio import PDB
import numpy as np


def pdb_to_protein_graph(pdb_path, center, bbox_size):
    """
    从PDB文件生成蛋白质图结构。

    :param pdb_path: PDB文件的路径。
    :param center: 口袋区域的中心 (x, y, z)。
    :param bbox_size: 口袋区域的边界框大小。
    :return: 转换后的图数据对象。
    """
    # 设置中心点和边界框大小
    center = torch.FloatTensor(center)
    prot_data = {}

    # 解析PDB文件
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    # 获取蛋白质的元素表和原子坐标
    atom_list = []
    node_features = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':  # 忽略氢原子
                        continue
                    resname = residue.get_resname()
                    if resname not in ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                                       'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'TRP', 'TYR', 'VAL']:
                        continue  # 只考虑标准氨基酸残基
                    x, y, z = atom.coord
                    atom_pos = torch.FloatTensor([x, y, z])

                    # 根据给定的中心点和边界框大小筛选原子
                    if (atom_pos - center).abs().max() > (bbox_size / 2):
                        continue

                    atom_list.append(atom)
                    node_features.append(atom_pos)

    if len(node_features) == 0:
        raise ValueError(f'No atoms found in the bounding box (center={center}, size={bbox_size}).')

    # 构建节点特征张量
    node_features = torch.stack(node_features, dim=0)  # (num_nodes, 3)

    # 计算残基之间的距离，并构建边
    edge_index = []
    edge_attr = []
    for i in range(len(node_features)):
        for j in range(i + 1, len(node_features)):
            dist = torch.norm(node_features[i] - node_features[j])
            if dist < bbox_size:  # 距离阈值，用于决定是否生成边
                edge_index.append([i, j])
                edge_index.append([j, i])  # 无向图
                edge_attr.append(dist.item())  # 边的特征为距离
                edge_attr.append(dist.item())  # 对称边

    # 将边连接和边特征转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    # # 创建图数据对象
    # data = Data(
    #     x=node_features,  # 节点特征
    #     edge_index=edge_index,  # 边连接
    #     edge_attr=edge_attr  # 边特征
    # )

    return node_features,edge_index,edge_attr


# 使用示例
# if __name__ == '__main__':
#     pdb_path = 'src/pdb/AAK1.pdb'  # PDB文件路径
#     center = [32.0, 28.0, 36.0]  # 口袋区域中心
#     bbox_size = 23.0  # 边界框大小
#
#     # 生成蛋白质图数据
#     protein_graph = pdb_to_protein_graph(pdb_path, center, bbox_size)
#
#     # 输出图的数据
#     print(f"节点特征：{protein_graph.x.shape}")
#     print(f"边的连接关系：{protein_graph.edge_index.shape}")
#     print(f"边的特征：{protein_graph.edge_attr.shape}")

