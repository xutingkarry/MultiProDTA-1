import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torch import nn
import torch_geometric.nn as gnn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, global_max_pool as gmp,GATConv,GINConv
from torch_geometric.nn import GCNConv, global_max_pool as gmp,GATConv,GINConv,global_mean_pool,global_add_pool,global_max_pool
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_max_pool as gmp, global_add_pool as gap, global_mean_pool as gep, global_sort_pool
from einops.layers.torch import Rearrange, Reduce
# -*- coding: utf-8 -*-

from .layers2coss_gcn6 import TransformerEncoderLayer
from einops import repeat
class FCLayer1(nn.Module):
    def __init__(self, in_size=128):
        super(FCLayer1, self).__init__()
        self.att=nn.Linear(in_size,in_size*2)
        self.gate=nn.Linear(in_size*2,in_size*2)
        self.fc1=nn.Linear(in_size*2,in_size)
        self.relu=nn.ReLU()
        self.b=nn.BatchNorm1d(in_size)
        self.dropout=nn.Dropout(0.2)
    def forward(self, pro_seq,pro_graph):
        att1=pro_seq
        att2 = self.dropout(self.relu(self.att(pro_graph)))
        fc=torch.sigmoid(self.gate(att1)+self.gate(att2))
        fusion=fc*att1+(1-fc)*att2
        fusion=self.dropout(self.b(self.relu(self.fc1(fusion))))
        return fusion
class FCLayer2(nn.Module):
    def __init__(self, in_size=128):
        super(FCLayer2, self).__init__()
        self.att=nn.Linear(8,8)
        self.gate=nn.Linear(in_size*2,in_size*2)
        self.fc1=nn.Linear(136,in_size)
        self.relu=nn.ReLU()
        self.b=nn.BatchNorm1d(in_size)
        self.dropout=nn.Dropout(0.2)

        self.fc2 = nn.Linear(132, in_size)
    def forward(self, pro_seq,pro_3d_graph):
        # att1=pro_seq
        # att2 = self.dropout(self.relu(self.att(pro_graph)))
        # fc=torch.sigmoid(torch.cat((self.gate(att1),self.gate(att2)),dim=1))
        # fusion=fc*att1+(1-fc)*att2
        # fusion=torch.cat((pro_seq,pro_graph),dim=1)
        # fusion=self.dropout(self.b(self.relu(self.fc1(fusion))))
        fusion1=torch.cat((pro_seq,pro_3d_graph),dim=1)
        fusion = self.dropout(self.b(self.relu(self.fc2(fusion1))))
        return fusion


class FCLayer3(nn.Module):
    def __init__(self, in_size=128):
        super(FCLayer3, self).__init__()

        self.project_pro_seq = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

        )
        self.project_pro_graph = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

        )
        self.project_pro_3d = nn.Sequential(
            nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

        )
        self.gate=nn.Linear(128,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1024)
        self.b=nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.2)
        self.att=nn.Sequential(
            nn.Linear(128*3,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, pro_seq,pro_graph,pro_3d_graph):
        # x = self.project_x(pro_seq)
        # xt = self.project_xt(pro_graph)
        # st = self.project_st(pro_3d_graph)
        # a = torch.cat((x, xt, st), 1)
        # a = torch.softmax(a, dim=1)
        # emb = torch.stack([pro_seq, pro_graph, pro_3d_graph], dim=1)
        # a = a.unsqueeze(dim=2)  # Reshape attention scores
        # fused_emb = torch.sum(a * emb, dim=1)  # Weighted sum of embeddings
        # fusion = self.dropout(self.relu(self.b(self.fc2(fused_emb))))
        att1=self.project_pro_graph(pro_graph)
        att2=self.project_pro_3d(pro_3d_graph)
        att3=self.project_pro_seq(pro_seq)
        fusion=torch.cat((att3,att2,att1),1)
        fusion=self.att(fusion)




        return fusion
class FCLayer4(nn.Module):
    def __init__(self, in_size=128):
        super(FCLayer4, self).__init__()

        self.project_pro_graph = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),

        )
        self.project_pro_3d = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.gate=nn.Linear(8, 8)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(136, 128)
        self.b=nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.2)
    def forward(self, pro_seq,pro_graph,pro_3d_graph):
        x = self.project_pro_graph(pro_graph)
        xt = self.project_pro_3d(pro_3d_graph)
        att=self.gate(x)+self.gate(xt)
        att=torch.sigmoid(att)
        fusion=att*x+(1-att)*att
        fusion=self.relu(self.b(fusion))
        fusion = self.dropout(self.b(self.relu(self.fc1(fusion))))
        fusion1 = torch.cat((pro_seq, fusion), dim=1)
        fusion = self.dropout(self.b(self.relu(self.fc2(fusion1))))
        return fusion


class proGraphRepresentation(nn.Module):
    """
    This class represents protein graphs and performs graph-based learning on protein structures.
    """

    def __init__(self, num_features_pro, dropout, output_dim):
        """
        Initializes the protein graph representation model.

        Parameters:
            num_features_pro (int): The number of input features for each protein node.
            dropout (float): The dropout rate for regularization.
            output_dim (int): The output dimension of the final protein embedding.
        """
        super().__init__()

        # Define layers for protein graph convolution
        self.pro_conv = nn.ModuleList([])
        self.pro_conv.append(GCNConv(num_features_pro, num_features_pro * 4))
        self.pro_conv.append(
            GATConv(num_features_pro * 4, num_features_pro * 4, heads=4, dropout=dropout, concat=False))
        self.pro_conv.append(
            GATConv(num_features_pro * 4, num_features_pro * 4, heads=4, dropout=dropout, concat=False))

        self.pro_out_feats = num_features_pro * 4

        # Linear layers for protein sequence transformation
        self.pro_seq_fc1 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_seq_fc2 = nn.Linear(num_features_pro * 4, num_features_pro * 4)

        # Bias for protein sequence transformation
        self.pro_bias = nn.Parameter(torch.rand(1, num_features_pro * 4))
        torch.nn.init.uniform_(self.pro_bias, a=-0.2, b=0.2)

        # Global layer for protein features
        self.global_fc = Sequential(
            Linear(num_features_pro * 4, 1024),
            ReLU(),
            nn.Dropout(dropout),
            Linear(1024, output_dim),
            nn.Dropout(dropout)
        )

        self.relu = nn.ReLU()

    def forward(self, data):
        """
        Forward pass for the protein graph representation.

        Parameters:
            data: Data object containing graph structure and
        Forward pass for the protein graph representation.

        Parameters:
            data: Data object containing graph structure and protein node features.

        Returns:
            Tensor: The final protein graph-level representation.
        """
        # Get the protein input features
        x, edge_index, weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        n = x.size(0)

        # Apply each convolution layer
        for i in range(len(self.pro_conv)):
            if i == 0:
                xc = self.pro_conv[i](x, edge_index, weight)
            else:
                xc = self.pro_conv[i](x, edge_index)

            # Apply ReLU activation to intermediate layers
            if i < len(self.pro_conv) - 1:
                xc = self.relu(xc)

            # Skip connection for the first layer
            if i == 0:
                x = xc
                continue

            # Protein sequence transformation with gating mechanism
            pro_z = torch.sigmoid(
                self.pro_seq_fc1(xc) + self.pro_seq_fc2(x) + self.pro_bias.expand(n, self.pro_out_feats))

            # Combine the transformed features with the original features
            x = pro_z * xc + (1 - pro_z) * x

        # Apply global max pooling to aggregate node-level features into a graph-level feature
        x = gmp(x, batch)

        # Pass through the final global layer
        x = self.global_fc(x)
        return x



class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
                subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
                ptr=None, return_attn=False):
        #
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                         edge_attr=edge_attr, degree=degree,
                         subgraph_node_index=subgraph_node_index,
                         subgraph_edge_index=subgraph_edge_index,
                         subgraph_indicator_index=subgraph_indicator_index,
                         subgraph_edge_attr=subgraph_edge_attr,
                         ptr=ptr,
                         return_attn=return_attn
                         )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size=78, in_size_prot=33,num_class=2, d_model=128, num_heads=8,num_features_xc=92,
                 dim_feedforward=512, dropout=0.2, num_layers=1,
                 batch_norm=True, abs_pe=False, abs_pe_dim=0,
                 n_output=1, n_filters=32, embed_dim=128, num_features_xt=25, output_dim=128,
                 gnn_type="gcn", se="gnn", use_edge_attr=False, num_edge_features=4,
                 in_embed=False, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):
        super().__init__()

        # compound:gcn+Transformer
        # protein:cnn block+Transformer

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        self.embedding= nn.Linear(in_features=in_size,out_features=d_model,bias=False)#（78，128，false）


        self.use_edge_attr = use_edge_attr#边的属性,false
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        kwargs['edge_dim'] = None
        self.gnn_type = gnn_type#graph
        self.se = se#gnn
        #Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)#(128,8,512,0.0,true,graph,gnn,none)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)#(encoder_layer,4)
        self.global_pool = global_pool#mean
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool=='max':
            self.pooling=gnn.global_max_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool#true

        self.max_seq_len = max_seq_len#none
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )#((128,128),relu,(128,2))
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

        self.n_output = n_output#1




        #药物的1489
        # self.conlstm = nn.LSTM(256, 256, 1, dropout=0.2, bidirectional=True)
        self.com_fc1 = nn.Linear(128*2, 1024)
        self.com_bn1 = nn.BatchNorm1d(1024)
        self.com_fc2 = nn.Linear(1024, 512)
        self.com_bn2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)
        #对蛋白质序列：将蛋白质序列分割成两个部分，再使用卷积和交叉注意力机制提取特征

        self.p_embed = nn.Embedding(num_features_xt + 1, embed_dim)
        # target：512，1000 ，512，1000，128
        self.p_cnn1=nn.Conv1d(500,128,kernel_size=3,padding=1,stride=1)
        self.cnn_bn1=nn.BatchNorm1d(128)
        self.p_cnn2=nn.Conv1d(128,64,kernel_size=3,padding=1,stride=1)
        self.cnn_bn2 = nn.BatchNorm1d(64)
        self.p_cnn3 = nn.Conv1d(64, 16, kernel_size=8, padding=0, stride=1)
        self.cnn_bn3 = nn.BatchNorm1d(16)
        self.target_fc1=nn.Sequential(
            nn.Linear(121,128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.maxpool=nn.MaxPool1d(kernel_size=3,padding=1,stride=1)
        self.avgpool=nn.AvgPool1d(kernel_size=3,padding=1,stride=1)
        self.CAA=CAA_Block(128,8,1)
        self.CAA2=CAA_Block2()
        self.p_fc4 = nn.Linear(16 * 122, output_dim)
        self.p_bn4 = nn.BatchNorm1d(output_dim)

        #对蛋白质图进行处理

        # self.gin_conv1 = GINConv(nn.Sequential(nn.Linear(33, 128), nn.ReLU(),
        #                                        nn.Linear(128, 128)))
        self.gin_conv1=GCNConv(33,128)
        self.gin_conv2 = GCNConv(128, 128)

        # self.gin_conv2 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
        #                                         nn.Linear(64, 64)))
        # self.gin_conv3 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
        #                                        nn.Linear(64, 64)))
        # self.gin_conv4 = GINConv(nn.Sequential(nn.Linear(64,64), nn.ReLU(),
        #                                        nn.Linear(64, 64)))
        # self.gin_conv5 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
        #                                        nn.Linear(64, 64)))
        # self.pro_graph_conv1=GINConv(nn.Sequential(nn.Linear(33, 16), nn.ReLU(),
        #                                         nn.Linear(16, 16)))
        # self.pro_graph_conv2 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                         nn.Linear(16, 16)))
        # self.pro_graph_conv3 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                         nn.Linear(16, 16)))
        # self.pro_graph_con1=GATConv

        self.pro_graph_bn=nn.BatchNorm1d(128)
        self.pro_graph_linnear=nn.Linear(256,128)



        #对蛋白质口袋图进行神经训练

        # self.pro_3d_conv1 =GINConv(nn.Sequential(nn.Linear(3, 16), nn.ReLU(),
        #                                         nn.Linear(16, 16)))
        self.pro_3d_conv1=GCNConv(3, 16)
        self.pro_3d_conv2 = GCNConv(16, 16)
        # self.pro_3d_conv2 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                         nn.Linear(16, 16)))
        # self.pro_3d_conv3 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                           nn.Linear(16, 16)))
        # self.pro_3d_conv4 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                           nn.Linear(16, 16)))
        # self.pro_3d_conv5 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                           nn.Linear(16, 16)))

        # self.pro_3d_conv3 = GATConv(128, 128, heads=1, dropout=dropout, concat=False)
        self.pro_3d_bn = nn.BatchNorm1d(16)
        # self.pro_3d_gcn=GINConv(nn.Sequential(nn.Linear(3, 16), nn.ReLU(),
        #                                        nn.Linear(16, 16)))
        # self.pro_3d_gcn1 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                        nn.Linear(16, 16)))
        # self.pro_3d_gcn2 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                        nn.Linear(16, 16)))
        # self.pro_3d_gcn3 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                          nn.Linear(16, 16)))
        # self.pro_3d_gcn4 = GINConv(nn.Sequential(nn.Linear(16, 16), nn.ReLU(),
        #                                          nn.Linear(16, 16)))
        self.pro_graph3d_linnear = nn.Linear(32, 16)
        self.pro_3d_bn=nn.BatchNorm1d(16)
        self.pro_fusion=FCLayer3(in_size=128)

        # self.progra_encoder = proGraphRepresentation(33, dropout, 128)









    def forward(self, data_drug,data_pro,data_pro_3d):

        data_pro_seq = data_pro.pro_emb

        pro_s = data_pro_seq
        x, edge_index, batch, edge_attr = data_drug.x, data_drug.edge_index, data_drug.batch, data_drug.edge_attr


#对药物分子图进行处理
        output=x
        output=self.embedding(output)
        output = self.encoder(
            output,
            edge_index,
            None,
            edge_attr=edge_attr,
            degree=None,
            subgraph_node_index=None,
            subgraph_edge_index=None,
            subgraph_indicator_index=None,
            subgraph_edge_attr=None,
            ptr=data_drug.ptr,
            return_attn=None
        )
        output = self.pooling(output, data_drug.batch)

        #对蛋白质序列进行分析
        #对蛋白质序列进行分割，最后再使用深层次的CNN

        p_embed = self.p_embed(pro_s)

        # 第一个矩阵
        target_1 = p_embed[:, :500, :]

        # 第二个矩阵
        target_2 = p_embed[:, 500:1000, :]

        #每个矩阵大小为512*250*128
        #1.对每个矩阵进行cnn
        target_1=self.relu(self.cnn_bn1(self.p_cnn1(target_1)))
        target_1=self.relu(self.cnn_bn2(self.p_cnn2(target_1)))
        target_1=self.relu(self.cnn_bn3(self.p_cnn3(target_1)))
        target_1=self.target_fc1(target_1)
        query1=self.relu(self.maxpool(target_1)+self.avgpool(target_1))

        target_2 = self.relu(self.cnn_bn1(self.p_cnn1(target_2)))
        target_2 = self.relu(self.cnn_bn2(self.p_cnn2(target_2)))
        target_2 = self.relu(self.cnn_bn3(self.p_cnn3(target_2)))
        target_2=self.target_fc1(target_2)
        query2 = self.relu(self.maxpool(target_2)+self.avgpool(target_2))

        caa1=self.CAA(query2,target_1)
        caa2=self.CAA(query1,target_2)
        protein=self.CAA2(query1,caa1,query2,caa2)
        protein=protein.view(-1,16*122)
        protein=self.dropout(self.relu(self.p_bn4(self.p_fc4(protein))))


        #对蛋白质图进行提取提取特征
        x_pro, edge_index_pro, batch_pro = data_pro.x, data_pro.edge_index, data_pro.batch
        pro_graph = self.relu(self.gin_conv1(x_pro, data_pro.edge_index))
        pro_graph=self.pro_graph_bn(pro_graph)
        pro_graph = self.relu(self.gin_conv2(pro_graph, data_pro.edge_index))
        pro_graph = self.pro_graph_bn(pro_graph)

        pro_graph = torch.cat((global_add_pool(pro_graph, data_pro.batch),global_mean_pool(pro_graph, data_pro.batch)),dim=1)
        pro_graph= self.dropout(self.relu(self.pro_graph_linnear(pro_graph)))
        # pro_graph=self.progra_encoder(data_pro)




        # x_pro, edge_index_pro, batch_pro= data_pro.x, data_pro.edge_index, data_pro.batch
        # pro_graph = self.relu(self.pro_graph_bn(self.gin_conv1(x_pro, data_pro.edge_index)))
        # pro_graph = torch.cat((global_add_pool(pro_graph, data_pro.batch),global_mean_pool(pro_graph, data_pro.batch)),dim=1)
        # pro_graph= self.dropout(self.relu(self.pro_graph_linnear(pro_graph)))

        pro_3d_graph=self.relu(self.pro_3d_bn(self.pro_3d_conv1(data_pro_3d.x_3d,data_pro_3d.edge_index_3d)))
        pro_3d_graph = self.relu(self.pro_3d_bn(self.pro_3d_conv2(pro_3d_graph, data_pro_3d.edge_index_3d)))
        # pro_3d_graph = self.pro_3d_gcn1(pro_3d_graph, data_pro_3d.edge_index_3d)
        # pro_3d_graph = self.pro_3d_gcn2(pro_3d_graph, data_pro_3d.edge_index_3d)
        # pro_3d_graph = self.pro_3d_gcn3(pro_3d_graph, data_pro_3d.edge_index_3d)
        # pro_3d_graph = self.pro_3d_gcn4(pro_3d_graph, data_pro_3d.edge_index_3d)

        # pro_3d_graph=self.relu(self.pro_3d_bn(self.pro_3d_conv1(data_pro_3d.x_3d,data_pro_3d.edge_index_3d)))
        # pro_3d_graph = self.relu(self.pro_3d_bn(self.pro_3d_conv2(pro_3d_graph, data_pro_3d.edge_index_3d)))
        # pro_3d_graph = self.relu(self.pro_3d_bn(self.pro_3d_conv3(pro_3d_graph, data_pro_3d.edge_index_3d)))
        #
        # # pro_3d_graph=torch.cat((global_add_pool(pro_3d_graph, data_pro_3d.batch),global_mean_pool(pro_3d_graph, data_pro_3d.batch)),dim=1)
        pro_3d_graph=torch.cat((global_max_pool(pro_3d_graph, data_pro_3d.batch),global_add_pool(pro_3d_graph,data_pro_3d.batch)),dim=1)
        pro_3d_graph = self.dropout(self.relu(self.pro_3d_bn(self.pro_graph3d_linnear(pro_3d_graph))))#这个我改了，还没重新跑，要记得，1.26晚上23.56


        #将蛋白质图和蛋白质序列信息进行注意力特征结合
        pro_fusion=self.pro_fusion(protein,pro_graph,pro_3d_graph)







        con = torch.cat((output, pro_fusion), 1)
        xc = self.com_fc1(con)
        xc = self.relu(xc)
        xc = self.com_bn1(xc)
        xc = self.dropout(xc)
        xc = self.com_fc2(xc)
        xc = self.relu(xc)
        xc = self.com_bn2(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out





class CAA_Block2(nn.Module):
    def __init__(self):
        super(CAA_Block2, self).__init__()

        self.mlp=nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=7, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
    def forward(self, query1,target1,query2,target2):
        protein1=target1+query1
        protein2=target2+query2
        protein=torch.cat((protein1,protein2),dim=1)
        protein=self.mlp(protein)
        return protein



class CAA_Block(nn.Module):
    def __init__(self, d_model, n_head, nlayers, dropout=0.1, activation="relu"):
        super(CAA_Block, self).__init__()
        self.encoder = nn.ModuleList([CAA_Block_Layer(d_model, n_head, dropout, activation)
                                      for _ in range(nlayers)])
    def forward(self, q,kv, atten_mask=None):
        for layer in self.encoder:
            x = layer.forward(q,kv, atten_mask)
        return x


class CAA_Block_Layer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, activation="relu"):
        super().__init__()
        self.attn = CAttention(h=n_head, d_model=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)


        if activation == "relu":
            self.activation = F.relu
        if activation == "gelu":
            self.activation = F.gelu


    def forward(self, q,kv, atten_mask):

        # add & norm 1
        attn = self.dropout(self.attn(q, kv, kv, attn_mask=atten_mask))
        return attn
class CAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(CAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, atten = self.attention(query, key, value, attn_mask=attn_mask, dropout=self.dropout)#这个X就是公式中的Z，atten就是softmax中的那一坨内积

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)#view函数表示要重新定义矩阵的形状。
        return self.output_linear(x)




class ScaledDotProductAttention(nn.Module):#Attention(Q,K,V) = softmax(Q*Kt/sqrt(dk)) *V
    def forward(self, query, key, value, attn_mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)  # 保留位置为0的值，其他位置填充极小的数
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  # (batch, n_head, seqLen, dim)