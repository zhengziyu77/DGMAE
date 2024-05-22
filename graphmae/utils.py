import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np
import dgl
import torch
import torch.nn as nn
from torch import optim as optim
#from tensorboardX import SummaryWriter

import scipy.sparse as sp
from torch_geometric.utils import degree
from torch_geometric.utils import to_undirected
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import dgl.sparse as dsp
from torch_sparse import SparseTensor
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument('--num_nodes', type=int, default=None)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=False)
    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true",default=False)
    parser.add_argument("--use_cfg", action="store_true", default=True)
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.05, help='diffusion')
    parser.add_argument('--n_input', type=int, default=500)



    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")

    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer



def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
def drop_edge(graph, drop_rate,return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]#保留边的起始节点
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask].to(torch.long)#掩码边
    ddst = dst[~edge_mask].to(torch.long)

    if return_edges:
        return ng,(dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

# ------ logging ------


class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
    dgl_graph.edata['w'] = values.detach().cuda()
    dgl_graph = dgl.add_self_loop(dgl_graph)
    return dgl_graph
def adj_to_torch_sparse(support):
    idx = torch.nonzero(support).T  # 这里需要转置一下
    data = support[idx[0], idx[1]]
    coo_support = torch.sparse_coo_tensor(idx, data, support.shape)
    return coo_support
def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx
def adj_to_edge_index(adj):
    edge_index = torch.nonzero(adj).T
    return edge_index

def get_adj_from_edges(edges, weights, nnodes):
    adj = torch.zeros(nnodes, nnodes).cuda()
    adj[edges[0], edges[1]] = weights
    return adj
def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)
def adj_to_coo(adj):
    idx = torch.nonzero(adj).T  # 这里需要转置一下
    data = adj[idx[0], idx[1]]
    coo_support = torch.sparse_coo_tensor(idx, data, adj.shape)
    return coo_support

def hop_features(x, init_adj, k=2):
    
    for i in range(k):
        x = torch.mm(init_adj,x)
    return x


def heterophily_highfilter(norm_adj,x,t):
    #A = graph.adjacency_matrix(transpose=True).to(x.device)
    #init_adj =torch.FloatTensor(sp.coo_matrix((A.cpu()._values(), A.cpu()._indices()), shape=(x.shape[0], x.shape[0])).todense()).to(x.device)

    #normadj = norm_adj(init_adj).to(x.device)#低通
    I = torch.eye(norm_adj.shape[0]).to(x.device)
    L = I-norm_adj
    high_x = hop_features(x,L,t)
    return high_x

def heterophily_highfilter_sp(norm_adj,x,t,I):
    #A = graph.adjacency_matrix(transpose=True).to(x.device)
    #init_adj =torch.FloatTensor(sp.coo_matrix((A.cpu()._values(), A.cpu()._indices()), shape=(x.shape[0], x.shape[0])).todense()).to(x.device)

    #normadj = norm_adj(init_adj).to(x.device)#低通
    #I = torch.eye(norm_adj.shape[0]).to(norm_adj.device)
    sparse =True
    if sparse:
        N= x.shape[0]
        norm_coo = norm_adj.coo()
        norm_adj = dsp.from_coo(row = norm_coo[0], col = norm_coo[1], val =norm_coo[2], shape=(N,N))
        L = dsp.sub(I,norm_adj)

    else:
        L = I-(norm_adj.to_dense())
        L = adj_to_torch_sparse(L)
        #L = L.to(x.device)
    x = x.to(I.device)
    
    for i in range(t):
        x = dsp.matmul(L,x)
    return x.to(I.device),L.to(I.device)


def edge_index_to_adjacency_matrix(edge_index, num_nodes):  
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵  
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)  
      
    # 使用索引广播机制，一次性将边索引映射到邻接矩阵的相应位置上  
    adjacency_matrix[edge_index[0], edge_index[1]] = 1  
    adjacency_matrix[edge_index[1], edge_index[0]] = 1  
      
    return adjacency_matrix

def gen_normalized_adjs(edge_index, N):
    """
    returns the normalized adjacency matrix
    """
    row, col = edge_index

    adj = SparseTensor(row=row, col=col,sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)

    D_isqrt = deg.pow(-0.5)

    D_isqrt[D_isqrt == float("inf")] = 0
    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)

    #DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1) * adj
    #AD = adj * D_isqrt.view(1, -1) * D_isqrt.view(1, -1)

    return DAD,adj

def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1-edge_weights).to(torch.bool)

    return edge_index[:, (~sel_mask)],edge_index[:, (sel_mask)],edge_weights[~sel_mask]
