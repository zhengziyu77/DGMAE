from typing import Optional
from itertools import chain
from functools import partial
import torch
import torch.nn as nn
from .gin import GIN
from .gat import GAT
from .gcn import GCN

from .dot_gat import DotGAT

from .loss_func import sce_loss
from graphmae.utils import *
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
eps = 1e-8

def adj_to_torch_sparse(support):
    idx = torch.nonzero(support).T  # 这里需要转置一下
    data = support[idx[0], idx[1]]
    coo_support = torch.sparse_coo_tensor(idx, data, support.shape)
    return coo_support
def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
    dgl_graph.edata['w'] = values.detach().cuda()
    return dgl_graph
def mse_loss(x,y):
    criterion = torch.nn.MSELoss()
    return criterion(x,y)



def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )

    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
   
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            num_nodes:int,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,#掩码率
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,#删除边
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.alpha_l = alpha_l

        if encoder_type in ("gat", "dotgat"):#使用注意力模型
            enc_num_hidden = num_hidden // nhead #计算每个编码器注意力头的隐藏层输出维度
            enc_nhead = nhead#编码器的注意力头个数
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden #解码器的输入维度
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden #解码器隐藏层维度
    
        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )


        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))#编码前mask可训练参数，定义为（1，in_dim)的全0向量，in_dim等于节点特征维度


        if concat_hidden:#是否拼接隐藏层向量
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)#拼接后输入等于层数*单层隐藏层维度
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
           
        self.mlp =nn.Sequential(
            Linear(num_hidden, num_hidden,bias=False, weight_initializer='glorot'),
            nn.PReLU(),
            nn.Dropout(0.2),
            Linear(num_hidden, in_dim,bias=False, weight_initializer='glorot'))


    @property
    def output_hidden_dim(self):
        return self._output_hidden_size#返回输出隐藏层维度


    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, x, mask_rate,mask_nodes=None):
        num_nodes = g.num_nodes()
    
        if mask_nodes is None:
            perm = torch.randperm(num_nodes, device=x.device)
            num_mask_nodes = int(mask_rate * num_nodes)
            mask_nodes = perm[: num_mask_nodes]
            keep_nodes = perm[num_mask_nodes: ]
        else:
            mask_nodes = torch.squeeze(mask_nodes)
            num_mask_nodes = mask_nodes.shape[0]
            keep_nodes = 0


        if self._replace_rate > 0:
       
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]  
            out_x = x.clone()
            out_x[token_nodes] = 0.0  # token节点特征置为0
            out_x[noise_nodes] = x[noise_to_be_chosen]


        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0


        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x,mask_nodes=None):
        # ---- attribute reconstruction ----
        if self._encoder_type in ("gat","dotgat"):
            enc_rep, de_hidden, recon, loss,encode_attn,edge_index,mask_edge= self.mask_attr_prediction(g, x, mask_nodes)
        else:
            enc_rep, recon, loss= self.mask_attr_prediction(g, x, mask_nodes)
        loss_item = {"loss": loss.item()}
        if self._encoder_type in ("gat","dotgat"):
            return recon, loss, loss_item,encode_attn,edge_index
     
        elif self._encoder_type in ("gcn","gin"): 
            return enc_rep, recon, loss, loss_item
    
    def mask_attr_prediction(self, g, x, mask_nodes=None):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate, mask_nodes)
        mask_edges = 0
        if self._drop_edge_rate > 0:
        
            use_g,mask_edges = drop_edge(pre_use_g, self._drop_edge_rate,return_edges=True)

        else:
            use_g = pre_use_g
        #enc_rep= self.encoder(use_g, use_x)
        edge = use_g.edges()
        src = edge[0].view(1,-1)
        dst = edge[1].view(1,-1)
        edge_index = torch.cat((src,dst), dim=0).to(torch.int64)
        if self._encoder_type in ("mlp", "linear"):
            enc_rep = self.encoder(use_x)
        elif self._encoder_type in ("gat","dotgat"):
            enc_rep,  all_hidden,encode_attn= self.encoder(use_g, use_x, return_hidden=True)

        else:
            enc_rep,  all_hidden= self.encoder(use_g, use_x, return_hidden=True)

       
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        
        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
       
        if self._decoder_type not in ("mlp", "linear"):
            rep[mask_nodes] = 0


            

        if self._decoder_type in ("mlp", "linear") :
      
            recon = self.decoder(rep)
        elif self._decoder_type in ("gat") :
            recon,all_hidden,decoder_attn = self.decoder(pre_use_g, rep,return_hidden=True)
        
        else:
            recon= self.decoder(pre_use_g, rep)
        x_init = x[mask_nodes]

        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init) 
     
        if self._encoder_type in ("gat","dotgat") :
            return enc_rep,all_hidden, recon,loss,encode_attn,edge_index,mask_edges
        else:
            return enc_rep,recon,loss

    def embed(self, g, x):
        if self._encoder_type in ("mlp", "linear"):
            rep = self.encoder(x)
        elif self._encoder_type in ("gat","dotgat"):
            rep,_ = self.encoder(g, x)
        
        else:
            rep= self.encoder(g, x)
      
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
  