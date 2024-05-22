import logging
import random

import numpy as np
from tqdm import tqdm
import torch
import scipy.sparse as sp
from graphmae.utils import *
import torch.nn.functional as F
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model
from graphmae.models.loss_func import sce_loss
import dgl
from kmeans import *

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

from draw import *
from cluster import *
bias=1e-10
def gumbel_sampling(edges_weights_raw,temperature):
    eps = (bias - (1 - bias)) * torch.rand(edges_weights_raw.size()) + (1 - bias)
    gate_inputs = torch.log(eps) - torch.log(1 - eps)
    gate_inputs = gate_inputs.to(edges_weights_raw.device)
    gate_inputs = (gate_inputs + edges_weights_raw) / temperature
    #gate_inputs = (edges_weights_raw) / temperature
    return torch.sigmoid(gate_inputs).squeeze()



def pretrain(dataset_name, model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    num_features = x.shape[1]
    num_nodes = x.shape[0]
    epoch_iter = tqdm(range(max_epoch))

    mask_nodes = None

    y = graph.ndata["label"]


    I = dsp.identity(shape=(num_nodes, num_nodes)).to(device)

    graph1 = dgl.remove_self_loop(graph)
    edge = graph.edges()
    src = edge[0].view(1,-1)
    dst = edge[1].view(1,-1)
    edge_index = torch.cat((src,dst), dim=0).to(torch.int64)
 
    degree = graph.in_degrees()#+graph.out_degrees()

    for epoch in epoch_iter:

        model.train()
        

        mask_rate=model._mask_rate


        perm = torch.randperm(num_nodes)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes].to(device)
        keep_nodes = perm[num_mask_nodes: ].to(device)
        enc_rep,all_hidden,encode_attn= model.encoder(graph,x, return_hidden=True)
        x_r1,loss1,loss_dict, encode_attn,edge_index= model(graph,x,mask_nodes)
        encode_attn = torch.mean(encode_attn,dim=1).squeeze()
    

        enattn = torch.reshape(encode_attn, [-1]).to(device)

        weights_lp = torch.sigmoid(enattn)
 
        weights_hp = 1 - weights_lp

        h_edge,mask_edge,edge_w = drop_edge_weighted(edge_index,weights_hp,0.3,0.7)
    
        maskadj,attA = gen_normalized_adjs(h_edge,num_nodes)
      
    
        high_x = model.mlp(enc_rep)
     
        hx,L= heterophily_highfilter_sp(maskadj,x,args.hop,I)
        diff =high_x - x_r1#
        high_loss = ( sce_loss((diff)[keep_nodes],(hx)[keep_nodes],4)).mean() 
        hetero_loss =  high_loss 
       
        a1 = 0.9
        a2 = 0.1
        loss = a1 * loss1 +a2 *hetero_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}:heter_loss:{hetero_loss.item():.4f}: recon_loss{loss1.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) %50000== 0:
            node_classification_evaluation(dataset_name,model,graph,x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # return best_model
    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler


    acc_list = []
    estp_acc_list = []


    #设置随机种子
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)
        
        graph, (num_features, num_classes) = load_dataset(dataset_name)#加载数据
        args.num_features = num_features
        args.num_nodes = graph.num_nodes()
        graph =graph.to(device)


        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None
   
        model = build_model(args)#建立模型
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)#建立优化器

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]#节点特征

        if not load_model:
            #训练
            model= pretrain(dataset_name,model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f,linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("./save_model/"+dataset_name+"_checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "./save_model/"+dataset_name+"_checkpoint.pt")
        
        model = model.to(device)
        
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(dataset_name, model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device,linear_prob)
 
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
   
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(np.array(estp_acc_list))

    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
