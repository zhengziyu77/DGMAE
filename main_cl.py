import logging

import numpy as np
from tqdm import tqdm
import torch
import scipy.sparse as sp
from graphmae.utils import *
import torch.nn.functional as F
from graphmae.datasets.data_util import load_dataset
from graphmae.models import build_model,loss_func
from graphmae.models.loss_func import sce_loss
import dgl
from kmeans import *
#from edge_discriminator import *
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

from cluster import *

bias=1e-4
def gumbel_sampling(edges_weights_raw,temperature):
    eps = (bias - (1 - bias)) * torch.rand(edges_weights_raw.size()) + (1 - bias)
    gate_inputs = torch.log(eps) - torch.log(1 - eps)
    gate_inputs = gate_inputs.to(edges_weights_raw.device)
    gate_inputs = (gate_inputs + edges_weights_raw) / temperature
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
    edge = graph1.edges()
    src = edge[0].view(1,-1)
    dst = edge[1].view(1,-1)
    edge_index1 = torch.cat((src,dst), dim=0).to(torch.int64)

    degree = graph.in_degrees().cpu().numpy()#+graph.out_degrees()

    
    for epoch in epoch_iter:

        model.train()
 
        mask_rate=model._mask_rate
        perm = torch.randperm(num_nodes)
        num_mask_nodes = int(mask_rate * num_nodes)#
        mask_nodes = perm[: num_mask_nodes].to(device)
        keep_nodes = perm[num_mask_nodes: ].to(device)
        x_r1,loss1,loss_dict, encode_attn,edge_index= model(graph,x ,mask_nodes)
        

        encode_attn = torch.mean(encode_attn,dim=1).squeeze()
    
        enattn = torch.reshape(encode_attn, [-1]).to(device)
        #weights_lp = gumbel_sampling(enattn,0.05)
        weights_lp = torch.sigmoid(enattn)

        weights_hp = 1 - weights_lp

        h_edge,mask_edge,edge_w = drop_edge_weighted(edge_index,weights_hp,0.5,1.0)

        maskadj,attA = gen_normalized_adjs(h_edge,num_nodes)
        hx,L= heterophily_highfilter_sp(maskadj,x,1,I)
      
        enc_rep,all_hidden,_= model.encoder(graph,x, return_hidden=True)
      
        high_x = model.mlp(enc_rep)
      
        diff =(high_x -x_r1)
        high_loss = (sce_loss((diff)[keep_nodes],(hx)[keep_nodes],5).mean() )

        hetero_loss =  high_loss 
        a1 = 0.2
        a2 = 0.8
        loss = a1 * loss1 +a2 *hetero_loss#

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}:heter_loss:{hetero_loss.item():.4f}: recon_loss{loss1.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) %500== 0:
           
            nmi, ari, acc,f1= clustering(enc_rep, y, num_classes,0)
            print(f'Final result: nmi:{nmi:.4f}, ari: {ari:.4f},acc: {acc:.4f}, f1: {f1:.4f}' )  


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
    nmi_list = []
    ari_list =[]
    f1_list =[]
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
        y = graph.ndata["label"]

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

        enc_rep = model.embed(graph,x)

        for i in range(10):
            #set_random_seed(i)
            nmi, ari, acc,f1= clustering(enc_rep, y, num_classes,i)

            print(f'Final result: nmi:{nmi:.2f}, ari: {ari:.2f},acc: {acc:.2f}, f1: {f1:.2f}' )  

            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            f1_list.append(f1)


        if logger is not None:
            logger.finish()
  
        final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
        final_nmi, final_nmi_std = np.mean(nmi_list), np.std(nmi_list)
        final_ari, final_ari_std = np.mean(ari_list), np.std(ari_list)
        final_f1, final_f1_std = np.mean(f1_list), np.std(f1_list)
        print(f"# final_acc: {final_acc:.2f}±{final_acc_std:.2f}")
        print(f"# final_nmi: {final_nmi:.2f}±{final_nmi_std:.2f}")
        print(f"# final_ari: {final_ari:.2f}±{final_ari_std:.2f}")
        print(f"# final_f1: {final_f1:.2f}±{final_f1_std:.2f}")



# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs_cl.yml")
    print(args)
    main(args)
