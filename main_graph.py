import logging
from tqdm import tqdm
import numpy as np

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import torch.nn.functional as F
from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs
)
from graphmae.utils import *
import scipy.sparse as sp
from graphmae.datasets.data_util import load_graph_classification_dataset
from graphmae.models import build_model
def sce_loss(x, y, alpha=3):

    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x*y).sum(dim=-1)).pow_(alpha)
    return loss

def graph_classification_evaluation(model, pooler, dataloader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_g, labels) in enumerate(dataloader):
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            out = model.embed(batch_g, feat)
            out = pooler(batch_g, out)

            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
    return test_f1


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std



def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob=True, logger=None):
    train_loader, eval_loader = dataloaders


    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()

        loss_list = []
        for batch in train_loader:
            batch_g, _ = batch
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            num_features = feat.shape[1]
            A = batch_g.adjacency_matrix(transpose=True).to(device)
            init_adj =torch.FloatTensor(sp.coo_matrix((A.cpu()._values(), A.cpu()._indices()), shape=(feat.shape[0], feat.shape[0])).todense()).to(device)
            edge = batch_g.edges()
            src = edge[0].view(1,-1)
            dst = edge[1].view(1,-1)
            edge_index = torch.cat((src,dst), dim=0).to(torch.int64)
     
            model.train()
            mask_rate = model._mask_rate
            num_nodes = feat.shape[0]
            perm = torch.randperm(num_nodes, device=feat.device)
            num_mask_nodes = int(mask_rate * num_nodes)
            mask_nodes = perm[: num_mask_nodes]
            keep_nodes = perm[num_mask_nodes: ]
    
            enc_rep1,x_r1,loss1, loss_dict= model(batch_g, feat, mask_nodes)

            maskadj,attA = gen_normalized_adjs(edge_index,num_nodes, x_r1)

            I = dsp.identity(shape=(num_nodes, num_nodes)).to(device)
         

            hx,L= heterophily_highfilter_sp(maskadj,feat,1,I)

            enc_rep= model.encoder(batch_g,feat)


            high_x = model.mlp(enc_rep)

            diff =(high_x - x_r1)
        
            high_loss = sce_loss(diff[keep_nodes],hx[keep_nodes],2).mean()
            a1 =0.7
            a2 = 0.3
            loss = a1 * loss1 + a2 * high_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)
        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    return model

            
def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.FloatTensor(labels)


    return batch_g, labels


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
    pooling = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size
    args.num_nodes = 0

    graphs, (num_features, num_classes) = load_graph_classification_dataset(dataset_name, deg4feat=deg4feat)
    args.num_features = num_features

    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)
    
    train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError

    acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        if not load_model:
    
            model = pretrain(model, pooler, (train_loader, eval_loader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob,  logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()
        test_f1 = graph_classification_evaluation(model, pooler, eval_loader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False)
        acc_list.append(test_f1)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{np.array(acc_list).std():.4f}")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs_g.yml")
    print(args)
    main(args)
