import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from graphmae.utils import create_optimizer, accuracy
import numpy as np
from graphmae.utils import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score,average_precision_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
def eval_test_mode(embedding, labels, train_mask, val_mask, test_mask):

    X = embedding.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy()
    X = normalize(X, norm='l2')

    X_train = X[train_mask.cpu()]
    X_val = X[val_mask.cpu()]
    X_test = X[test_mask.cpu()]
    y_train = Y[train_mask.cpu()]
    y_val = Y[val_mask.cpu()]
    y_test = Y[test_mask.cpu()]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    y_pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)

    return acc_test * 100, acc_val * 100
def node_classification_evaluation(dataset_name,model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
  
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph, x.to(device))
            #x = x.detach()
            in_feat = x.shape[1]
        encoder = LogisticRegression1(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc = linear_probing_for_transductive_node_classiifcation(dataset_name,encoder, graph, x,num_classes, lr_f, weight_decay_f,optimizer_f, max_epoch_f, device, mute)


    return final_acc, estp_acc


def linear_probing_for_transductive_node_classiifcation(dataset_name, model, graph,feat, num_classes,lr_f, weight_decay_f,optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    graph = graph.to(device)

    x = feat.to(device)
    test_acc=0
    test_auc = 0
    estp_test_acc=0
    estp_test_auc=0
    if dataset_name in ("cora", "citeseer", "pubmed","syn","wiki","dee","photo","computer","phy","cs","ogbn-arxiv","arxiv-years","twitch_gamer","flicker","corafull","syn_cora"):
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]
    
    #train_mask = graph.ndata["train_mask"]
    #val_mask = graph.ndata["val_mask"]
    #test_mask = graph.ndata["test_mask"]
    #highhomo_mask = graph.ndata["highhomo_mask"]
    #lowhomo_mask = graph.ndata["lowhomo_mask"]
        labels = graph.ndata["label"]



        best_val_acc = 0
        best_val_epoch = 0
        best_model = None


        if not mute:
            epoch_iter = tqdm(range(max_epoch))
        else:
            epoch_iter = range(max_epoch)

        for epoch in epoch_iter:
            model.train()
            out = model(graph, x)

            loss = criterion(out[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pred = model(graph, x)
                val_acc = accuracy(pred[val_mask], labels[val_mask])
                val_loss = criterion(pred[val_mask], labels[val_mask])
                test_acc = accuracy(pred[test_mask], labels[test_mask])
                test_loss = criterion(pred[test_mask], labels[test_mask])

                #highhomo_test_acc = accuracy(pred[highhomo_mask],labels[highhomo_mask])
                #lowhomo_test_acc = accuracy(pred[lowhomo_mask], labels[lowhomo_mask])
            
            if val_acc >= best_val_acc:
            #if test_acc >= best_val_acc:
                best_val_acc = test_acc
                best_val_epoch = epoch
                best_model = copy.deepcopy(model)

            if not mute:
                epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")
                #epoch_iter.set_description(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")

        best_model.eval()
        with torch.no_grad():
            pred = best_model(graph, x)
            estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        if mute:
            print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
            #print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, AP:{AP:.4f}, AUC:{auc:.4f},  Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

            #print(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")

        else:
            print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
            #print(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")
    elif dataset_name in ("mines","quest","toloker"):
        test_idx_auc=[] 
        estp_idx_auc=[]
        best_idx_auc=[]
        idxs = graph.ndata["train_mask"].shape[1]


        for idx in range(0,idxs):

            train_mask = graph.ndata["train_mask"][:,idx]
            val_mask = graph.ndata["val_mask"][:,idx]
            test_mask = graph.ndata["test_mask"][:,idx]
            labels = graph.ndata["label"]
            best_val_auc = 0
            best_val_epoch = 0
            best_model = None
            if not mute:
                epoch_iter = tqdm(range(max_epoch))
            else:
                epoch_iter = range(max_epoch)
            model= LogisticRegression1(feat.shape[1], 1)   
            model.to(device)
            optimizer = create_optimizer("adam", model, lr_f, weight_decay_f)
            #acc_test, acc_val = eval_test_mode(x,labels,train_mask,val_mask,test_mask)
            #print(acc_test)
            y_true = labels.squeeze().long()
            for epoch in epoch_iter:
                model.train()
                out = model(graph, x).squeeze(1)
      
                loss = bce(out[train_mask], y_true[train_mask].float())
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    pred = model(graph, x).squeeze(1)
                    val_auc = roc_auc_score(y_true[val_mask].cpu().numpy(),
                                        pred[val_mask].cpu().numpy()).item()
                    val_loss = bce(pred[val_mask], y_true[val_mask].float())
  
 
                    #y_pred =  pred[test_mask].max(1)[1].type_as(labels)
                    test_loss = bce(pred[test_mask], y_true[test_mask].float())
                    #highhomo_test_acc = accuracy(pred[highhomo_mask],labels[highhomo_mask])
                    #lowhomo_test_acc = accuracy(pred[lowhomo_mask], labels[lowhomo_mask])
                
                if val_auc >= best_val_auc:
                #if test_acc >=best_val_acc:
                    best_val_auc = val_auc
                    best_val_epoch = epoch
                    best_model = copy.deepcopy(model)
                    best_test_auc = test_auc

                if not mute:
                    epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_auc:{val_auc}, test_loss:{test_loss.item(): .4f}, test_auc:{test_auc: .4f}")
                    #epoch_iter.set_description(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")

            best_model.eval()
            with torch.no_grad():
                pred = best_model(graph, x).squeeze(1)
                estp_test_auc =roc_auc_score(y_true[test_mask].cpu().numpy(),
                                        pred[test_mask].cpu().numpy()).item()
                test_auc = roc_auc_score(y_true[test_mask].cpu().numpy(),
                                        pred[test_mask].cpu().numpy()).item()
                   
              
                auc = roc_auc_score(y_true[test_mask].cpu().numpy(),y_score=pred[test_mask].cpu().numpy()).item()
                #AP = average_precision_score(y_true[test_mask].cpu().numpy(),y_pred[test_mask].cpu().numpy(), average='macro')
                print(f"---  current split: {idx:.1f}, early-stopping-TestAUU: {estp_test_auc:.4f},AUC: {auc:.4f}")
            best_idx_auc.append(best_test_auc)#保存对于最好val时的测试结果
            test_idx_auc.append(test_auc)
            estp_idx_auc.append(estp_test_auc)
        test_auc, test_auc_std = np.mean(test_idx_auc), np.std(test_idx_auc)
        estp_test_auc, estp_auc_std = np.mean(estp_idx_auc), np.std(estp_idx_auc)
        val_test_auc, val_test_std = np.mean(best_idx_auc), np.std(best_idx_auc)

        if mute:
            print(f"# IGNORE: --- TestAuc: {test_auc:.4f}±{test_auc_std:.4f}, early-stopping-TestAuc: {estp_test_auc:.4f}±{estp_auc_std:.4f}, Best ValAuc: {best_val_auc:.4f} in epoch {best_val_epoch} --- ")
            #print(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")
            print(f"# IGNORE: --- TestAuc: {test_auc:.4f}, early-stopping-TestAuc: {estp_test_auc:.4f},  AUC:{auc:.4f},  Best ValAuc: {best_val_auc:.4f} in epoch {best_val_epoch} --- ")

        else:
            print(f"---  TestAuc: {test_auc*100:.2f}±{test_auc_std*100:.2f}, early-stopping-TestAuc: {estp_test_auc*100:.2f}±{estp_auc_std*100:.2f}, Best ValAuc: {best_val_auc*100:.2f} in epoch {best_val_epoch} --- ")
            #print(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")
    elif dataset_name in ("inj_cora"):
        test_idx_auc=[] 
        estp_idx_auc=[]
        best_idx_auc=[]



        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]
        labels = graph.ndata["label"]
        best_val_auc = 0
        best_val_epoch = 0
        best_model = None
        idx=0
        if not mute:
            epoch_iter = tqdm(range(max_epoch))
        else:
            epoch_iter = range(max_epoch)
        model= LogisticRegression1(feat.shape[1], 1)   
        model.to(device)
        optimizer = create_optimizer("adam", model, lr_f, weight_decay_f)
        #acc_test, acc_val = eval_test_mode(x,labels,train_mask,val_mask,test_mask)
        #print(acc_test)
        y_true = labels.squeeze().bool()
        for epoch in epoch_iter:
            model.train()
            out = model(graph, x).squeeze(1)
    
            loss = bce(out[train_mask], y_true[train_mask].float())
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pred = model(graph, x).squeeze(1)
                val_auc = roc_auc_score(y_true[val_mask].cpu().numpy(),
                                    pred[val_mask].cpu().numpy()).item()
                val_loss = bce(pred[val_mask], y_true[val_mask].float())


                #y_pred =  pred[test_mask].max(1)[1].type_as(labels)
                test_loss = bce(pred[test_mask], y_true[test_mask].float())
                #highhomo_test_acc = accuracy(pred[highhomo_mask],labels[highhomo_mask])
                #lowhomo_test_acc = accuracy(pred[lowhomo_mask], labels[lowhomo_mask])
            
            if val_auc >= best_val_auc:
            #if test_acc >=best_val_acc:
                best_val_auc = val_auc
                best_val_epoch = epoch
                best_model = copy.deepcopy(model)
                best_test_auc = test_auc

            if not mute:
                epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_auc:{val_auc}, test_loss:{test_loss.item(): .4f}, test_auc:{test_auc: .4f}")
                #epoch_iter.set_description(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")

            best_model.eval()
            with torch.no_grad():
                pred = best_model(graph, x).squeeze(1)
                estp_test_auc =roc_auc_score(y_true[test_mask].cpu().numpy(),
                                        pred[test_mask].cpu().numpy()).item()
                test_auc = roc_auc_score(y_true[test_mask].cpu().numpy(),
                                        pred[test_mask].cpu().numpy()).item()
                   
              
                auc = roc_auc_score(y_true[test_mask].cpu().numpy(),y_score=pred[test_mask].cpu().numpy(),multi_class='ovo').item()
                #AP = average_precision_score(y_true[test_mask].cpu().numpy(),y_pred[test_mask].cpu().numpy(), average='macro')
            best_idx_auc.append(best_test_auc)#保存对于最好val时的测试结果
            test_idx_auc.append(test_auc)
            estp_idx_auc.append(estp_test_auc)
        print(f"---  current split: {idx:.1f}, early-stopping-TestAUU: {estp_test_auc:.4f},AUC: {auc:.4f}")
        test_auc, test_auc_std = np.mean(test_idx_auc), np.std(test_idx_auc)
        estp_test_auc, estp_auc_std = np.mean(estp_idx_auc), np.std(estp_idx_auc)
        val_test_auc, val_test_std = np.mean(best_idx_auc), np.std(best_idx_auc)

        if mute:
            print(f"# IGNORE: --- TestAuc: {test_auc:.4f}±{test_auc_std:.4f}, early-stopping-TestAuc: {estp_test_auc:.4f}±{estp_auc_std:.4f}, Best ValAuc: {best_val_auc:.4f} in epoch {best_val_epoch} --- ")
            #print(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")
            print(f"# IGNORE: --- TestAuc: {test_auc:.4f}, early-stopping-TestAuc: {estp_test_auc:.4f},  AUC:{auc:.4f},  Best ValAuc: {best_val_auc:.4f} in epoch {best_val_epoch} --- ")

        else:
            print(f"---  TestAuc: {test_auc:.4f}±{test_auc_std:.4f}, early-stopping-TestAuc: {estp_test_auc:.4f}±{estp_auc_std:.4f}, Best ValAuc: {best_val_auc:.4f} in epoch {best_val_epoch} --- ")
            #print(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")

    else:
        test_idx_acc=[] 
        estp_idx_acc=[]
        best_idx_acc=[]
        idxs = graph.ndata["train_mask"].shape[1]


        for idx in range(0,idxs):

            train_mask = graph.ndata["train_mask"][:,idx]
            val_mask = graph.ndata["val_mask"][:,idx]
            test_mask = graph.ndata["test_mask"][:,idx]
            labels = graph.ndata["label"]
            best_val_acc = 0
            best_val_epoch = 0
            best_model = None
            if not mute:
                epoch_iter = tqdm(range(max_epoch))
            else:
                epoch_iter = range(max_epoch)
            model= LogisticRegression1(feat.shape[1], num_classes)   
            model.to(device)
            optimizer = create_optimizer("adam", model, lr_f, weight_decay_f)
            #acc_test, acc_val = eval_test_mode(x,labels,train_mask,val_mask,test_mask)
            #print(acc_test)

            for epoch in epoch_iter:
                model.train()
                out = model(graph, x)

                loss = criterion(out[train_mask], labels[train_mask])
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    pred = model(graph, x)
                    val_acc = accuracy(pred[val_mask], labels[val_mask])
                    val_loss = criterion(pred[val_mask], labels[val_mask])
  
 
                    #y_pred =  pred[test_mask].max(1)[1].type_as(labels)
                    test_loss = criterion(pred[test_mask], labels[test_mask])
                    #highhomo_test_acc = accuracy(pred[highhomo_mask],labels[highhomo_mask])
                    #lowhomo_test_acc = accuracy(pred[lowhomo_mask], labels[lowhomo_mask])
                
                if val_acc >= best_val_acc:
                #if test_acc >=best_val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    best_model = copy.deepcopy(model)
                    best_test_acc = test_acc

                if not mute:
                    epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")
                    #epoch_iter.set_description(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")

            best_model.eval()
            with torch.no_grad():
                pred = best_model(graph, x)
                estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
                test_acc = accuracy(pred[test_mask], labels[test_mask])
                #print(pred[test_mask].shape)
                
                print(f"---  current split: {idx:.1f}, early-stopping-TestAcc: {estp_test_acc:.4f}, val_test_acc: {best_test_acc:.4f}")
            best_idx_acc.append(best_test_acc)#保存对于最好val时的测试结果
            test_idx_acc.append(test_acc)
            estp_idx_acc.append(estp_test_acc)
        test_acc, test_acc_std = np.mean(test_idx_acc), np.std(test_idx_acc)
        estp_test_acc, estp_acc_std = np.mean(estp_idx_acc), np.std(estp_idx_acc)
        val_test_acc, val_test_std = np.mean(best_idx_acc), np.std(best_idx_acc)

        if mute:
            print(f"# IGNORE: --- TestAcc: {test_acc:.4f}±{test_acc_std:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}±{estp_acc_std:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
            #print(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")
            #print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f},  AUC:{auc:.4f},  Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

        else:
            print(f"---  TestAcc: {test_acc:.4f}±{test_acc_std:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}±{estp_acc_std:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
            #print(f"# highhomo_test_acc: {highhomo_test_acc: .4f}, lowhomo_test_acc: {lowhomo_test_acc: .4f}")


        # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc


def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    if len(labels.shape) > 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    train_mask, val_mask, test_mask = mask

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)  

        best_val_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()

  
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")

    return test_acc, estp_test_acc


class LogisticRegression1(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits

