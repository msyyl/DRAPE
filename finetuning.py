# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from models import AdversarialNetwork
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve
from itertools import cycle
import config
from utility import edge_extract,Adversarial_loss,Entloss
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)

def multi_eval_epoch(model, loader, node_x, edge_index, drug, device):
    model.eval()
    total_loss = 0
    alpha = 0
    y_true, y_pred, y_mask, cancer_labels = [], [], [], []

    for x, y, mask, label in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        mask = (mask > 0)

        with torch.no_grad():
            _, yp, _ = model(x, alpha, node_x, edge_index)
            loss_mat = nn.BCEWithLogitsLoss()(yp, y.double())
            loss_mat = torch.where(
                mask, loss_mat,
                torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(mask)
            total_loss += loss

            y_true += y.cpu().detach().numpy().tolist()
            y_pred += yp.cpu().detach().numpy().tolist()
            y_mask += mask.cpu().detach().numpy().tolist()
            cancer_labels += label.cpu().detach().numpy().tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mask = np.array(y_mask)
    cancer_labels = np.array(cancer_labels)

    auc_list, aupr_list, acc_list, f1_list = [], [], [], []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1.0) > 0 and np.sum(y_true[:, i] == 0.0) > 0:
            is_valid = (y_mask[:, i] > 0)
            auc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
            aupr_list.append(auprc(y_true[is_valid, i], y_pred[is_valid, i]))
            f1_list.append(f1_score(y_true[is_valid, i], (y_pred[is_valid, i] >= 0.5).astype('int')))
            acc_list.append(accuracy_score(y_true[is_valid, i], (y_pred[is_valid, i] >= 0.5).astype('int')))
        else:
            print('{} is invalid'.format(i))

    all_results = [auc_list, aupr_list, acc_list, f1_list]

    return total_loss / len(loader), np.array(all_results), y_true, y_pred, y_mask, cancer_labels

def training(encoder, classifier, s_dataloader, t_dataloader, drug, prototype, params_str, **kwargs):
    da_network = AdversarialNetwork(encoder, classifier, len(drug)).to(kwargs['device'])
    optimizer = torch.optim.AdamW(da_network.parameters(), lr=kwargs['lr'])
    LossFunc_domain = nn.BCEWithLogitsLoss()
    LossFunc_class = nn.BCEWithLogitsLoss(reduction='none')
    best_loss_sum = np.inf

    node_x = torch.from_numpy(config.drug_feat.astype('float32'))
    node_x = node_x.to(kwargs['device'])
    edge_index = edge_extract(config.label_graph)
    edge_index = torch.from_numpy(edge_index.astype('int'))
    edge_index = edge_index.to(kwargs['device'])
    Entropy_loss = Entloss().to(kwargs['device'])

    print("\n============Domain adaptation for TCGA data================")
    for epoch in range(int(kwargs['uda_num_epochs'])):
        return_loss_sum = 0
        len_loader = min(len(s_dataloader), len(t_dataloader))
        da_network.train(True)
        optimizer.zero_grad()
        # Begining domain training
        for i, batch in enumerate(zip(t_dataloader, cycle(s_dataloader))):
            # Domain adapation loss
            p = float(i + epoch * len_loader) / int(kwargs['uda_num_epochs']) / len_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            s_x = batch[1][0].to(kwargs['device'])
            s_y = batch[1][1].to(kwargs['device'])
            s_mask = batch[1][2].to(kwargs['device'])
            t_x = batch[0][0].to(kwargs['device'])
            s_type = batch[1][3].to(kwargs['device'])
            t_type = batch[0][3].to(kwargs['device'])

            domain_s, s_yp, s_feat = da_network(s_x, alpha, node_x, edge_index)
            domain_t, t_yp, t_feat = da_network(t_x, alpha, node_x, edge_index)
            transfer_loss = Adversarial_loss(domain_s, domain_t, LossFunc_domain)
            
            # Multi-label (drug) classification loss
            loss_mat = LossFunc_class(s_yp, s_y.double())
            loss_mat = torch.where(s_mask>0, loss_mat,
                                   torch.zeros(loss_mat.shape).to(kwargs['device']))
            classifier_loss = torch.sum(loss_mat) / torch.sum(s_mask)

            geo_loss = Geometric_Loss(
                s_feat=s_feat,
                s_type=s_type,
                intra_weight=kwargs.get('lambda_intra', 1.0),
                inter_weight=kwargs.get('lambda_inter', 1.0)
            )
            drug_ent_loss = Entropy_loss(t_feat, prototype)

            # Returns the total loss
            total_loss = transfer_loss + classifier_loss+drug_ent_loss+geo_loss
            total_loss.backward()
            optimizer.step()
            return_loss_sum += total_loss.cpu().detach().item()
        
        if (best_loss_sum > return_loss_sum):
            best_loss_sum = return_loss_sum
            torch.save(da_network.state_dict(), os.path.join(kwargs['model_save_folder'], 'AdversarialNetwork.pt')) 
            
        if (epoch+1) % 50 == 0:
            print('Domain adapation training epoch = {}'.format(epoch+1))
            print('Domain adapation loss : {:.4f}'.format(return_loss_sum/len_loader))
    
    da_network.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'AdversarialNetwork.pt')))   
    
    return da_network
def Geometric_Loss(s_feat, s_type,intra_weight=1.0, inter_weight=1.0, align_weight=1.0):

    device = s_feat.device
    total_intra_loss = 0.0
    total_inter_loss = 0.0
    intra_count, inter_count = 0, 0

    num_s_classes = int(s_type.max().item()) + 1
    for c in range(num_s_classes):
        idx_s = (s_type == c)
        if idx_s.sum() < 2: continue
        feat_c = s_feat[idx_s]
        n = feat_c.size(0)
        cos_sim_matrix = F.cosine_similarity(feat_c.unsqueeze(1), feat_c.unsqueeze(0), dim=2)
        dist_matrix = 1.0 - cos_sim_matrix
        coeff = 1.0 / (n * (n - 1))
        total_intra_loss += coeff * dist_matrix.sum()
        intra_count += 1

    s_protos_list = []
    for c in range(num_s_classes):
        idx_s = (s_type == c)
        if idx_s.sum() > 0:
            s_protos_list.append(s_feat[idx_s].mean(dim=0, keepdim=True))

    if len(s_protos_list) > 1:
        s_protos = torch.cat(s_protos_list, dim=0)
        s_protos_norm = F.normalize(s_protos, p=2, dim=1)
        inter_sim_matrix = torch.matmul(s_protos_norm, s_protos_norm.T)
        mask = ~torch.eye(inter_sim_matrix.size(0), dtype=torch.bool, device=device)
        total_inter_loss += (inter_sim_matrix[mask].pow(2)).mean()
        inter_count += 1

    if intra_count > 0:
        total_intra_loss = total_intra_loss / intra_count
    if inter_count > 0:
        total_inter_loss = total_inter_loss / inter_count


    final_loss = intra_weight * total_intra_loss + inter_weight * total_inter_loss

    return final_loss

def testing(model, t_dataloader, drug, device):
    model.train(False)
    node_x = torch.from_numpy(config.drug_feat.astype('float32'))
    node_x = node_x.to(device)
    edge_index = edge_extract(config.label_graph)
    edge_index = torch.from_numpy(edge_index.astype('int'))
    edge_index = edge_index.to(device)
    test_loss, results, y_true, y_pred, y_mask,cancer_labels  = multi_eval_epoch(model=model,
                                          loader=t_dataloader,
                                          node_x=node_x,
                                          edge_index=edge_index,
                                          drug=drug,
                                          device=device)
    return test_loss, results, y_true, y_pred, y_mask,cancer_labels
