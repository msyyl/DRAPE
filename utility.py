# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
from rdkit.Chem import rdFingerprintGenerator, DataStructs,BRICS
from rdkit import Chem
import pandas as pd
import re
import os
import torch.nn as nn
import config
device = config.CUDA_ID


def edge_extract(adj_mat):
    # Converting the adjacency matrix to pytorch format, including edges and their corresponding weights 
    x, y = np.where(adj_mat != 1)
    adj_index = np.vstack((x, y))
    return adj_index

def wrap_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])
    return aux_dict

def make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])


class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def _download(self):
        pass

    def _process(self):
        pass
#       if not os.path.exists(self.processed_dir):
#          os.makedirs(self.processed_dir)

    def process(self, graphs_dict):
        data_list = []
        for sample in graphs_dict:
            features, edge_index = sample[0], sample[1]
            GraphData = DATA.Data(x=features, edge_index=edge_index)
            data_list.append(GraphData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
class FP:
    def __init__(self, fp, names):
        self.fp = fp
        self.names = names

    def __str__(self):
        return "%d bit FP" % len(self.fp)

    def __len__(self):
        return len(self.fp)
def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA.to(device)
def get_cfps(mol, radius=2, nBits=512, useFeatures=False, counts=False, dtype=np.float32):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nBits,
        includeRedundantEnvironments=True,
        countSimulation=False
    )

    arr = np.zeros((nBits,), dtype=dtype)

    if counts is True:
        fp_info = gen.GetFingerprintInfo(mol)

        bit_counts = np.zeros(nBits, dtype=dtype)
        for _, _, bitId in fp_info:
            bit_counts[bitId] += 1

        return FP(bit_counts, range(nBits))

    else:
        fp = gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return FP(arr, range(nBits))
def get_Morgan(smiles):
    m = Chem.MolFromSmiles(smiles)
    Finger = get_cfps(m)
    fp = Finger.fp
    fp = fp.tolist()
    return fp
def BRICS_GetMolFrags(smi):
    mol = Chem.MolFromSmiles(smi)
    smarts = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smarts)
    #---mol Decompose
    sub_smi = BRICS.BRICSDecompose(mol)
    sub_smi = [re.sub(r'\[\d+\*\]','*',item) for item in sub_smi]
    return sub_smi, smarts
def process_drug_features(
        drug_smiles_path: str,
        drug_list: list,
        max_frags: int = 20,
        frag_dim: int = 512,
        fp_size: int = 64,
        device: str = 'cuda:0'
):

    SMILES = pd.read_csv(drug_smiles_path, header=0, index_col=0) #(178,2)
    SMILES.index = SMILES.index.map(str.lower)

    drugs_to_keep = [item.lower() for item in drug_list] #(9)
    smis = SMILES.loc[drugs_to_keep]['Isosmiles'].values

    drug_subfeat_list = []
    drug_fragments = {}
    SMARTS = []
    masks = []

    for smi in smis:
        sub_smi, sm = BRICS_GetMolFrags(smi)
        sub_features = np.array([get_Morgan(item) for item in sub_smi])
        current_len = len(sub_features)
        mask1 = np.ones(current_len).reshape(1, -1)
        mask2 = np.zeros(max_frags - current_len).reshape(1, -1)
        masks.append(np.concatenate((mask1, mask2), axis=1))

        if current_len < max_frags:
            padding = np.zeros((max_frags - current_len, frag_dim))
            sub_features = np.concatenate((sub_features, padding), axis=0)

        drug_subfeat_list.append(sub_features)
        SMARTS.append(sm)
        drug_fragments[str(smi)] = sub_smi

    drug_subfeat_np = np.array(drug_subfeat_list, dtype=np.float32)
    drug_e = torch.tensor(drug_subfeat_np, device=device)  # Shape: (N, max_frags, frag_dim)

    drug_mask = torch.tensor(np.array(masks), device=device)
    drug_mask = drug_mask.squeeze(dim=1)  # Shape: (N, max_frags)

    mol_list = [Chem.MolFromSmiles(x) for x in list(SMILES.loc[drugs_to_keep]['Isosmiles'])]
    fp_list = np.array([Chem.RDKFingerprint(x, fpSize=fp_size) for x in mol_list])

    return fp_list,drug_e,drug_mask


def Adversarial_loss(domain_s, domain_t, BCEloss):
    # Initialize the pseudo-labels of the domain confrontation
    domain_labels = torch.cat((domain_s, domain_t), dim=0)
    bs_s = domain_s.size(0)
    bs_t = domain_t.size(0)
    target_labels = torch.from_numpy(np.array([[0]] * bs_s + [[1]] * bs_t).astype('float32')).to(device)
    domain_loss = BCEloss(domain_labels, target_labels)

    return domain_loss


class Entloss(nn.Module):

    def __init__(self, EPS=1e-8):
        super(Entloss, self).__init__()
        self.EPS = EPS

    def forward(self, t_feat, prototype):
        s_proto_ids, s_proto_feats = prototype
        drug_proto_feats = s_proto_feats.to(t_feat.device)

        n_protos = drug_proto_feats.shape[0]
        assert n_protos % 2 == 0, \
            f"Expected even number of drug prototypes, but got {n_protos}!"

        t_feat_norm = torch.nn.functional.normalize(t_feat, p=2, dim=1)
        drug_proto_norm = torch.nn.functional.normalize(drug_proto_feats, p=2, dim=1)
        sim_matrix = torch.matmul(t_feat_norm, drug_proto_norm.t())  # (batch_size, n_protos)

        n_drugs = n_protos // 2
        sim_matrix = sim_matrix.view(-1, n_drugs, 2)

        sim_sensitive = sim_matrix[:, :, 1]  # (batch_size, n_drugs)
        sim_resistant = sim_matrix[:, :, 0]  # (batch_size, n_drugs)

        pseudo_sensitivity = torch.sigmoid(sim_sensitive - sim_resistant)

        entropy = - (
                pseudo_sensitivity * torch.log(pseudo_sensitivity + self.EPS) +
                (1 - pseudo_sensitivity) * torch.log(1 - pseudo_sensitivity + self.EPS)
        )

        drug_refine_loss = entropy.mean()
        return drug_refine_loss