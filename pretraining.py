import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import torch
import numpy as np
from models import ShareMLP,PREMODEL
from itertools import chain
from itertools import cycle
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')

def eval_epoch(model, loader, device):
    avg_loss = 0
    for x_batch in loader:
        x_batch, x_labels = x_batch[0].to(device), x_batch[1].to(device)
        with torch.no_grad():
            inputs, recon, z,s_latent,s_latent_ = model(x_batch, x_labels)
            loss = model.loss_function(inputs, recon, z,s_latent,s_latent_)
            avg_loss += loss.cpu().detach().item() / x_batch.size(0)
    return avg_loss

def ae_train_step(model_s, model_t, s_batch, t_batch, device, optimizer):
    s_x, s_labels = s_batch[0].to(device), s_batch[1].to(device)
    t_x, t_labels = t_batch[0].to(device), t_batch[1].to(device)
    input_s, recon_s, z_s,s_latent,s_latent_ = model_s(s_x, s_labels)
    input_t, recon_t, z_t,t_latent,t_latent_ = model_t(t_x, t_labels)
    s_loss = model_s.loss_function(input_s, recon_s, z_s,s_latent,s_latent_)
    t_loss = model_t.loss_function(input_t, recon_t, z_t,t_latent,t_latent_)
    loss = s_loss + t_loss
    loss.backward()
    optimizer.step()
    return loss.cpu().detach().item() / s_x.size(0)

def training(s_dataloaders, t_dataloaders, **kwargs):
    s_train = s_dataloaders[0]
    s_test = s_dataloaders[1]
    t_train = t_dataloaders[0]
    t_test = t_dataloaders[1]
    
    shared_encoder = ShareMLP(input_dim = kwargs['input_dim'],
                         output_dim = kwargs['latent_dim'],
                         hidden_dims = kwargs['encoder_hidden_dims'],
                         drop = kwargs['drop']).to(kwargs['device'])   
    shared_decoder = ShareMLP(input_dim = 2 * kwargs['latent_dim'],
                         output_dim = kwargs['input_dim'],
                         hidden_dims = kwargs['decoder_hidden_dims'],
                         drop = kwargs['drop']).to(kwargs['device'])
    s_PRE = PREMODEL(encoder = shared_encoder,
                    decoder = shared_decoder,
                    input_dim = kwargs['input_dim'],
                    output_dim = kwargs['latent_dim'],
                    hidden_dims = kwargs['encoder_hidden_dims'],
                    drop = kwargs['drop'],
                    norm_flag = kwargs['norm_flag'],
                    noise_flag = True).to(kwargs['device'])
    t_PRE = PREMODEL(encoder = shared_encoder,
                    decoder = shared_decoder,
                    input_dim = kwargs['input_dim'],
                    output_dim = kwargs['latent_dim'],
                    hidden_dims = kwargs['encoder_hidden_dims'],
                    drop = kwargs['drop'],
                    norm_flag = kwargs['norm_flag'],
                    noise_flag = True).to(kwargs['device'])
    AE_params = [s_PRE.private_encoder.parameters(),
                 t_PRE.private_encoder.parameters(),
                 shared_encoder.parameters(),
                 shared_decoder.parameters()]

    #AE = EncoderDecoder(shared_encoder, shared_decoder, noise_flag=True)
    ae_optimizer = torch.optim.AdamW(chain(*AE_params), lr=kwargs['lr'])
    best_threshold = np.inf
    print("================Pre-training for feature extraction================")
    if kwargs['retrain_flag']:
        # starting pre-training for domain feature 
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            train_loss_all = 0
            val_loss_all = 0
            t_PRE.train(); s_PRE.train()
            ae_optimizer.zero_grad()
            for i, batch in enumerate(zip(t_train, cycle(s_train))):
                train_loss = ae_train_step(model_s=s_PRE, model_t=t_PRE,
                                           s_batch=batch[1], t_batch=batch[0],
                                           device=kwargs['device'], optimizer=ae_optimizer)
                train_loss_all += train_loss
            
            t_PRE.eval(); s_PRE.eval()
            s_val_loss = eval_epoch(model=s_PRE, loader=s_test, device=kwargs['device'])
            t_val_loss = eval_epoch(model=t_PRE, loader=t_test, device=kwargs['device'])
            
            val_loss_all = s_val_loss + t_val_loss
            if (epoch+1) % 50 == 0:
                print('AE training epoch = {}, loss = {:.4f}'.format(epoch+1, val_loss_all))
            if (val_loss_all < best_threshold):
                torch.save(s_PRE.state_dict(), os.path.join(kwargs['model_save_folder'], 's_PRE.pt'))
                torch.save(t_PRE.state_dict(), os.path.join(kwargs['model_save_folder'], 't_PRE.pt'))
                best_threshold = val_loss_all
    else:
        try:
            s_PRE.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 's_PRE.pt')))
            t_PRE.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 't_PRE.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")
    t_PRE.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 't_PRE.pt')))
    
    return t_PRE.shared_encoder

def drug_prototype(labeled_ccle, encoder, device):

    def extract_labeled_features(dataset):
        encoder.eval()
        feats = []
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
            for batch in loader:
                feats.append(encoder(batch[0].to(device)).cpu().numpy())
        return np.concatenate(feats, axis=0)

    feat_s_labeled = extract_labeled_features(labeled_ccle)
    drug_labels = labeled_ccle.tensors[2].cpu().numpy()
    assert feat_s_labeled.shape[0] == drug_labels.shape[0], "Features and labels mismatch!"

    proto_ids = []
    proto_feats = []
    n_drugs = drug_labels.shape[1]

    for drug_idx in range(n_drugs):
        labels_for_drug = drug_labels[:, drug_idx]
        resistant_mask = (labels_for_drug == 0)
        sensitive_mask = (labels_for_drug == 1)

        if resistant_mask.sum() > 0 and sensitive_mask.sum() > 0:
            resistant_proto = feat_s_labeled[resistant_mask].mean(axis=0)
            sensitive_proto = feat_s_labeled[sensitive_mask].mean(axis=0)

            proto_ids.append([1, drug_idx, 0])
            proto_feats.append(resistant_proto)

            proto_ids.append([1, drug_idx, 1])
            proto_feats.append(sensitive_proto)

    s_proto_ids = torch.tensor(proto_ids, dtype=torch.long)
    s_proto_feats = torch.tensor(np.array(proto_feats), dtype=torch.float32)

    return [s_proto_ids, s_proto_feats]

