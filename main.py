import os
import pandas as pd
import torch
import json
import argparse
import itertools
import dataload
import config
import pretraining, classifier, finetuning
from copy import deepcopy
from utility import dict_to_str, make_dir, wrap_params
import time
import warnings
import random, numpy as np


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.metrics import auc, precision_recall_curve

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)

def main(args, params_dict, drug):
    start_time = time.time()
    device = config.CUDA_ID
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    with open(os.path.join('train_params.json'), 'r') as f:
        training_params = json.load(f)
    training_params['unlabeled'].update(params_dict)
    training_params['labeled'].update(params_dict)
    param_str = dict_to_str(params_dict)
    model_save_folder = os.path.join('model_save')
    training_params.update({'device': device,
                            'input_dim': gex_features_df.shape[1] - 1,
                            'model_save_folder': os.path.join(model_save_folder, param_str),
                            'retrain_flag': args.retrain_flag,
                            'norm_flag': args.norm_flag,
                            'alph': args.alph,
                            'beta': args.beta})
    make_dir(training_params['model_save_folder'])

    # Data construction for the pre-training
    s_dataloaders, t_dataloaders = dataload.get_unlabeled_dataloaders(gex_features_df=gex_features_df,
                                                                      seed=42,
                                                                      test_ratio=0.25,
                                                                      batch_size=training_params['unlabeled'][
                                                                          'batch_size'])

    # Start pretraining, obtain shared encoder
    encoder = pretraining.training(s_dataloaders=s_dataloaders,
                                   t_dataloaders=t_dataloaders,
                                   **wrap_params(training_params, type='unlabeled'))
    labeled_ccle,gene_names = dataload.get_ccle_multi_labeled(gex_features_df=gex_features_df,
                                                   drug=drug,
                                                   measurement=args.measurement,
                                                   threshold_gdsc=args.thres_g)
    drug_prototypes = pretraining.drug_prototype(labeled_ccle=labeled_ccle,
                                                 encoder=encoder,
                                                 device=training_params['device'])

    # Data construction for the fine-tuning
    labeled_dataloader = dataload.get_multi_labeled_dataloader(gex_features_df=gex_features_df,
                                                               seed=42,
                                                               batch_size=training_params['labeled']['batch_size'],
                                                               drug=drug,
                                                               ccle_measurement=args.measurement,
                                                               threshold_gdsc=args.thres_g,
                                                               threshold_label=args.thres_s,
                                                               n_splits=args.n)
    # start finetuning, obtain UDA predictor
    fold = 0
    all_results = []

    for train_labeled_ccle, test_labeled_ccle, labeled_tcga in labeled_dataloader:
        print('\n################ Dataset Fold: {} ################'.format(fold))
        ft_encoder = deepcopy(encoder)
        print('================Drugs:================\n', drug)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = classifier.training(
                encoder=ft_encoder,
                train_dataloader=train_labeled_ccle,
                val_dataloader=test_labeled_ccle,
                drug=drug,
                **wrap_params(training_params, type='labeled'))
            predictor.load_state_dict(torch.load(
                os.path.join(model_save_folder, param_str, 'predictor.pt')))

            network = finetuning.training(
                encoder=ft_encoder,
                classifier=predictor,
                s_dataloader=train_labeled_ccle,
                t_dataloader=labeled_tcga,
                drug=drug,
                prototype=drug_prototypes,
                params_str=param_str,
                **wrap_params(training_params, type='labeled'))

            print("\n================Transfer testing for TCGA data================")
            test_loss, results, y_true, y_pred, y_mask, cancer_labels = finetuning.testing(
                model=network,
                t_dataloader=labeled_tcga,
                drug=drug,
                device=training_params['device'])

            print('testing loss : {:.4f}'.format(test_loss))
            print('test-avg (TCGA) metrics (auc,aupr,f1,acc): {}'.format(
                np.around(results.mean(axis=1), 4)))

        results = pd.DataFrame(results)
        results.columns = drug;
        results.index = ['auc', 'aupr', 'f1', 'acc']
        task_save_folder = os.path.join('results', 'fold-' + str(fold))
        make_dir(task_save_folder)
        file_name = os.path.join(task_save_folder, param_str)
        with open(f'{file_name}.csv', 'w') as f:
            results.to_csv(f)
        fold = fold + 1
        all_results.append(results)


    # Calculate the average result of 5-CV
    avg_result = np.mean(np.array(all_results), 0)
    avg_result = pd.DataFrame(avg_result)
    avg_result.columns = drug;
    avg_result.index = ['auc', 'aupr', 'f1', 'acc']
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{param_str}.csv')
    avg_result.to_csv(file_path, encoding='utf-8', lineterminator='\n')

    # Computer the model running time
    elapsed = time.time() - start_time
    print('9-drug elapsed time: ', round(elapsed, 4) / 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining and Fine_tuning')
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='Z_SCORE',
                        choices=['Z_SCORE', 'LN_IC50', 'AUC'])
    parser.add_argument('--thres_gdsc', dest='thres_g', nargs='?', type=float, default=0.0)
    parser.add_argument('--thres_label', dest='thres_s', nargs='?', type=float, default=0.1)
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)
    parser.add_argument('--alph', dest='alph', nargs='?', type=float, default=0.2, help='Coefficient of transfer loss')
    parser.add_argument('--beta', dest='beta', nargs='?', type=float, default=0.3,
                        help='Coefficient of contrastive loss')

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)
    args = parser.parse_args()

    params_grid = {
        "pretrain_num_epochs": [1],#200
        "uda_num_epochs": [1],#200
        "train_num_epochs": [1] #100
    }
    keys, values = zip(*params_grid.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param in params_list:
        main(args=args, params_dict=param,
             drug=["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
                   "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"])