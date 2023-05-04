
import argparse
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from configs.basic_config import get_cfg
from models.dmrib import DMRIB
from utils import print_network, get_y_preds
from datatool import (get_val_transformations, get_val_dataset, get_train_dataset)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args


@torch.no_grad()
def extract_repres(dataloader, model):
    model.eval()
    targets = []
    # consist_reprs = []
    # vspecific_reprs = []
    concate_reprs = []
    for Xs, target in tqdm(dataloader, desc='Extract repres'):
        Xs = [x.cuda(non_blocking=True) for x in Xs]
        # consist_repr_ = model.consistency_features(Xs)
        # vspecific_repr_ = model.vspecific_features(Xs, single=True)
        concate_repr_ = model.commonZ(Xs)
        targets.append(target)
        # consist_reprs.append(consist_repr_.detach().cpu())
        # vspecific_reprs.append(vspecific_repr_.detach().cpu())
        concate_reprs.append(concate_repr_.detach().cpu())
    targets = torch.concat(targets, dim=-1).numpy()
    # consist_reprs = torch.vstack(consist_reprs).squeeze().detach().cpu().numpy()
    # vspecific_reprs = torch.vstack(vspecific_reprs).squeeze().detach().cpu().numpy()
    concate_reprs = torch.vstack(concate_reprs).squeeze().detach().cpu().numpy()
    return concate_reprs,  targets
    
    
    
def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    device = torch.device(
        f"cuda:{config.device}") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Use: {device}')
    model_path = os.path.join(config.train.log_dir, 'rimcac', 'final_model.pth')
    
    # Create contrastive model.
    model = DMRIB(config, device=device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model = model.cuda()
    print_network(model)
    
    val_transformations = get_val_transformations(config)
    val_dataset = get_val_dataset(config, val_transformations)
    train_dataset = get_train_dataset(config, val_transformations)
    val_dataloader = DataLoader(val_dataset,
                                config.train.batch_size,
                                num_workers=config.train.num_workers,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    train_dataloader = DataLoader(train_dataset,
                                config.train.batch_size,
                                num_workers=config.train.num_workers,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    
    train_concate_reprs, train_targets = extract_repres(train_dataloader, model)
    test_concate_reprs, test_targets = extract_repres(val_dataloader, model)
    
    accs = []
    nmis = []
    aris = []
    for _ in tqdm(range(10), desc='Clustering'):
        seed = torch.randint(9999, (1, )).numpy()[0]
        km = KMeans(n_clusters=config.dataset.class_num, random_state=seed, init='random', n_init=100)
        concate_repr = np.r_[train_concate_reprs, test_concate_reprs]
        concate_targets = np.r_[train_targets, test_targets]
        concat_preds = km.fit_predict(concate_repr)
        # consist_preds = km.fit_predict(consist_reprs)
        # vspec_preds = km.fit_predict(vspecific_reprs)
        
        concat_preds = get_y_preds(concate_targets, concat_preds, config.dataset.class_num)
        # consist_preds = get_y_preds(consist_preds, targets, config.dataset.class_num)
        # vspec_preds = get_y_preds(vspec_preds, targets, config.dataset.class_num)
        
        acc = metrics.accuracy_score(concate_targets, concat_preds)
        nmi = metrics.normalized_mutual_info_score(concate_targets, concat_preds)
        ari = metrics.adjusted_rand_score(concate_targets, concat_preds)
        
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)
    print(accs)    
    accs = np.array(accs)
    nmis = np.array(nmis)
    aris = np.array(aris)
    print(f"[Clustering] For 10 runs, ACC: {accs.mean():.4f}({accs.std():.4f}), NMI: {nmis.mean():.4f}({nmis.std():.4f}), ARI: {aris.mean():.4f}({aris.std():.4f})")
    
    # For classification
    accs = []
    ps = []
    fscores = []
    for _ in tqdm(range(10), desc='Classification'):
        seed = torch.randint(9999, (1, )).numpy()[0]
        svc = SVC(random_state=seed)
        svc.fit(train_concate_reprs, train_targets)
        concat_preds = svc.predict(test_concate_reprs)
        # consist_preds = svc.fit_predict(consist_reprs)
        # vspec_preds = svc.fit_predict(vspecific_reprs)
        
        acc = metrics.accuracy_score(test_targets, concat_preds)
        p = metrics.precision_score(test_targets, concat_preds, average='macro')
        f1 = metrics.f1_score(test_targets, concat_preds, average='macro')
        
        accs.append(acc)
        ps.append(p)
        fscores.append(f1)
        
    accs = np.array(accs)
    ps = np.array(ps)
    fscores = np.array(fscores)
    
    print(f"[Classification] For 10 runs, ACC: {accs.mean():.4f}({accs.std():.4f}), P: {ps.mean():.4f}({ps.std():.4f}), F1: {fscores.mean():.4f}({fscores.std():.4f})")
    
    
    
if __name__ == '__main__':
    main()        