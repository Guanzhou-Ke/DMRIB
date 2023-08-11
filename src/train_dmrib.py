
import argparse
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import wandb
from tqdm import tqdm

from configs.basic_config import get_cfg
from models.dmrib import DMRIB
from utils import (clustering_by_representation,
                   reproducibility_setting,
                   plot_embedding_by_tsne,
                   print_network)
from datatool import (get_val_transformations,
                      get_train_dataset,
                      get_val_dataset)
from optimizer import adjust_learning_rate, get_optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args


@torch.no_grad()
def valid_by_kmeans(val_dataloader, model):
    model.eval()
    targets = []
    consist_reprs = []
    vspecific_reprs = []
    concate_reprs = []
    for Xs, target in tqdm(val_dataloader, desc='Kmeans evaluation'):
        Xs = [x.cuda(non_blocking=True) for x in Xs]
        consist_repr_ = model.consistency_features(Xs)
        vspecific_repr_ = model.vspecific_features(Xs, single=True)
        concate_repr_ = model.commonZ(Xs)
        targets.append(target)
        consist_reprs.append(consist_repr_.detach().cpu())
        vspecific_reprs.append(vspecific_repr_.detach().cpu())
        concate_reprs.append(concate_repr_.detach().cpu())
    targets = torch.concat(targets, dim=-1).numpy()
    consist_reprs = torch.vstack(consist_reprs).squeeze().detach().cpu().numpy()
    vspecific_reprs = torch.vstack(vspecific_reprs).squeeze().detach().cpu().numpy()
    concate_reprs = torch.vstack(concate_reprs).squeeze().detach().cpu().numpy()
    result = {}
    acc, nmi, ari, _, p, fscore = clustering_by_representation(consist_reprs, targets)
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore
    
    acc, nmi, ari, _, p, fscore = clustering_by_representation(vspecific_reprs, targets)
    result['vspec-acc'] = acc
    result['vspec-nmi'] = nmi
    result['vspec-ari'] = ari
    result['vspec-p'] = p
    result['vspec-fscore'] = fscore
    
    acc, nmi, ari, _, p, fscore = clustering_by_representation(concate_reprs, targets)
    result['cat-acc'] = acc
    result['cat-nmi'] = nmi
    result['cat-ari'] = ari
    result['cat-p'] = p
    result['cat-fscore'] = fscore
    return result


def train_a_epoch(args, train_dataloader, model, optimizer, epoch, lr):
    losses = []
    loss_parts = defaultdict(list)
    model.train()
    if args.verbose:
        pbar = tqdm(train_dataloader, ncols=0, unit=" batch")
    
        
    for Xs, _ in train_dataloader:
        Xs = [x.cuda(non_blocking=True) for x in Xs]
        
        loss, loss_part = model.train_view_specific_encoder(Xs)
        
        optimizer.zero_grad()
        # We have to split each view specific encoder's loss to ensure the reconstruction.
        for idx in range(len(loss)):
            loss[idx].backward(retain_graph=True)
        loss[-1].backward()
        optimizer.step()
        
        loss = np.array([l.item() for l in loss])
        
        losses.append(np.mean(loss))
        
        if len(loss_part) > 0:
            for lp in loss_part:
                loss_parts[lp[0]].append(lp[1])
        
        if args.verbose:
            pbar.update()
            pbar.set_postfix(
                tag='TRAIN',
                epoch=epoch,
                loss=f"{np.mean(losses):.6f}",
                lr=f"{lr:.6f}"
            )
    
    if args.verbose:
        pbar.close()
        
    return np.mean(losses), loss_parts




def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    device = torch.device(
        f"cuda:{config.device}") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Use: {device}')

    consistency_encoder_path = os.path.join(config.train.log_dir, 'selflabel', 'final_model.pth')
    selflabel_status = torch.load(os.path.join(config.train.log_dir, 'selflabel', 'clustering_stats.pth'))
    evaluate_intervals = config.train.evaluate
    
    use_wandb = config.wandb
    
    result_dir = os.path.join(config.train.log_dir, 'dmrib')
    os.makedirs(result_dir, exist_ok=True)
    checkpoint_path = os.path.join(result_dir, 'checkpoint.pth')
    finalmodel_path = os.path.join(result_dir, 'final_model.pth')

    # For reproducibility
    reproducibility_setting(config.seed)

    # Create contrastive model.
    model = DMRIB(config, device=device, consistency_pretrain_weight=torch.load(consistency_encoder_path, map_location='cpu'))


    val_transformations = get_val_transformations(config)
    train_dataset = get_train_dataset(config, val_transformations)
    val_dataset = get_val_dataset(config, val_transformations)
    train_dataloader = DataLoader(train_dataset,
                                  config.train.batch_size,
                                  num_workers=config.train.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                config.train.batch_size,
                                num_workers=config.train.num_workers,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    
    dl = DataLoader(val_dataset, 16, shuffle=True)
    recon_samples = next(iter(dl))[0]
    recon_samples = [x.cuda(non_blocking=True) for x in recon_samples]
    
    # optimizer
    optimizer = get_optimizer(model.parameters(), config.train.lr, config.train.optim)
    
    # Checkpoint
    if config.train.resume and os.path.exists(checkpoint_path):
        print('Restart from checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])        
        start_epoch = checkpoint['epoch']
    else:
        print('Start a new training')
        start_epoch = 0
        
    model = model.cuda()
    print_network(model)
    
    if use_wandb:
        wandb.init(project=config.project_name,
                config=config,
                name=f"{config.dataset.name}-eid:{config.experiment_id}-DMRIB")
        wandb.watch(model, log='all', log_graph=True, log_freq=15)
    
    
    # Start scan training.
    for epoch in range(start_epoch, config.train.epochs):
        
        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        
        # Train
        loss, loss_parts = train_a_epoch(config, train_dataloader, model, optimizer, epoch, lr)
        if use_wandb:
            wandb.log({'lr': lr}, step=epoch)
            wandb.log({'DMRIB-train-loss': loss}, step=epoch)
            for k, v in loss_parts.items():
                v_ = np.mean(v)
                wandb.log({k: v_}, step=epoch)

        
        if epoch % evaluate_intervals == 0:
            orignal_grid, recon_grid = reconstruction(model, recon_samples)
    
            sample_grid = sampling(config, model, selflabel_status['hungarian_match'])
            
            kmeans_result = valid_by_kmeans(val_dataloader, model)
            
            print("RIM-CAC:", kmeans_result)
            print("Selflabel:", selflabel_status)
            
            if use_wandb:
                wandb.log({'traget': wandb.Image(orignal_grid), 'generated': wandb.Image(recon_grid)}, step=epoch)
                # wandb.log(kmeans_result, step=epoch)
                wandb.log({'conditional-samples': wandb.Image(sample_grid)}, step=epoch)
                
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, checkpoint_path)
        
    model.eval()
    # Save final model
    torch.save(model.state_dict(), finalmodel_path) 
    # Visualization
    if config.dataset.name == 'mvc-10':
        pass
    else:
        plot_embedding_by_tsne(val_dataloader, model, os.path.join(result_dir, 'embedding.png'))


def sampling(args, model, hungarian_match):
    """
    Sampling from conditional vaes
    """
    sample_nums = args.train.samples_num
    lidx = [x for _, x in hungarian_match]
    outs = model.sampling(sample_nums, lidx)
    sample_grid = make_grid(outs.detach().cpu(), nrow=sample_nums*args.views)
    return sample_grid


def reconstruction(model, recon_samples):
    recon_Xs = model.enc_dec(recon_samples)
    orignal_view = torch.cat([x.detach().cpu() for x in recon_samples])
    orignal_grid = make_grid(orignal_view)
    recon_view = torch.cat([x.detach().cpu() for x in recon_Xs])
    recon_grid = make_grid(recon_view)
    return orignal_grid, recon_grid
    

    
if __name__ == '__main__':
    main()
    