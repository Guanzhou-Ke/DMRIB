
import argparse
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
from tqdm import tqdm

from configs.basic_config import get_cfg
from models.consistency_model import ConsistencyEncoder
from utils import (get_predictions,
                   clustering_by_representation,
                   hungarian_evaluate,
                   reproducibility_setting,
                   EarlyStopper,
                   print_network)
from datatool import (get_train_transformations,
                      get_val_transformations,
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
    reprs = []
    for Xs, target in tqdm(val_dataloader, desc='Kmeans evaluation'):
        Xs = [x.cuda(non_blocking=True) for x in Xs]
        repr_ = model.consistency_repr(Xs)
        targets.append(target)
        reprs.append(repr_.detach().cpu())
    targets = torch.concat(targets, dim=-1).numpy()
    reprs = torch.vstack(reprs).squeeze().detach().cpu().numpy()
    acc, nmi, ari, _, p, fscore = clustering_by_representation(reprs, targets)
    return {'kmeans-acc': acc, 'kmeans-nmi': nmi, 'kmeans-ari':ari, 'kmeans-p': p, 'kmeans-fscore': fscore}


def train_a_epoch(args, train_dataloader, model, optimizer, epoch, lr):
    losses = []
    model.train()
    if args.verbose:
        pbar = tqdm(train_dataloader, ncols=0, unit=" batch")
    
        
    for weak_xs, strong_xs, _ in train_dataloader:
        
        weak_xs = [x.cuda(non_blocking=True) for x in weak_xs]
        strong_xs = [x.cuda(non_blocking=True) for x in strong_xs]
        
        loss = model.get_loss(weak_xs, strong_xs, stage='selflabel')

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if args.verbose:
            pbar.update()
            pbar.set_postfix(
                tag='TRAIN',
                epoch=epoch,
                loss=f"{np.mean(losses):.4f}",
                lr=f"{lr:.4f}",
            )
    
    if args.verbose:
        pbar.close()
        
    return np.mean(losses)




def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    device = torch.device(
        f"cuda:{config.device}") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Use: {device}')

    scan_dir = os.path.join(config.train.log_dir, 'scan')
    scan_model_path = os.path.join(scan_dir, 'final_model.pth')

    evaluate_intervals = config.train.evaluate
    
    result_dir = os.path.join(config.train.log_dir, 'selflabel')
    os.makedirs(result_dir, exist_ok=True)
    checkpoint_path = os.path.join(result_dir, 'checkpoint.pth')
    finalmodel_path = os.path.join(result_dir, 'final_model.pth')

    # For reproducibility
    reproducibility_setting(config.seed)
    use_wandb = config.wandb
    
    # Create contrastive model.
    model = ConsistencyEncoder(config, device)

    train_transformations = get_train_transformations(config, task='selflabel')
    val_transformations = get_val_transformations(config)
    train_dataset = get_train_dataset(config, {'weak': val_transformations, 'strong': train_transformations})
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
    
    
    # optimizer
    optimizer = get_optimizer(model.parameters(), config.train.lr, config.train.optim)
    
    # Checkpoint
    if config.train.resume and os.path.exists(checkpoint_path):
        print('Restart from checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_pretrain_weights(checkpoint['model'], stage='selflabel')
        optimizer.load_state_dict(checkpoint['optimizer'])        
        start_epoch = checkpoint['epoch']
    else:
        print("Start a new training.")
        start_epoch = 0
        model_state = torch.load(scan_model_path, map_location='cpu')
        model.load_pretrain_weights(model_state, stage='selflabel')
        
    model = model.cuda()
    print_network(model)
    if use_wandb:
        wandb.init(project=config.project_name,
                config=config,
                name=f"{config.dataset.name}-eid:{config.experiment_id}-selflabel")
        wandb.watch(model, log='all', log_graph=True, log_freq=15)
        
    earlystop = EarlyStopper(3)
    # Start scan training.
    for epoch in range(start_epoch, config.train.epochs):
        
        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        if use_wandb:
            wandb.log({'lr': lr}, step=epoch)
        
        # Train
        loss = train_a_epoch(config, train_dataloader, model, optimizer, epoch, lr)
        if use_wandb:
            wandb.log({'fine-tune-train-loss': loss}, step=epoch)

        
        if epoch % evaluate_intervals == 0:
            print('Make prediction on validation set ...')
            predictions = get_predictions(config, val_dataloader, model)
            
            kmeans_result = valid_by_kmeans(val_dataloader, model)
            print(kmeans_result)
                
            print('Evaluate with hungarian matching algorithm ...')
            clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False)
            print(clustering_stats) 
            clustering_stats.pop('hungarian_match')
            if use_wandb:
                wandb.log(clustering_stats, step=epoch)
                
        if earlystop.early_stop(loss):
            break
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, checkpoint_path)
        
        
    # Evaluate and save the final model
    print('Evaluate model at the end')
    predictions = get_predictions(config, val_dataloader, model)
    clustering_stats = hungarian_evaluate(0, predictions, 
                            class_names=val_dataset.classes, 
                            compute_confusion_matrix=True, 
                            confusion_matrix_file=os.path.join(result_dir, 'confusion_matrix.png'))
    print(clustering_stats)
    torch.save(clustering_stats, os.path.join(result_dir, 'clustering_stats.pth'))  
    
    # Save final model
    torch.save(model.state_dict(), finalmodel_path) 
    
    
    
if __name__ == '__main__':
    main()
    