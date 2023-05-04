
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
                   scan_evaluate,
                   clustering_by_representation,
                   hungarian_evaluate,
                   reproducibility_setting,
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
    for anchor, neighbor, _, target in tqdm(val_dataloader, desc='Kmeans evaluation'):
        anchor = [x.cuda(non_blocking=True) for x in anchor]
        repr_ = model.consistency_repr(anchor)
        targets.append(target)
        reprs.append(repr_.detach().cpu())
    targets = torch.concat(targets, dim=-1).numpy()
    reprs = torch.vstack(reprs).squeeze().detach().cpu().numpy()
    acc, nmi, ari, _, p, fscore = clustering_by_representation(reprs, targets)
    return {'kmeans-acc': acc, 'kmeans-nmi': nmi, 'kmeans-ari':ari, 'kmeans-p': p, 'kmeans-fscore': fscore}


def train_a_epoch(args, train_dataloader, model, optimizer, epoch, lr):
    losses = []
    consistency_losses = []
    entropy_losses = []
    update_cluster_head_only = args.selflabel.update_cluster_head_only
    if args.verbose:
        pbar = tqdm(train_dataloader, ncols=0, unit=" batch")
    
    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN
        
    for anchors, neighbors, _, _ in train_dataloader:
        
        anchors = [x.cuda(non_blocking=True) for x in anchors]
        neighbors = [x.cuda(non_blocking=True) for x in neighbors]
        
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            total_loss, consistency_loss, entropy_loss = model.get_loss(anchors_features, neighbors_features, forward_pass='head')
            
        else: # Calculate gradient for backprop of complete network
            total_loss, consistency_loss, entropy_loss = model.get_loss(anchors, neighbors)

        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        consistency_losses.append(consistency_loss)
        entropy_losses.append(entropy_loss)
        
        if args.verbose:
            pbar.update()
            pbar.set_postfix(
                tag='TRAIN',
                epoch=epoch,
                loss=f"{np.mean(losses):.4f}",
                consis=f"{np.mean(consistency_losses):.4f}",
                entropy=f"{np.mean(entropy_losses):.4f}",
                lr=f"{lr:.4f}",
            )
    
    if args.verbose:
        pbar.close()
        
    return np.mean(losses), np.mean(consistency_losses), np.mean(entropy_losses)




def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    device = torch.device(
        f"cuda:{config.device}") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Use: {device}')

    pretext_dir = os.path.join(config.train.log_dir, 'pretext')
    pretext_model_path = os.path.join(pretext_dir, 'final_model.pth')

    topk_neighbors_train_path = os.path.join(
        pretext_dir, 'topk_neighbors_train.npy')
    topk_neighbors_val_path = os.path.join(
        pretext_dir, 'topk_neighbors_val.npy')
    evaluate_intervals = config.train.evaluate

    result_dir = os.path.join(config.train.log_dir, 'scan')
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
    train_dataset = get_train_dataset(
        config, train_transformations, neighbors=True, topk_neighbors_train_path=topk_neighbors_train_path)
    val_dataset = get_val_dataset(
        config, val_transformations, neighbors=True, topk_neighbors_val_path=topk_neighbors_val_path)
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
        model.load_pretrain_weights(checkpoint['model'], stage='scan')
        optimizer.load_state_dict(checkpoint['optimizer'])        
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']
    else:
        print('Load pretext model.')
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None
        model_state = torch.load(pretext_model_path, map_location='cpu')
        model.load_pretrain_weights(model_state, stage='scan')
        
    model = model.cuda()
    print_network(model)
    
    if use_wandb:
        wandb.init(project=config.project_name,
                config=config,
                name=f"{config.dataset.name}-eid:{config.experiment_id}-scan")
        wandb.watch(model, log='all', log_graph=True, log_freq=15)
    
    
    # Start scan training.
    for epoch in range(start_epoch, config.train.epochs):
        
        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        if use_wandb:
            wandb.log({'lr': lr}, step=epoch)
        
        # Train
        loss, consistency, entropy = train_a_epoch(config, train_dataloader, model, optimizer, epoch, lr)
        if use_wandb:
            wandb.log({'train-loss': loss}, step=epoch)
            wandb.log({'consistency-loss': consistency}, step=epoch)
            wandb.log({'entropy-loss': entropy}, step=epoch)
        
        
        if epoch % evaluate_intervals == 0:
            print('Make prediction on validation set ...')
            predictions = get_predictions(config, val_dataloader, model, have_neighbors=True)
            scan_stats = scan_evaluate(predictions)
            print(scan_stats)
            lowest_loss_head = scan_stats['lowest_loss_head']
            lowest_loss = scan_stats['lowest_loss']
            if use_wandb:
                wandb.log(scan_stats, step=epoch)
            
            kmeans_result = valid_by_kmeans(val_dataloader, model)
            print(kmeans_result)
            
            if lowest_loss < best_loss:
                print('New lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
                print('Lowest loss head is %d' %(lowest_loss_head))
                best_loss = lowest_loss
                best_loss_head = lowest_loss_head
                torch.save({'model': model.state_dict(), 'head': best_loss_head}, finalmodel_path)
            else:
                print('No new lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
                print('Lowest loss head is %d' %(best_loss_head))
                
            print('Evaluate with hungarian matching algorithm ...')
            clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
            print(clustering_stats) 
            clustering_stats.pop('hungarian_match')
            if use_wandb:
                wandb.log(clustering_stats, step=epoch)
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head}, checkpoint_path)
        
        
        # Evaluate and save the final model
    print('Evaluate best model based on SCAN metric at the end')
    model_checkpoint = torch.load(finalmodel_path, map_location='cpu')
    model.load_state_dict(model_checkpoint['model'])
    predictions = get_predictions(config, val_dataloader, model, have_neighbors=True)
    clustering_stats = hungarian_evaluate(model_checkpoint['head'], predictions, 
                            class_names=val_dataset.dataset.classes, 
                            compute_confusion_matrix=True, 
                            confusion_matrix_file=os.path.join(result_dir, 'confusion_matrix.png'))
    print(clustering_stats)    
    
    
    
if __name__ == '__main__':
    main()
    