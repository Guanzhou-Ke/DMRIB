
import argparse
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
from tqdm import tqdm

from configs.basic_config import get_cfg
from models.contrastive_model import ContrastiveModel
from utils.memory_bank import MemoryBank
from utils import (fill_memory_bank,
                   knn_evaluation,
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


def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    device = torch.device(
        f"cuda:{config.device}") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Use: {device}')
    
    result_dir = os.path.join(config.train.log_dir, 'pretext')
    os.makedirs(result_dir, exist_ok=True)
    checkpoint_path = os.path.join(result_dir, 'checkpoint.pth')
    finalmodel_path = os.path.join(result_dir, 'final_model.pth')
    topk_neighbors_train_path = os.path.join(result_dir, 'topk_neighbors_train.npy')
    topk_neighbors_val_path = os.path.join(result_dir, 'topk_neighbors_val.npy')
    evaluate_intervals = config.train.evaluate
    
    use_wandb = config.wandb
    
    # For reproducibility
    reproducibility_setting(config.seed)

    # Create contrastive model.
    model = ContrastiveModel(config, device)
    model = model.to(device)
    print_network(model)

    # Create Dataset
    train_transforms = get_train_transformations(config, task='pretext')
    val_transforms = get_val_transformations(config)
    train_dataset = get_train_dataset(config, train_transforms, neighbors=False)
    val_dataset = get_val_dataset(config, val_transforms)
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
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    # Dataset w/o augs for knn eval
    base_dataset = get_train_dataset(config, val_transforms)
    base_dataloader = DataLoader(base_dataset,
                                config.train.batch_size,
                                num_workers=config.train.num_workers,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    memory_bank_base = MemoryBank(len(base_dataset), config, device)
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset), config, device)
    memory_bank_val.cuda()
    
    # optimizer
    optimizer = get_optimizer(model.parameters(), config.train.lr, config.train.optim)
    
    
    # Checkpoint
    if config.train.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    if use_wandb:
        wandb.init(project=config.project_name,
                config=config,
                name=f"{config.dataset.name}-eid:{config.experiment_id}-pretext")
        wandb.watch(model, log='all', log_graph=True, log_freq=15)
    
    # Training
    for epoch in range(start_epoch, config.train.epochs):
        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        if use_wandb:
            wandb.log({'lr': lr}, step=epoch)
        
        # Train
        loss = train_a_epoch(config, train_dataloader, model, optimizer, epoch, device, lr)
        if use_wandb:
            wandb.log({'pretext-train-loss': loss}, step=epoch)
        
        # fill memory bank
        fill_memory_bank(base_dataloader, model, memory_bank_base, device)
        
        if epoch % evaluate_intervals == 0:
            knn_result = knn_evaluation(val_dataloader, model, memory_bank_base, device)
            print(f"[Evaluate {epoch}/{config.train.epochs}] {knn_result}")
            if use_wandb:
                wandb.log(knn_result, step=epoch)
            
        # Checkpoint
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch+1}, checkpoint_path)
        
    # Save final model
    torch.save(model.state_dict(), finalmodel_path)
    
    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print('Fill memory bank for mining the nearest neighbors (train) ...')
    fill_memory_bank(base_dataloader, model, memory_bank_base, device)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(topk_neighbors_train_path, indices)   

   
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print('Fill memory bank for mining the nearest neighbors (val) ...')
    fill_memory_bank(val_dataloader, model, memory_bank_val, device)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(topk_neighbors_val_path, indices)
        
        
def train_a_epoch(args, train_dataloader, model, optimizer, epoch, device, lr):
    model.train()
    losses = []
    if args.verbose:
        pbar = tqdm(train_dataloader, ncols=0, unit=" batch")
        
    for data in train_dataloader:
        Xs, y = data
        Xs = [x.to(device) for x in Xs]
        
        loss = model.get_loss(Xs)
        
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


if __name__ == "__main__":
    main()