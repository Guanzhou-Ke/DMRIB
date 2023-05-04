
import os

import torch
from torch import nn
from torch.nn import functional as F

from networks import build_mlp, build_off_the_shelf_cnn
from losses.scanloss import SCANLoss, ConfidenceBasedCE


class ConsistencyEncoder(nn.Module):
    
    def __init__(self, args, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.args = args
        self.nheads = args.selflabel.nheads
        self.pooling_method = args.fusion.pooling_method
        if args.backbone.type == 'cnn':
            self.enc = build_off_the_shelf_cnn(name=args.consis_enc.backbone,
                                                channels=args.consis_enc.channels,
                                                maxpooling=args.consis_enc.max_pooling)
        elif args.backbone.type == 'mlp':
            self.enc = build_mlp(layers=args.consis_enc.backbone,
                                 activation=args.consis_enc.activation,
                                 first_norm=args.consis_enc.first_norm)
        else:
            raise ValueError('Backbone type error.')
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(args.hidden_dim, args.dataset.class_num) for _ in range(self.nheads)])
        
        
    
    def load_pretrain_weights(self, model_state, stage='scan'):
        self.stage = stage
        
        if stage == 'scan': # Weights are supposed to be transfered from contrastive training
            missing = self.load_state_dict(model_state, strict=False)
            self.criterion = SCANLoss(self.args.selflabel.scan_entropy_weight)
        elif stage == 'selflabel': # Weights are supposed to be transfered from scan 
            try:
                missing = self.load_state_dict(model_state, strict=True)
            except:
                missing = self.load_state_dict(model_state['model'], strict=True)
            self.criterion = ConfidenceBasedCE(self.args.selflabel.confidence_threshold, self.args.selflabel.apply_class_balancing)
        else:
            raise NotImplementedError   
  
               
        
    def forward(self, x, forward_pass='default', train=True):
        if forward_pass == 'default':
            features = self.consistency_repr(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]
            
            if train:
                return features, out
            else:
                return features, out[0]

        elif forward_pass == 'backbone':
            out = self.consistency_repr(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]
            if train:
                # if training, we return all head outputs.
                return out
            else:
                # else, we just return the best one.
                return out[0]
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        
        
    
    def get_loss(self, anchors, neighbors, forward_pass='default', stage='scan'):
          
        if stage == 'scan':
            if forward_pass == 'default':
                anchors = self.consistency_repr(anchors)
                neighbors = self.consistency_repr(neighbors)
        
            anchors_output = self(anchors, forward_pass='head')
            neighbors_output = self(neighbors, forward_pass='head')
            total_loss, consistency_loss, entropy_loss = [], [], []
            for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
                total_loss_, consistency_loss_, entropy_loss_ = self.criterion(anchors_output_subhead,
                                                                            neighbors_output_subhead)
                total_loss.append(total_loss_)
                consistency_loss.append(consistency_loss_.item())
                entropy_loss.append(entropy_loss_.item())
                
            total_loss = torch.sum(torch.stack(total_loss, dim=0))
            consistency_loss = sum(consistency_loss)
            entropy_loss = sum(entropy_loss)
            return total_loss, consistency_loss, entropy_loss
        elif stage == 'selflabel':
            with torch.no_grad():
                _, weak_out = self(anchors, forward_pass='default', train=False)
            _, strong_out = self(neighbors, forward_pass='default', train=False)
            loss = self.criterion(weak_out, strong_out)
    
            return loss
        else:
            raise ValueError(f'The stage of consistency encoder must be scan or selflabel, but got {self.stage}')

    
    
    def consistency_repr(self, Xs, normlization=False):
        features = [self.enc(x) for x in Xs]
        
        if self.pooling_method == 'mean':
            # use mean pooling to get common feature
            common_z = torch.stack(features, dim=-1).mean(dim=-1)
        elif self.pooling_method == 'sum':
            common_z = torch.stack(features, dim=-1).sum(dim=-1)
        elif self.pooling_method == 'first':
            common_z = features[0]
        if normlization:
            return F.normalize(common_z, dim=-1)
        else:
            return common_z