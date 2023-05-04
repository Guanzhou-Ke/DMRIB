
import torch
from torch import nn
from torch.nn import functional as F

from networks import build_mlp, build_off_the_shelf_cnn
from losses.dclloss import DCLLoss, DCLWLoss


class ContrastiveModel(nn.Module):
    """
    Contrastive Model for pretext training.
    """

    def __init__(self, args, device='cpu') -> None:
        super().__init__()
        self.device = device
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
        
        # projection head.
        self.projection_head = nn.Sequential(nn.Linear(args.consis_enc.output_dim, args.consis_enc.output_dim, bias=False), 
                                             nn.BatchNorm1d(args.consis_enc.output_dim),
                                             nn.ReLU(inplace=True), 
                                             nn.Linear(args.consis_enc.output_dim, args.consis_enc.project_dim, bias=True))
        self.pooling_method = args.fusion.pooling_method
        # loss
        if args.consis_enc.loss_type == 'dcl':
            self.contrast_criterion = DCLLoss(args.consis_enc.temperature)
        elif args.consis_enc.loss_type == 'dclw':
            self.contrast_criterion = DCLWLoss(args.consis_enc.temperature)
        else:
            raise ValueError('Loss type must be `dcl` or `dclw`.')
        
       
    def forward(self, x):
        feature = self.enc(x)
        cont_out = self.projection_head(feature)
        return feature, F.normalize(cont_out, dim=-1)
    
    
    def get_loss(self, Xs):
        """
        Using infoNCE to extract consistency information.
        """
        loss = 0.
        views = len(Xs)
        for i in range(views):
            for j in range(i+1, views):
                x1, x2 = Xs[i], Xs[j]
                _, out1 = self(x1)
                _, out2 = self(x2)
        
                loss += (self.contrast_criterion(out1, out2) + self.contrast_criterion(out2, out1))
        # Normalize.
        loss /= (views-1)
        return loss
    
    
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