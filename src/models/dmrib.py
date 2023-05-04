import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from utils import label_to_one_hot
from .consistency_model import ConsistencyEncoder
from .vspecific_encoder import ViewSpecificVAE


class DMRIB(nn.Module):

    def __init__(self, args, device='cpu', consistency_pretrain_weight=None) -> None:
        super().__init__()
        self.args = args
        self.device = device
        self.views = self.args.views
        
        
        # consistency encoder.
        if self.args.consis_enc.enable:
            self.consis_enc = ConsistencyEncoder(self.args, self.device)
            if consistency_pretrain_weight is not None:
                self.consis_enc.load_pretrain_weights(consistency_pretrain_weight, stage='selflabel')

        if self.args.vspecific.enable:
            # create view-specific encoder.
            for i in range(self.args.views):
                self.__setattr__(f"venc_{i+1}", ViewSpecificVAE(self.args, 
                                                                vid=i+1, 
                                                                channels=self.args.consis_enc.channels,
                                                                device=self.device))
            
        # Common feature pooling method. mean, sum, or first
        self.pooling_method = self.args.fusion.pooling_method
        
        # disentangled
        self.cz_mu = nn.Linear(self.args.hidden_dim, self.args.vspecific.latent_dim, bias=False)
        self.cz_var = nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.vspecific.latent_dim, bias=False),
            nn.Softplus(),
        )

    
    def sampling(self, samples_nums, labels):
        """
        samples_num: e
        """
        outs = []
        for label in labels:
            lidx = torch.tensor(label).repeat(samples_nums).to(self.device)
            y = label_to_one_hot(label_idx=lidx, num_classes=self.args.dataset.class_num)
            for i in range(self.args.views):
                venc = self.__getattr__(f"venc_{i+1}")
                out = venc.sample(samples_nums, y)
                outs.append(out)
        return torch.cat(outs)
    
    
    def train_view_specific_encoder(self, Xs, mask_idx: torch.BoolTensor = None):
        tot_loss = []
        recorder = [] 

        with torch.no_grad():
            consistency_z, y = self.consis_enc(Xs, train=False)
            
        y = label_to_one_hot(y.argmax(dim=1), num_classes=self.args.dataset.class_num).detach()
        cz = consistency_z.detach()
        
        for i in range(self.views):
            loss = 0.
            venc = self.__getattr__(f"venc_{i+1}")
            
            recons_loss, kld_loss = venc.get_loss(Xs[i], y=y, mask_idx=mask_idx)
            loss += (recons_loss + self.args.vspecific.kld_weight * kld_loss)
            
            recorder.append((f'view-{i+1}-recons-loss', recons_loss.item()))
            recorder.append((f'view-{i+1}-kld-loss', self.args.vspecific.kld_weight * kld_loss.item()))
            
            
            cmu, cstd = self.cz_mu(cz), self.cz_var(cz) + 1e-7
            
            # disentangled 
            vmu, vstd = venc.encode(Xs[i])
            vstd = F.softplus(vstd) + 1e-7

            disent_loss = self.args.disent.lam * self.disentangled(cmu, cstd, vmu, vstd)
            recorder.append((f'view-{i+1}-disentangled-loss', disent_loss.item()))
            loss += disent_loss
            recorder.append((f"view-{i+1}-total-loss", loss.item()))
            
            tot_loss.append(loss)
            
        
        return tot_loss, recorder
    
    
    def __fusion(self, Xs, mask_idx=None, ftype='C'):
        if self.args.consis_enc.enable:
            with torch.no_grad():
                consis_features, _ = self.consis_enc(Xs, train=False)

        vspecific_features = []
        if self.args.vspecific.enable:
            for i in range(self.views):
                venc = self.__getattr__(f"venc_{i+1}")
                feature = venc.latent(Xs[i], mask_idx=mask_idx)
                vspecific_features.append(feature)  
                
        if ftype == 'C':
            features = consis_features
        elif ftype == "V":
            features = vspecific_features[self.args.vspecific.best_view]
        elif ftype == "CV":
            best_view_features = vspecific_features[self.args.vspecific.best_view]
            features = torch.cat([consis_features, best_view_features], dim=-1)
        else:
            raise ValueError("Less than one kind information available.")
        
        return features
    
    
    def disentangled(self, cmu, cstd, vmu, vstd):
        consist_dist = Independent(Normal(loc=cmu, scale=cstd), 1)
        cz = consist_dist.rsample()
        
        vspecific_dist = Independent(Normal(loc=vmu, scale=vstd), 1)
        vz = vspecific_dist.rsample()
        # Upper bound of mutual information between consistency and specificity
        loss = consist_dist.log_prob(cz) - vspecific_dist.log_prob(vz)
        
        return loss.mean()
    
    
    def enc_dec(self, Xs, mask_idx=None):
        y = None
        if self.args.consis_enc.enable:
            with torch.no_grad():
                _, probs = self.consis_enc(Xs, train=False)
            y = F.log_softmax(probs, dim=1)
            y = label_to_one_hot(y.argmax(dim=1), 
                                num_classes=self.args.dataset.class_num)
        outs = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            out, _, _ = venc(Xs[i], y=y, mask_idx=mask_idx)
            outs.append(out)
        return outs
    
    
    def generate(self, z):
        outs = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            out = venc.decode(z)
            outs.append(out)
        return torch.cat(outs)
    

    @torch.no_grad()
    def commonZ(self, Xs, mask_idx=None):
        return self.__fusion(Xs, mask_idx=mask_idx, ftype=self.args.fusion.type)
    
    @torch.no_grad()
    def consistency_features(self, Xs, mask_idx=None):
        return self.__fusion(Xs, mask_idx=mask_idx, ftype='C')
    
    @torch.no_grad()
    def vspecific_features(self, Xs, mask_idx=None, single=False):
        vspecific_features = []
        if self.args.vspecific.enable:
            for i in range(self.views):
                venc = self.__getattr__(f"venc_{i+1}")
                feature = venc.latent(Xs[i], mask_idx=mask_idx)
                vspecific_features.append(feature)
        if single:
            return vspecific_features[self.args.vspecific.best_view]  
        return vspecific_features
    

    
    @torch.no_grad()
    def predict(self, Xs, mask_idx=None):
        _, probs = self.consis_enc(Xs, train=False)
        y = F.log_softmax(probs, dim=1)
        return y.detach().cpu().argmax(dim=1)

    