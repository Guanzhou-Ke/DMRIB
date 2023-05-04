
import math

import torch
import numpy as np

    
def get_optimizer(params, lr=1e-3, op_name='adam'):

    if op_name == 'sgd':
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=False, weight_decay=0.0001)
    elif op_name == 'adam':
        opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0001, betas=[0.9, 0.999])
    elif op_name == 'adamw':
        opt = torch.optim.AdamW(params, lr=lr)
    else:
        raise ValueError('optimizer must be sgd or adam.')
    return opt


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.train.lr
    
    if args.train.scheduler == 'cosine':
        eta_min = lr * (args.train.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.train.epochs)) / 2
         
    elif args.train.scheduler == 'step':
        steps = np.sum(epoch > np.array(args.train.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.train.lr_decay_rate ** steps)

    elif args.train.scheduler == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(args.train.scheduler))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class AdaptedLossWeight:
    """
    impletement multiple losses weight.
    References to: https://github.com/rickgroen/cov-weighting/
    @inproceedings{groenendijk2020multi,
        title={Multi-Loss Weighting with Coefficient of Variations},
        author={Groenendijk, Rick and Karaoglu, Sezer and Gevers, Theo and Mensink, Thomas},
        booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
        pages={1469--1478},
        year={2020}
    }
    """
    def __init__(self, args, losses_num, device='cpu'):
        super(AdaptedLossWeight, self).__init__()
        self.args = args
        self.device = device
        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = True if self.args.train.mean_sort == 'decay' else False
        self.mean_decay_param = self.args.train.mean_decay_param
        self.num_losses = losses_num

        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_std_l = None

    def __call__(self, unweighted_losses):

        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
                self.device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)
        return loss