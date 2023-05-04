import argparse
import os

import torch
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from configs.basic_config import get_cfg
from models.rimcac import RIMCAC
from utils import print_network, visualization, label_to_one_hot



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args


def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    # device = torch.device('cpu')
    device = torch.device(
        f"cuda:{config.device}") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Use: {device}')
    model_path = os.path.join(config.train.log_dir, 'rimcac', 'final_model.pth')
    
    # Create contrastive model.
    model = RIMCAC(config, device=device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model = model.cuda()
    print_network(model)
    
    # cidx = [7, 5, 9, 2, 4, 3, 6, 1, 8, 0]
    cidx = [5] * 10
    
    # con_grid = visualization.conditional_sampling(model, cidx, sample_nums=4)
    # plt.imshow(con_grid.permute(1, 2, 0))
    # plt.axis('off')
    # plt.show()
    samples_nums = 8
    vidx = 1
    class_num = 10
    ys = []
    zs = []
    for label in cidx:
        lidx = torch.tensor(label).repeat(samples_nums)
        y = label_to_one_hot(label_idx=lidx, num_classes=class_num)
        z = traverse_continuous_line(vidx, samples_nums, 10, sample_prior=True)
        ys.append(y)
        zs.append(z)
    zs = torch.cat(zs, dim=0)
    ys = torch.cat(ys, dim=0)
    traverse_grid = visualization.sampling_by_z(model, ys, zs, nrow=samples_nums, device=device)
    plt.imshow(traverse_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    
    
def traverse_continuous_line(idx, size, cont_dim, begain=0.1, end=0.99, sample_prior=False):
        """
        Returns a (size, cont_dim) latent sample, corresponding to a traversal
        of a continuous latent variable indicated by idx.

        Parameters
        ----------
        idx : int or None
            Index of continuous latent dimension to traverse. If None, no
            latent is traversed and all latent dimensions are randomly sampled
            or kept fixed.

        size : int
            Number of samples to generate.
        """
        if sample_prior:
            samples = np.random.normal(size=(size, cont_dim))
        else:
            samples = np.zeros(shape=(size, cont_dim))

        if idx is not None:
            # Sweep over linearly spaced coordinates transformed through the
            # inverse CDF (ppf) of a gaussian since the prior of the latent
            # space is gaussian
            # cdf_traversal = np.linspace(0.05, 0.95, size)
            # cdf_traversal = np.linspace(0.01, 0.99, size)
            cdf_traversal = np.linspace(begain, end, size)
            cont_traversal = stats.norm.ppf(cdf_traversal)

            for i in range(size):
                samples[i, idx] = cont_traversal[i]

        return torch.Tensor(samples)    
    
if __name__ == '__main__':
    main()
    