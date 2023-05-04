
import os

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as transforms

from utils.augment import Augment, Cutout, RGB2YUV, RGB2Lab, RGB2YCbCr, RGB2LUV


DEFAULT_DATA_ROOT = './data'
PROCESSED_DATA_ROOT = os.path.join(DEFAULT_DATA_ROOT, 'processed')
RAW_DATA_ROOT = os.path.join(DEFAULT_DATA_ROOT, 'raw')

# -------------------------------------------------------------------------------
#                      Functional area.
# -------------------------------------------------------------------------------



def plain_transforms(img):
    """
    plain transformation operation.
    """
    return img



def coil(root, n_objs=20, n_views=3):
    """
    Download: 
    https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php
    
    1. coil-20: 
    http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip
    
    
    2. coil-100:
    http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip
    """
    from skimage.io import imread
    from sklearn.model_selection import train_test_split
    assert n_objs in [20, 100]
    data_dir = os.path.join(root, f"coil-{n_objs}")
    img_size = (1, 128, 128) if n_objs == 20 else (3, 128, 128)
    n_imgs = 72

    n = (n_objs * n_imgs) // n_views

    views = []
    labels = []

    img_idx = np.arange(n_imgs)

    for obj in range(n_objs):
        obj_list = []
        obj_img_idx = np.random.permutation(img_idx).reshape(n_views, n_imgs // n_views)
        labels += (n_imgs // n_views) * [obj]

        for view, indices in enumerate(obj_img_idx):
            sub_view = []
            for i, idx in enumerate(indices):
                if n_objs == 20:
                    fname = os.path.join(data_dir, f"obj{obj + 1}__{idx}.png")
                    img = imread(fname)[None, ...]
                else:
                    fname = os.path.join(data_dir, f"obj{obj + 1}__{idx * 5}.png")
                    img = imread(fname)
                if n_objs == 100:
                    img = np.transpose(img, (2, 0, 1))
                sub_view.append(img)
            obj_list.append(np.array(sub_view))
        views.append(np.array(obj_list))
    views = np.array(views)
    views = np.transpose(views, (1, 0, 2, 3, 4, 5)).reshape(n_views, n, *img_size)
    cp = views.reshape(-1, *img_size)
    # print(cp.shape)
    # print(cp[:, 0, :].mean(), cp[:, 0, :].std())
    labels = np.array(labels)
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(list(range(n)), labels, test_size=0.2, random_state=42)
    return views[:, X_train_idx, :, :, :], views[:, X_test_idx, :, :, :], y_train, y_test


def get_train_transformations(args, task='pretext'):
    need_hflip = args['training_augmentation'].hflip
    if task == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**args['training_augmentation']['random_resized_crop']),
            transforms.RandomHorizontalFlip() if need_hflip else plain_transforms,
            transforms.ToTensor(),
            transforms.Normalize(**args['training_augmentation']['normalize'])
        ])
    elif task == 'pretext':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**args['training_augmentation']['random_resized_crop']),
            transforms.RandomHorizontalFlip() if need_hflip else plain_transforms,
            transforms.RandomApply([
                transforms.ColorJitter(**args['training_augmentation']['color_jitter'])
            ], p=args['training_augmentation']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**args['training_augmentation']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**args['training_augmentation']['normalize'])
        ])
    elif task == 'selflabel':
        return transforms.Compose([
            transforms.RandomCrop(args['training_augmentation']['crop_size']),
            # transforms.CenterCrop(args.training_augmentation.crop_size),
            transforms.RandomHorizontalFlip() if need_hflip else plain_transforms,
            # transforms.RandomResizedCrop(**args['training_augmentation']['random_resized_crop']),
            Augment(args['training_augmentation']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**args['training_augmentation']['normalize']),
            Cutout(n_holes=args['training_augmentation']['cutout_kwargs']['n_holes'],
                   length=args['training_augmentation']['cutout_kwargs']['length'],
                   random=args['training_augmentation']['cutout_kwargs']['random'])])
    else:
        raise ValueError(f'Invalid augmentation strategy {task}')


def get_val_transformations(args):
    return transforms.Compose([
            # transforms.CenterCrop(args.valid_augmentation.crop_size),
            transforms.Resize(args.valid_augmentation.crop_size),
            transforms.ToTensor(), 
            transforms.Normalize(**args['valid_augmentation']['normalize']) if args.valid_augmentation.use_normalize else plain_transforms])


def edge_transformation(img):
    """
    edge preprocess functuin.
    """
    trans = transforms.Compose([image_edge,transforms.ToPILImage()])
    return trans(img)


def image_edge(img):
    """
    :param img:
    :return:
    """
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return edge


def generate_tiny_dataset(name, dataset, sample_num=100):
    """
    Tiny data set for T-SNE to visualize the representation's structure.
    Only support EdgeMNIST, FashionMNIST.
    """
    assert name in ['EdgeMnist', 'FashionMnist']
    y = dataset.targets.unique()
    x1s = []
    x2s = []
    ys = []

    for _ in y:
        idx = dataset.targets == _
        x1, x2, yy = dataset.data[0][idx, :], dataset.data[1][idx, :], dataset.targets[idx]
        x1, x2, yy = x1[:sample_num], x2[:sample_num], yy[:sample_num]
        x1s.append(x1)
        x2s.append(x2)
        ys.append(yy)
    
    x1s = torch.vstack(x1s)
    x2s = torch.vstack(x2s)
    ys = torch.concat(ys)
    
    tiny_dataset = {
        "x1": x1s,
        "x2": x2s,
        "y": ys
    }
    os.makedirs("./experiments/tiny-data/", exist_ok=True)
    torch.save(tiny_dataset, f'./experiments/tiny-data/{name}_tiny.plk')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

# -------------------------------------------------------------------------------
#                      Dataset Area.
# -------------------------------------------------------------------------------

class IncompleteMultiviewDataset(Dataset):
    # TODO: 在训练consistency encoder的时候，我们需要返回不缺失的样本，这样才能得到一致性。否则没法建模，
    # 而在训练view-specific encoder的时候，我们就根据有缺失的来即可。
    def __init__(self) -> None:
        super().__init__()
      

class AugmentDataset(Dataset):
    
    def __init__(self, dataset, weak_aug, strong_aug) -> None:
        super().__init__()
        
        self.dataset = dataset
        
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        Xs, targets = self.dataset.__getitem__(index)
        weak_xs = [self.weak_aug(x) for x in Xs]
        strong_xs = [self.strong_aug(x) for x in Xs]
        
        return weak_xs, strong_xs, targets
        

class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor_Xs, targets = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor_Xs, _ = self.dataset.__getitem__(neighbor_index)

        anchor_Xs = [self.anchor_transform(x) for x in anchor_Xs]
        neighbor_Xs = [self.neighbor_transform(x) for x in neighbor_Xs]

        # output['anchor'] = anchor_Xs
        # output['neighbor'] = neighbor_Xs
        # output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        # output['target'] = targets
        
        return anchor_Xs, neighbor_Xs, torch.from_numpy(self.indices[index]), targets
        

class EdgeMNISTDataset(torchvision.datasets.MNIST):
    """
    """
    def __init__(self, 
                 root, 
                 train=True, 
                 transform=None, 
                 target_transform=None, 
                 download=False,
                 views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        
    
    def __getitem__(self, idx):
        
        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        
        # original-view transforms
        view0 = img
        # edge-view transforms
        view1 = edge_transformation(img)
        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return [view0, view1], self.targets[idx]
    

class EdgeFMNISTDataset(torchvision.datasets.FashionMNIST):
        
    def __init__(self, root: str, train: bool = True, 
                    transform=None, 
                    target_transform=None, download: bool = False, views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        
        # original-view transforms
        view0 = img
        # edge-view transforms
        view1 = edge_transformation(img)
        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return [view0, view1], self.targets[idx]


class Cifar10Dataset(torchvision.datasets.cifar.CIFAR10):
    """
    A three views version of Cifar10.
    """
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False, views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.rgb2view1 = RGB2LUV()
        # self.rgb2view2 = RGB2YCbCr()
        self.to_pil = transforms.ToPILImage()
        
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        
        # original-view transforms
        rgb = img
        # L,ab transforms
        view1 = self.to_pil(self.rgb2view1(img))
        # Y,uv transforms
        # view2 = self.to_pil(self.rgb2view2(img))
        if self.transform:
            rgb = self.transform(rgb)
            view1 = self.transform(view1)
            # view2 = self.transform(view2)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # return [rgb, view1, view2], self.targets[idx]
        return [rgb, view1], self.targets[idx]


class COIL20Dataset(Dataset):
    
    def __init__(self, root: str, train: bool = True, 
                    transform=None, 
                    target_transform=None, download: bool = False, views=2) -> None:
        
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(20))
        self.train = train
        self.views = views
        self.to_pil = transforms.ToPILImage()
        X_train, X_test, y_train, y_test = coil(self.root, n_objs=20, n_views=self.views)
        if self.train:
            self.data = X_train
            self.targets = torch.from_numpy(y_train).long()
        else:
            self.data = X_test
            self.targets = torch.from_numpy(y_test).long()
            
    
    def __getitem__(self, index):
        views = [np.transpose(self.data[view, index, :], (1, 2, 0)) for view in range(self.views)]
        target = self.targets[index]
        
        views = [self.to_pil(v) for v in views]
        
        if self.transform:
            views = [self.transform(x) for x in views]
            
        if self.target_transform:
            target = self.target_transform(target)
            
        return views, target
    
    
    def __len__(self) -> int:
        return self.data.shape[1]
    
    


class MultiViewClothingDataset(Dataset):
    """
    Refers to: Kuan-Hsien Liu, Ting-Yen Chen, and Chu-Song Chen. 
    MVC: A Dataset for View-Invariant Clothing Retrieval and Attribute Prediction, ACM ICMR 2016.
    Total: 161260 images. 10 classes. 
    (In fact, I found that it has many fails when I downloaded them. So, subject to the actual number)
    The following the size of number is my actual number:
        2 views train size: 27543, test size: 6886
        3 views train size: 26200, test size: 6550
        4 views train size: 25292, test size: 6323
        5 views train size: 7111, test size: 1778
    """
  
    def __init__(self, root: str, train: bool = True, 
                    transform=None, 
                    target_transform=None, download: bool = False, views=2) -> None:
        
        super().__init__()
        self.classes_name = {
            "Shirts & Tops": 0,
            "Coats & Outerwear": 1,
            "Pants": 2,
            "Dresses": 3,
            "Underwear & Intimates": 4,
            "Jeans": 5,
            "Sweaters": 6,
            "Swimwear": 7,
            "Sleepwear": 8,
            "Underwear": 9
        }
        self.target2class = {v: k for k, v in self.classes_name.items()}
        self.classes = [k for k, v in self.classes_name.items()]
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.views = views
        # image loader.
        self.loader = pil_loader
        
        self.indices = torch.load(os.path.join(self.root, f'{self.views}V-indices.pth'))
        self.data, self.targets = self.indices['train'] if self.train else self.indices['test']
        
        
    def __getitem__(self, index: int):
        try:
            raw_data = [self.loader(os.path.join(self.root, path)) for path in self.data[index]['path']]
        except:
            print([os.path.join(self.root, path) for path in self.data[index]['path']])
            raise 
        
        if self.transform:
            views_data = [self.transform(x) for x in raw_data]
        else:
            views_data = raw_data
        target = torch.tensor(self.classes_name[self.targets[index]]).long()
        if self.target_transform:
            target = self.target_transform(target)
        
        return views_data, target
    
    
    def __len__(self) -> int:
        return len(self.data)
        
    

__dataset_dict = {
    'EdgeMnist': EdgeMNISTDataset,
    'FashionMnist': EdgeFMNISTDataset,
    'cifar10': Cifar10Dataset,
    'mvc-10': MultiViewClothingDataset,
    'coil-20': COIL20Dataset,
}


def get_train_dataset(args, transform, neighbors=False, topk_neighbors_train_path=None):
    data_class = __dataset_dict.get(args.dataset.name, None)
    if data_class is None:
        raise ValueError("Dataset name error.")
    if isinstance(transform, dict):
        train_set = data_class(root=args.dataset.root, train=True, download=True, views=args.views)
        train_set = AugmentDataset(train_set, transform['weak'], transform['strong'])
    else:
        train_set = data_class(root=args.dataset.root, train=True, transform=transform, download=True, views=args.views)
    
    if neighbors:
        indices = np.load(topk_neighbors_train_path)
        train_set = NeighborsDataset(train_set, indices, args.selflabel.num_neighbors)
        
    return train_set


def get_val_dataset(args, transform, neighbors=False, topk_neighbors_val_path=None):
    data_class = __dataset_dict.get(args.dataset.name, None)
    if data_class is None:
        raise ValueError("Dataset name error.")
    val_set = data_class(root=args.dataset.root, train=False, transform=transform, views=args.views)
    
    if neighbors:
        indices = np.load(topk_neighbors_val_path)
        val_set = NeighborsDataset(val_set, indices, 5) # Only use 5
        
    return val_set


if __name__ == '__main__':
    pass
    
    from configs.basic_config import get_cfg
    from torchvision.utils import make_grid
    args = get_cfg('/home/hades/notebooks/Experiment/RIM-CAC/src/configs/pretext/pretext_mvc10.yaml')
    
    
    train_trans = get_train_transformations(args)
    dataset = MultiViewClothingDataset(args.dataset.root, train=True, transform=train_trans, download=False, views=args.views)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, 8, True)
    
    samples = next(iter(loader))
    data = samples[0]
    
    grid = make_grid(torch.cat(data, dim=0), normalize=True)
    
    from matplotlib import pyplot as plt
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    

    
            
        
        
