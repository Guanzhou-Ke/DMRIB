import sys
# 3.8 supported
# from math import prod

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.cluster import KMeans
from munkres import Munkres


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            print(validation_loss)
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_masked(batch_size, shapes, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.
        Args:
          batch_size: the ba
          shapes: the shape of data.
          missing_rate: missing ratio. [0, 1]
        Returns: 
          mask: torch.ByteTensor
    """
    masks = []
    for shape in shapes:
        mask = np.r_[[np.random.choice([0, 1], size=shape, p=[1-missing_rate, missing_rate]) for _ in range(batch_size)]]
        masks.append(torch.BoolTensor(mask))
    return masks


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def reproducibility_setting(seed):
    """
    set the random seed to make sure reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    print('Global seed:', seed)

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(f'Total number of parameters: {num_params}, size: {num_params/1e6*32/8:.2f} M')


def clustering_by_representation(X_rep, y):
    """Get scores of clustering by representation"""
    n_clusters = np.size(np.unique(y))

    kmeans_assignments, _ = get_cluster_sols(X_rep, ClusterClass=KMeans, n_clusters=n_clusters,
                                              init_args={'n_init': 10, 'random_state': 42})
    if np.min(y) == 1:
        y = y - 1
    return clustering_metric(y, kmeans_assignments)
    


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    """Get classification metric"""
   
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    return accuracy, precision, f_score


def clustering_metric(y_true, y_pred, decimals=4):
    """Get clustering metric"""
    n_clusters = np.size(np.unique(y_true))
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)
    
    class_acc, p, fscore = classification_metric(y_true, y_pred_ajusted)
    
    # ACC
    acc = clustering_accuracy(y_true, y_pred)
    acc = np.round(acc, decimals)
    
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    return acc, nmi, ari, class_acc, p, fscore

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    """Computes the predicted labels, where label assignments now
        correspond to the actual labels in y_true (as estimated by Munkres)

        Args:
            cluster_assignments: array of labels, outputted by kmeans
            y_true:              true labels
            n_clusters:          number of clusters in the dataset

        Returns:
            a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """Using either a newly instantiated ClusterClass or a provided cluster_obj, generates
        cluster assignments based on input data.

        Args:
            x: the points with which to perform clustering
            cluster_obj: a pre-fitted instance of a clustering class
            ClusterClass: a reference to the sklearn clustering class, necessary
              if instantiating a new clustering class
            n_clusters: number of clusters in the dataset, necessary
                        if instantiating new clustering class
            init_args: any initialization arguments passed to ClusterClass

        Returns:
            a tuple containing the label assignments and the clustering object
    """
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj


def classify_via_svm(train_X, train_Y, test_X, test_Y, **kwargs):
    """
    A simple classifier for representation learning. (SVM)
    ---
    Args:
      train_X: (N, D) training matrix. 
      train_Y: (N, 1) training labels.
      test_X: (M, D) test matrix.
      test_Y: (M, 1) test labels.
      
    Return:
      A result values (@acc, @percision, @fscore)
    """
    # We suggest to use the default setting.
    clf = SVC(**kwargs)
    clf.fit(train_X, train_Y)
    preds = clf.predict(test_X)
    acc, p, fscore = classification_metric(test_Y, preds)
    return acc, p, fscore


def classify_via_vote(train_X, train_Y, test_X, test_Y, n=1):
    """
    References to: https://github.com/XLearning-SCU/2022-TPAMI-DCP/blob/26a67a693ab7392a8c9e002b96f90137ea7fd196/utils/classify.py#L9
    Sometimes the prediction accuracy will be higher in this way.
    ---
    Args: 
      train_X: train set's latent space data
      train_Y: label of train set
      test_X: test set's latent space data
      test_Y: label of test set
      n: Similar to K in k-nearest neighbors algorithm
    
    Return: 
      A result values (@acc, @percision, @fscore)
    """
    F_h_h = np.dot(test_X, np.transpose(train_X))
    gt_list = []
    train_Y = train_Y.reshape(len(train_Y), 1)
    for _ in range(n):
        F_h_h_argmax = np.argmax(F_h_h, axis=1)
        F_h_h_onehot = convert_to_one_hot(F_h_h_argmax, len(train_Y))
        F_h_h = F_h_h - np.multiply(F_h_h, F_h_h_onehot)
        gt_list.append(np.dot(F_h_h_onehot, train_Y))
    gt_ = np.array(gt_list).transpose(2, 1, 0)[0].astype(np.int64)
    count_list = []
    count_list.append([np.argmax(np.bincount(gt_[i])) for i in range(test_X.shape[0])])
    gt_pre = np.array(count_list).transpose()
    
    acc, p, fscore = classification_metric(test_Y, gt_pre)
    return acc, p, fscore
    
    
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def label_to_one_hot(label_idx, num_classes) -> torch.Tensor:
    return torch.nn.functional.one_hot(label_idx, 
                                       num_classes=num_classes)
    
def one_hot_to_label(one_hot_arr: torch.Tensor) -> torch.Tensor:
    return one_hot_arr.argmax(dim=1)


def weighted_sum(tensors, weights, normalize_weights=True):
    """
    Multi-view tensor weighted-sum fusion.
    ----
    :param: tensors: multi-view tensors list.  [B x D] x N
    :param: weights: fusion weights, tensor, N
    --- 
    return a fusion tensor, B x D.
    """
    if normalize_weights:
        weights = torch.nn.functional.softmax(weights, dim=0)
    out = torch.sum(weights[None, None, :] * torch.stack(tensors, dim=-1), dim=-1)
    return out


@torch.no_grad()
def knn_evaluation(val_loader, model, memory_bank, device):
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    model.eval()

    for Xs, target in val_loader:
        Xs, target = [x.to(device) for x in Xs], target.to(device)

        output = model.consistency_repr(Xs, normlization=True)
        total_num += output.size(0)
        output = memory_bank.weighted_knn(output) 
        
        total_top1 += torch.sum((output[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_top5 += torch.sum((output[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        # print(output.unique())
        # total_top1 = 100*torch.mean(torch.eq(output, target).float()).item()

    total_top1 = (total_top1 / total_num) * 100
    total_top5 = (total_top5 / total_num) * 100
    return {'knn_acc@1': total_top1, 'knn_acc@5': total_top5}


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, device):
    model.eval()
    memory_bank.reset()

    for i, (Xs, targets) in enumerate(loader):
        Xs, targets = [x.to(device) for x in Xs], targets.to(device)
        output = model.consistency_repr(Xs, normlization=True)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))


@torch.no_grad()
def scan_evaluate(predictions):
    from losses.scanloss import entropy
    # Evaluate model based on SCAN loss.
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1,1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()
        
        # Consistency loss       
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = torch.nn.functional.binary_cross_entropy(similarity, ones).item()
        
        # Total loss
        total_loss = - entropy_loss + consistency_loss
        
        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}            
 
            
@torch.no_grad()
def get_predictions(args, dataloader, model, have_neighbors=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = [[] for _ in range(args.selflabel.nheads)]
    probs = [[] for _ in range(args.selflabel.nheads)]
    targets = []
    neighbors = []
    if have_neighbors:
        for anchor, _, possible_neighbors, target in dataloader:
            anchor = [x.cuda(non_blocking=True) for x in anchor]
            _, output = model(anchor)
            for i, output_i in enumerate(output):
                predictions[i].append(torch.argmax(output_i, dim=1))
                probs[i].append(torch.nn.functional.softmax(output_i, dim=1))
            targets.append(target)
            neighbors.append(possible_neighbors)
    else:
        for Xs, target in dataloader:
            Xs = [x.cuda(non_blocking=True) for x in Xs]
            _, output = model(Xs)
            for i, output_i in enumerate(output):
                predictions[i].append(torch.argmax(output_i, dim=1))
                probs[i].append(torch.nn.functional.softmax(output_i, dim=1))
            targets.append(target)

    predictions = [torch.cat(pred_, dim = 0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if have_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors} for pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in zip(predictions, probs)]

    return out
    

@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, class_names=None, 
                        compute_purity=True, compute_confusion_matrix=True,
                        confusion_matrix_file=None):
    from .visualization import plot_confusion_matrix
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())
    
    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)

    # Compute confusion matrix
    if compute_confusion_matrix:
        plot_confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(), 
                              class_names, confusion_matrix_file)

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}

if __name__ == '__main__':
    pass
    
    
    