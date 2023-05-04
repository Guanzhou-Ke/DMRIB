
import numpy as np
import torch


class DCLLoss(torch.nn.Module):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    SMALL_NUM = np.log(1e-45)
    
    def forward(self, z1, z2):
        return self.get_loss(z1, z2)
    
    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCLLoss, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
    

    def get_loss(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * self.SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()


class DCLWLoss(DCLLoss):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLWLoss, self).__init__(weight_fn=weight_fn, temperature=temperature)
        

if __name__ == "__main__":
    pass
    # z1, z2 = torch.rand(10, 512), torch.rand(10, 512)
    
    # # for DCL
    # loss_fn = DCLLoss(temperature=0.5)
    # print("[DCL] loss(z1, z1):", loss_fn(z1, z1), "loss(z1, z2):", loss_fn(z1, z2))

    # # for DCLW
    # loss_fn = DCLWLoss(temperature=0.5, sigma=0.5)
    # print("[DCLW] loss(z1, z1):", loss_fn(z1, z1), "loss(z1, z2):", loss_fn(z1, z2))