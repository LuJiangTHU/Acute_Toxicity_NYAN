import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self, opt):
        super(Regressor, self).__init__()
        self.in_features = opt['in_features']
        self.out_tasks = opt['out_tasks']
        self.W = nn.Parameter(torch.rand(self.in_features, self.out_tasks)-0.5,
                              requires_grad=True)
        # self.edge_weights = opt['edge_weights']

    def forward(self, x):
        out = torch.mm(x, self.W)
        return out


    def regression_loss(self, pred, target, target_mask):
        '''
        :param pred: (batch_size, num_tasks), float
        :param target: (batch_size, num_tasks), float
        :param target_mask: (batch_size, num_tasks), 0 or 1
        :return: MSE averaged on batch
        '''
        mse = torch.mul((pred - target).pow(2), target_mask)
        loss = mse.sum()/target_mask.sum()
        return loss

    # def regularization_loss(self):
    #     W_dis = F.pdist(self.W.t(),p=2)
    #     return torch.mean(W_dis * self.edge_weights)




def create_model(opt):
    return Regressor(opt)