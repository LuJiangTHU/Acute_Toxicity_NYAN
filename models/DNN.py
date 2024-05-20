import torch
import torch.nn as nn

class FCL(nn.Module):  # 全连接层
    def __init__(self, in_features, out_features, dropout_prob = 0):
        super(FCL, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Linear',nn.Linear(in_features=in_features,
                                                  out_features=out_features,
                                                  bias=True))
        self.layers.add_module('BatchNorm',nn.BatchNorm1d(out_features))
        self.layers.add_module('Dropout', nn.Dropout(dropout_prob))
        # self.layers.add_module('Relu', nn.Tanh())
        self.layers.add_module('Relu', nn.ReLU())

    def forward(self, x):
        out  = self.layers(x)
        return out


class DNN(nn.Module):  # 多任务学习器
    def __init__(self, opt):
        super(DNN, self).__init__()
        self.in_features = opt['in_features']
        self.out_features = opt['out_features']
        self.num_layers = opt['num_layers']
        self.Dropout_p = opt['Dropout_p']

        if type(self.out_features) == int:
            self.out_features = [self.out_features for _ in range(self.num_layers)]
        assert (type(self.out_features) ==list and len(self.out_features) == self.num_layers)


        if type(self.Dropout_p)==float:
            self.Dropout_p =[self.Dropout_p for _ in range(self.num_layers)]
        assert(type(self.Dropout_p)==list and len(self.Dropout_p)==self.num_layers)

        num_features = [self.in_features, ] + self.out_features
        FC_layers = []
        for i in range(self.num_layers):
            FC_layers.append(FCL(num_features[i], num_features[i+1], dropout_prob=self.Dropout_p[i]))

        self.FC_layers = nn.Sequential(*FC_layers)

    def forward(self, x):
        out = self.FC_layers(x)
        return out

def create_model(opt):
    return DNN(opt)


