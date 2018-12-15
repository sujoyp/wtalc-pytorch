import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class Model(torch.nn.Module):
    def __init__(self, n_feature, n_class):
        super(Model, self).__init__()

        self.fc = nn.Linear(n_feature, n_feature)
        self.fc1 = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, n_class)
        self.dropout = nn.Dropout(0.7)

        self.apply(weights_init)

        #self.train()

    def forward(self, inputs, is_training=True):

        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)
        #x = F.relu(self.fc1(x))
        #if is_training:
        #    x = self.dropout(x)

        
        return x, self.classifier(x)
