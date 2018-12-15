from __future__ import print_function
import argparse
import os
import torch
from model import Model
from video_dataset import Dataset
from test import test
from train import train
from tensorboard_logger import Logger
import options
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.optim as optim

if __name__ == '__main__':

    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    
    dataset = Dataset(args)
    if not os.path.exists('./ckpt/'):
       os.makedirs('./ckpt/')
    if not os.path.exists('./logs/' + args.model_name):
       os.makedirs('./logs/' + args.model_name)
    logger = Logger('./logs/' + args.model_name)
    
    model = Model(dataset.feature_size, dataset.num_class).to(device)

    if args.pretrained_ckpt is not None:
       model.load_state_dict(torch.load(args.pretrained_ckpt))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    
    for itr in range(args.max_iter):
       train(itr, dataset, args, model, optimizer, logger, device)
       if  itr % 500 == 0 and not itr == 0:
          torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
          test(itr, dataset, args, model, logger, device)
    
