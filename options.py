import argparse

parser = argparse.ArgumentParser(description='WTALC')
parser.add_argument('--lr', type=float, default=0.0001,help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model-name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num-class', default=20, help='number of classes (default: )')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on (default: )')
parser.add_argument('--max-seqlen', type=int, default=750, help='maximum sequence length during training (default: 750)')
parser.add_argument('--Lambda', type=float, default=0.5, help='weight on Co-Activity Loss (default: 0.5)')
parser.add_argument('--num-similar', default=3, help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=100000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used I3D or UNT (default: I3D)')


