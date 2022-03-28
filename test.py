import argparse
import os.path
from torch import nn
import torch
from torch.utils.data import DataLoader
from model.network import Model
from tensorboardX import SummaryWriter
import numpy as np
from dataset.loadDataset import LoadData
from utils.metric import PSNR

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('--test_root', type=str, required=True,
                    help='path to test dataset')
parser.add_argument('--save_path', type=str, required=True,
                    help='path to test dataset')
parser.add_argument('--pre_trained', type=str, required=True)
parser.add_argument('--num_work', default=4, type=int)
args = parser.parse_args()


class Tester():
    def __init__(self):
        # param
        self.pre_trained = args.pre_trained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # load dataset
        test_data = LoadData(args.test_root, 20)
        self.test_data = DataLoader(test_data, num_workers=args.num_work, batch_size=1,
                                    shuffle=False)
        # load network
        self.net = Model()
        self.net = self.net.to(self.device)
        self.net.initialize()

    # training process
    def __call__(self):
        # load pre-trained model
        if self.pre_trained is not None:
            checkpoint = torch.load(os.path.join(self.pre_trained))
            self.net.load_state_dict(checkpoint['weight'])
            start_epoch = checkpoint['epoch']
            print('Load pretrained model success.')
        else:
            raise AssertionError

        self.net.eval()
        count_iter = 0
        total_psnr = 0
        self.net.eval()
        for test_data in self.test_data:
            gt, noisy = test_data
            noisy = noisy.to(self.device)
            gt = gt.to(self.device)
            with torch.no_grad():
                pred = self.net(noisy)
            psnr = PSNR(gt, pred)
            total_psnr += psnr
            count_iter += 1
        print('Evaluation: Average PSNR is %.4fdB' % (total_psnr / count_iter))


if __name__ == '__main__':
    test = Tester()
    test()
