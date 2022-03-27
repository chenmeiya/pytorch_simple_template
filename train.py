import argparse
import os.path

from torch import nn
import torch
from torch.utils.data import DataLoader
from model.network import Model
from tensorboardX import SummaryWriter
import numpy as np
from dataset.loadDataset import Dataset
from utils.metric import PSNR

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('--batch_size', default=4, type=int,
                    help='batch size in training')
parser.add_argument('--train_root', default=None, type=str, required=True,
                    help='path to training dataset')
parser.add_argument('--test_root', default=None, type=str, required=True,
                    help='path to test dataset')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--pre_trained', default=None, type=str)
parser.add_argument('--exp_dir', default='./experiments', type=str)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--num_work', default=4, type=int)
args = parser.parse_args()


class Trainer():
    def __init__(self, args):
        # param
        self.epoch = args.epoch
        self.pre_trained = args.pre_trained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = os.path.join(args.exp_dir, 'checkpoint')
        self.log_dir = os.path.join(args.exp_dir, 'logs')
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # load dataset
        train_data = Dataset(args.train_root, 20)
        val_data = Dataset(args.test_root, 20)
        self.train_load = DataLoader(train_data, num_workers=args.num_work, batch_size=args.batch_size,
                                     shuffle=False)
        self.val_data = DataLoader(val_data, num_workers=args.num_work, batch_size=1,
                                   shuffle=False)
        # load network
        self.net = Model()
        self.net = self.net.to(self.device)
        self.net.initialize()

        # optimizer
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr)

        # loss
        self.loss = nn.L1Loss()

        # writer
        self.writer = SummaryWriter(self.log_dir)
        img = torch.rand(1, 3, 64, 64).to(self.device)
        self.writer.add_graph(self.net, img)

    # training process
    def __call__(self):
        # load pre-trained model
        start_epoch = 0
        if self.pre_trained is not None:
            checkpoint = torch.load(os.path.join(self.pre_trained))
            self.net.load_state_dict(checkpoint['weight'])
            start_epoch = checkpoint['epoch']
            print('Load pretrained model success.')
        print('Start training from epoch %d' % start_epoch)

        for epoch in range(start_epoch, args.epoch):
            self.net.train()
            iter_count = 0
            for data in self.train_load:
                iter_count += 1
                self.net.zero_grad()
                self.optimizer.zero_grad()

                # data prepare
                gt, in_img = data
                in_img = in_img.to(self.device)
                gt = gt.to(self.device)

                # pred
                pred = self.net(in_img)

                # backward
                loss = self.loss(pred, gt)
                loss.backward()
                self.optimizer.step()

                # print
                if np.mod(iter_count, 50) == 0:
                    print('Epoch %d: step cout %d, loss is %f' % (epoch, iter_count, loss))

            # save model
            ckpt = {
                'weight': self.net.state_dict(),
                'epoch': epoch
            }
            torch.save(ckpt, os.path.join(self.ckpt_dir, str(epoch) + '.pkl'))

            # record
            self.writer.add_scalar('loss', global_step=epoch, scalar_value=loss)
            for name, param in self.net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=epoch)

            # evaluation
            count_iter = 0
            total_psnr = 0
            self.net.eval()
            for test_data in self.val_data:
                gt, noisy = test_data
                noisy = noisy.to(self.device)
                with torch.no_grad():
                    pred = self.net(noisy)
                psnr = PSNR(gt, pred)
                total_psnr += psnr
                count_iter += 1
            print('Evaluation: Average PSNR is %.4fdB' % (total_psnr/count_iter))

        print('Finished training.')


if __name__ == '__main__':
    train = Trainer(args)
    train()
