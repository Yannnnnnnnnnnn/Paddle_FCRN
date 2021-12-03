import os
import time
import argparse
import configparser
import datetime

import paddle
import paddle.nn
import paddle.nn.utils

from nyu_dataset import getTrainingTestingDataset
from model import ResNet50UpProj
from loss import huber_loss
from eval import mean_relative_error,root_squd_error,log10_error

def parse_arguments():
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()
    return args

paddle.set_device('gpu')

def main(args):

    epochs = args.epochs
    lr = args.lr
    bs = args.bs

    model = ResNet50UpProj()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)
    
    traindataset, valdataset = getTrainingTestingDataset()
    train_loader = paddle.io.DataLoader(traindataset, batch_size=bs)
    val_loader = paddle.io.DataLoader(valdataset, batch_size=bs)

    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    for epoch in range(epochs):

        N = len(train_loader)

        model.train()
        for i, sampled_batch in enumerate(train_loader):

            # train
            depth_pred = model(sampled_batch['image'])
            depth_gt = sampled_batch['depth']

            loss = huber_loss(depth_gt, depth_pred)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # validation
        model.eval()
        for i, sampled_batch in enumerate(val_loader):

            val_depth_pred = model(sampled_batch['image'])
            val_depth_gt = sampled_batch['depth']

            val_mae = mean_relative_error(val_depth_pred.numpy(),val_depth_gt.numpy())
            val_rmse = root_squd_error(val_depth_pred.numpy(),val_depth_gt.numpy())
            val_log10 = log10_error(val_depth_pred.numpy(),val_depth_gt.numpy())
        
            print('val_mae:',val_mae)
            print('val_rmse:',val_rmse)
            print('val_log10:',val_log10)

        # save
        paddle.save(model.state_dict(), "./logs/DenseDepth_epochs_{}.pdparams".format(epoch))
        paddle.save(optimizer.state_dict(), "./logs/Adam_epochs_{}.pdopt".format(epoch))


if __name__ == '__main__':
    pass