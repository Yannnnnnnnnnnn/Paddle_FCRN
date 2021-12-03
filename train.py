import os
import time
import argparse
import configparser
import datetime

import paddle
import paddle.nn
import paddle.nn.utils
paddle.disable_static()
paddle.set_device('gpu')

from loss import huber_loss
from model import ResNet50UpProj
from nyu_dataset import getTrainingTestingDataset
from eval import mean_relative_error,root_squd_error,log10_error


def main(args):

    epochs = args.epochs
    lr = args.lr
    bs = args.bs

    model = ResNet50UpProj()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)

    traindataset, valdataset = getTrainingTestingDataset(nyu_path=args.nyu_v2_path)
    train_loader = paddle.io.DataLoader(traindataset, batch_size=bs)
    val_loader = paddle.io.DataLoader(valdataset, batch_size=bs)

    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    
    train_log = open('./logs/train_log.log','w+')
    train_log.write('epoch: '+ str(epochs)+'\n')
    train_log.write('lr: '+ str(lr)+'\n')
    train_log.write('bs: '+str(bs)+'\n')
    train_log.flush()

    valid_log = open('./logs/valid_log.log','w+')
    valid_log.write('epoch: '+ str(epochs)+'\n')
    valid_log.write('lr: '+ str(lr)+'\n')
    valid_log.write('bs: '+str(bs)+'\n')
    valid_log.flush()

    for epoch in range(epochs):

        loss_sum = 0
        loss_cnt = 0
        model.train()
        for i, sampled_batch in enumerate(train_loader):

            # train
            depth_pred = model(sampled_batch['image'])
            depth_gt = sampled_batch['depth']

            loss = huber_loss(depth_gt, depth_pred)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            loss_sum += loss[0].numpy()[0]
            loss_cnt += 1.0

            print('TRAIN: progress: ',i," in ",len(train_loader),' loss:',loss[0].numpy()[0])
            break
        
        loss_sum /= loss_cnt
        train_log.write('epoch:'+str(epoch)+'\t loss:'+str(loss_sum)+'\n')
        train_log.flush()

        # validation
        val_mae_sum = 0
        val_rmse_sum = 0
        val_log10_sum = 0
        val_cnt = 0
        model.eval()
        for i, sampled_batch in enumerate(val_loader):

            val_depth_pred = model(sampled_batch['image'])
            val_depth_gt = sampled_batch['depth']

            val_mae = mean_relative_error(val_depth_pred.numpy(),val_depth_gt.numpy())
            val_rmse = root_squd_error(val_depth_pred.numpy(),val_depth_gt.numpy())
            val_log10 = log10_error(val_depth_pred.numpy(),val_depth_gt.numpy())
        
            val_mae_sum += val_mae
            val_rmse_sum += val_rmse
            val_log10_sum += val_log10
            val_cnt += 1

            print('VALID: progress: ',i," in ",len(val_loader),' rmse:',val_rmse)
            break

        val_mae_sum /= val_cnt
        val_rmse_sum /= val_cnt
        val_log10_sum /= val_cnt

        valid_log.write('epoch:'+str(epoch)+str(i)+'\t val_mae:'+str(val_mae_sum)+'\t val_rmse:'+str(val_rmse_sum)+'\t val_log10:'+str(val_log10_sum)+'\n')
        valid_log.flush()

        # save model
        paddle.save(model.state_dict(), "./logs/FCRN_epochs_{}_{}.pdparams".format(epoch,val_rmse_sum))
        paddle.save(optimizer.state_dict(), "./logs/Adam_epochs_{}_{}.pdopt".format(epoch,val_rmse_sum))

        break

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument('--nyu_v2_path', default='../nyu_data.zip', type=str, help='nyu v2 zip path')
    args = parser.parse_args()
    main(args)