# 2021-12-3
# created by Yan
# following "Deeper Depth Prediction with Fully Convolutional Residual Networks" Section 3.2
import paddle

def huber_loss(gt,pre):

    # batch size
    B = gt.shape[0]

    huber_loss = 0

    for b in range(B):

        gt_b = gt[b]
        pre_b = pre[b]

        diff = (gt_b-pre_b).abs()

        huber_c = 0.2 * diff.detach().max()
        huber_mask = ( diff > huber_c ) * 1.0

        huber_loss_b = diff * ( 1 - huber_mask ) + huber_mask * ( diff*diff + huber_c*huber_c )/(huber_c+1e-6)

        huber_loss += huber_loss_b.mean()

    return huber_loss/B

    # diff = (gt - pre).abs()
    # return diff.mean()

if __name__ == '__main__':

    gt = paddle.rand((2,1,3,3))
    pre = paddle.rand((2,1,3,3))
    loss = huber_loss(gt,pre)

    print('loss',loss)
