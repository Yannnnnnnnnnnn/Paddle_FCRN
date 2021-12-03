# 2021-12-3
# created by Yan
# following code from https://github.com/iro-cp/FCRN-DepthPrediction/tree/master/tensorflow

import paddle
import paddle.nn as nn

def interleave(tensors,axis):

    old_shape = tensors[0].shape[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)

    tensor = paddle.stack(tensors,axis=axis+1)
    tensor = tensor.reshape(new_shape)
    
    return tensor

class unpool_as_conv(nn.Layer):

    def __init__(self,in_channels=None,out_channels=None,BN=True,RELU=True):
        super().__init__()

        self.conv_a = nn.Conv2D(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.conv_b = nn.Conv2D(in_channels=in_channels,out_channels=out_channels,kernel_size=(2,3),stride=1,padding=[[0,0],[0,0],[1,0],[1,1]])
        self.conv_c = nn.Conv2D(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,2),stride=1,padding=[[0,0],[0,0],[1,1],[1,0]])
        self.cond_d = nn.Conv2D(in_channels=in_channels,out_channels=out_channels,kernel_size=(2,2),stride=1,padding=[[0,0],[0,0],[1,0],[1,0]])

        self.BN = BN
        self.bn = nn.BatchNorm2D(out_channels)

        self.RELU = RELU
        self.relu = nn.ReLU()
    
    def interleave(self,tensors,axis):
        old_shape = tensors[0].shape[1:]
        new_shape = [-1] + old_shape
        new_shape[axis] *= len(tensors)

        tensor = paddle.stack(tensors,axis=axis+1)
        tensor = tensor.reshape(new_shape)
        
        return tensor

    def forward(self,x):

        a = self.conv_a(x)
        b = self.conv_b(x)
        c = self.conv_c(x)
        d = self.cond_d(x)

        left = self.interleave([a,b],axis=2)
        righ = self.interleave([c,d],axis=2)
        out = self.interleave([left,righ],axis=3)

        if self.BN:
            out = self.bn(out)

        if self.RELU:
            out = self.relu(out)

        return out

class UpProj(nn.Layer):

    def __init__(self,in_channels=None,out_channels=None,BN=True):
        super().__init__()

        self.out = unpool_as_conv(in_channels=in_channels,out_channels=out_channels,BN=True,RELU=True)
        self.conv = nn.Conv2D(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)

        self.out_2 = unpool_as_conv(in_channels=in_channels,out_channels=out_channels,BN=True,RELU=False)

        self.BN = BN
        self.bn = nn.BatchNorm2D(out_channels)

        self.relu = nn.ReLU()

    def forward(self,x):
        
        out = self.out(x)
        out = self.conv(out)
        if self.BN:
            out = self.bn(out)
        
        out_2 = self.out_2(x)

        out = self.relu(out+out_2)

        return out

class ResNet50UpProj(nn.Layer):

    def __init__(self):
        super().__init__()

        # block 0
        # input x
        self.pool1 = nn.Sequential(
            nn.Conv2D(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=4),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3,stride=2)
        )
        self.bn2a_branch1 = nn.Sequential(
            nn.Conv2D(in_channels=64,out_channels=256,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(256)
        )

        # block 1
        # input : pool1
        self.bn2a_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64,out_channels=256,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(256)
        )

        # block 2
        # input bn2a_branch1+bn2a_branch2c
        self.bn2b_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=256,out_channels=64,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64,out_channels=256,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(256)
        )

        # block 3
        # input relu(bn2a_branch1+bn2a_branch2c)+bn2b_branch2c
        self.bn2c_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=256,out_channels=64,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64,out_channels=256,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(256)
        )

        # block 4
        self.bn3a_branch1 = nn.Sequential(
            nn.Conv2D(in_channels=256,out_channels=512,kernel_size=1,stride=2,padding=0,bias_attr=False),
            nn.BatchNorm2D(512)
        )

        # block 5
        self.bn3a_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=256,out_channels=128,kernel_size=1,stride=2,padding=0,bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(in_channels=128,out_channels=512,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(512)
        )

        # block 6
        self.bn3b_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=512,out_channels=128,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(in_channels=128,out_channels=512,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(512)
        )

        # block 7
        self.bn3c_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=512,out_channels=128,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(in_channels=128,out_channels=512,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(512)
        )

        # block 8
        self.bn3d_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=512,out_channels=128,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(in_channels=128,out_channels=128,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(in_channels=128,out_channels=512,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(512)
        )

        # block 9
        self.bn4a_branch1 = nn.Sequential(
            nn.Conv2D(in_channels=512,out_channels=1024,kernel_size=1,stride=2,padding=0,bias_attr=False),
            nn.BatchNorm2D(1024)
        )

        # block 10
        self.bn4a_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=512,out_channels=256,kernel_size=1,stride=2,padding=0,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=1024,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(1024)
        )

        # block 11
        self.bn4b_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=1024,out_channels=256,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=1024,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(1024)
        )

        # block 12
        self.bn4c_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=1024,out_channels=256,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=1024,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(1024)
        )
        
        # block 13
        self.bn4d_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=1024,out_channels=256,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=1024,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(1024)
        )

        # block 14
        self.bn4e_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=1024,out_channels=256,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=1024,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(1024)
        )

        # block 15
        self.bn4f_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=1024,out_channels=256,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(in_channels=256,out_channels=1024,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(1024)
        )

        # block 16
        self.bn5a_branch1 = nn.Sequential(
            nn.Conv2D(in_channels=1024,out_channels=2048,kernel_size=1,stride=2,padding=0,bias_attr=False),
            nn.BatchNorm2D(2048)
        )

        # block 17
        self.bn5a_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=1024,out_channels=512,kernel_size=1,stride=2,padding=0,bias_attr=False),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(in_channels=512,out_channels=2048,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(2048)
        )

        # block 18
        self.bn5b_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=2048,out_channels=512,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(in_channels=512,out_channels=2048,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(2048)
        )
    
        # block 19
        self.bn5c_branch2c = nn.Sequential(
            nn.Conv2D(in_channels=2048,out_channels=512,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,bias_attr=False),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.Conv2D(in_channels=512,out_channels=2048,kernel_size=1,stride=1,padding=0,bias_attr=False),
            nn.BatchNorm2D(2048)
        )


        self.layer1_BN = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(in_channels=2048,out_channels=1024,kernel_size=1,stride=1,padding=0,bias_attr=True),
            nn.BatchNorm2D(1024)
        )

        self.up_2x = UpProj(in_channels=1024,out_channels=512,BN=True)
        self.up_4x = UpProj(in_channels=512,out_channels=256,BN=True)
        self.up_8x = UpProj(in_channels=256,out_channels=128,BN=True)
        self.up_16x = UpProj(in_channels=128,out_channels=64,BN=True)
        self.convpred = nn.Conv2D(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1,bias_attr=True)


        self.relu  = nn.ReLU()


    def forward(self,x):
        
        # block 0
        pool1 = self.pool1(x)
        bn2a_branch1 = self.bn2a_branch1(pool1)

        # block 1
        bn2a_branch2c = self.bn2a_branch2c(pool1)

        # block 2-4
        res2a_relu = self.relu(bn2a_branch1+bn2a_branch2c)
        bn2b_branch2c = self.bn2b_branch2c(res2a_relu)
        res2b_relu = self.relu(res2a_relu+bn2b_branch2c)
        bn2c_branch2c = self.bn2c_branch2c(res2b_relu)

        res2c_relu = self.relu(res2b_relu+bn2b_branch2c)
        bn3a_branch1 = self.bn3a_branch1(res2c_relu)
        bn3a_branc2c = self.bn3a_branch2c(res2c_relu)
        res3a_relu = self.relu(bn3a_branch1+bn3a_branc2c)
        bn3b_branch2c = self.bn3b_branch2c(res3a_relu)
        res3b_relu = self.relu(res3a_relu+bn3b_branch2c)
        bn3c_branch2c = self.bn3c_branch2c(res3b_relu)
        res3c_relu = self.relu(res3b_relu+bn3c_branch2c)
        bn3d_branch2c = self.bn3d_branch2c(res3c_relu)

        res3d_relu = self.relu(res3c_relu+bn3d_branch2c)
        bn4a_branch1 = self.bn4a_branch1(res3d_relu)
        bn4a_branch2c = self.bn4a_branch2c(res3d_relu)
        res4a_relu = self.relu(bn4a_branch1+bn4a_branch2c)
        bn4b_branch2c = self.bn4b_branch2c(res4a_relu)
        res4b_relu = self.relu(res4a_relu+bn4b_branch2c)
        bn4c_branch2c = self.bn4c_branch2c(res4b_relu)
        res4c_relu = self.relu(res4b_relu+bn4c_branch2c)
        bn4d_branch2c = self.bn4d_branch2c(res4c_relu)
        res4d_relu = self.relu(res4c_relu+bn4d_branch2c)
        bn4e_branch2c = self.bn4e_branch2c(res4d_relu)
        res4e_relu = self.relu(res4d_relu+bn4e_branch2c)
        bn4f_branch2c = self.bn4f_branch2c(res4e_relu)

        res4f_relu = self.relu(res4e_relu+bn4f_branch2c)
        bn5a_branch1 = self.bn5a_branch1(res4f_relu)
        bn5a_branch2c = self.bn5a_branch2c(res4f_relu)

        res5a_relu = self.relu(bn5a_branch1+bn5a_branch2c)
        bn5b_branch2c = self.bn5b_branch2c(res5a_relu)

        res5b_relu = self.relu(res5a_relu+bn5b_branch2c)
        bn5c_branch2c = self.bn5c_branch2c(res5b_relu)

        layer1_BN = self.layer1_BN(res5b_relu+bn5c_branch2c)
        up_2x = self.up_2x(layer1_BN)
        up_4x = self.up_4x(up_2x)
        up_8x = self.up_8x(up_4x)
        up_16x = self.up_16x(up_8x)
        convpred = self.convpred(up_16x)

        return convpred
    

if __name__ == '__main__':

    x = paddle.rand((1,3,228,304))
    model = ResNet50UpProj()
    r = model(x)
    print(r.shape)

