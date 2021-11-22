import mxnet
import numpy as np
from numba import jit
from mxnet import nd, autograd
import os
from mxnet import initializer
from mxnet.gluon import *
from mxnet import init, nd
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet import base



# Blocks
class BasicBlockV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(nn.BatchNorm())
        if downsample:
            # self.downsample = nn.HybridSequential(prefix='')
            # self.downsample.add(nn.MaxPool2D(strides=2))
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual + x, act_type='relu')
        # print(residual.shape,x.shape)

        return x


class BottleneckV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels // 4, kernel_size=1, strides=stride))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels // 4, 1, channels // 4))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.body.add(nn.BatchNorm())
        if downsample:
            # self.downsample = nn.HybridSequential(prefix='')
            # self.downsample.add(nn.MaxPool2D(strides=2))
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x


class BasicBlockV2(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            # self.downsample = nn.HybridSequential(prefix='')
            # self.downsample.add(nn.MaxPool2D(strides=2))
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        return x + residual


class BottleneckV2(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels // 4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels // 4, stride, channels // 4)
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            # self.downsample = nn.HybridSequential(prefix='')
            # self.downsample.add(nn.MaxPool2D(strides=2))
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual


# Nets
class ResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features0 = nn.HybridSequential(prefix='')
            self.features0.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
            self.features0.add(nn.BatchNorm())
            self.features0.add(nn.Activation('relu'))
            self.features0.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2

                if i == 0:
                    self.features1 = self._make_layer(block, num_layer, channels[i + 1],
                                                      stride, i + 1, in_channels=channels[i])
                elif i == 1:
                    self.features2 = self._make_layer(block, num_layer, channels[i + 1],
                                                      stride, i + 1, in_channels=channels[i])
                elif i == 2:
                    self.features3 = self._make_layer(block, num_layer, channels[i + 1],
                                                      stride, i + 1, in_channels=channels[i])
                else:
                    self.features4 = self._make_layer(block, num_layer, channels[i + 1],
                                                      stride, i + 1, in_channels=channels[i])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers - 1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features0(x)#(12, 64, 112, 112)
        x = self.features1(x)#(12, 64, 56, 56)
        x = self.features2(x)#(12, 128, 28, 28)
        x = self.features3(x)#(12, 256, 14, 14)
        x = self.features4(x)#(12, 512, 7, 7)

        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]


# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


def get_model(args):
    '''Setup network'''
    ctx = [mxnet.gpu(gpu_id) for gpu_id in args.gpu]

    net = MYRESNET(cnn_arch='resnet18_v1', num_layers=18, cnn_feature_length=4096, num_views=args.num_views,
                   num_class=args.num_classes,
                   pretrained=args.pretrained, ctx=ctx, disable_sort=args.disable_sort)
    if args.checkpoint:
        net.load_parameters(args.checkpoint, ctx=ctx)
        # for _, w in net.features0.collect_params().items():
        #     w.grad_rep = 'null'
        # for _, w in net.features0.collect_params().items():
        #     print('feature0:',_,':',w)
        # sys.exit()
    else:
        net.features1.initialize(init=init.MSRAPrelu(), ctx=ctx)
        net.initialize(init=init.MSRAPrelu(), ctx=ctx)
    # net.hybridize()
    net.collect_params().setattr('grad_req', 'add')
    net.output.collect_params().setattr('lr_mult', args.output_lr_mult)

    return net


def get_resnet18_v1(version=1, num_layers=18, pretrained=False, ctx=cpu(),
                    root=os.path.join(base.data_dir(), 'models'), **kwargs):
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert version >= 1 and version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2." % version
    resnet_class = ResNetV1
    block_class = resnet_block_versions[version - 1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        # net.load_parameters('./viewConsistent/myparams/mymodel_params/myparams.params', ctx=ctx,ignore_extra=True)
        net.initialize(init.MSRAPrelu(), ctx=ctx)
    return net



class MYRESNET(nn.HybridBlock):
    def __init__(self, cnn_arch, num_layers, num_class, cnn_feature_length=4096, pretrained=False,
                 ctx=(0,), num_views=12, disable_sort=False, **kwargs):
        super(MYRESNET, self).__init__(**kwargs)
        self.arch_name = cnn_arch
        self.pretrained = pretrained
        self.num_views = num_views
        self.cnn_feature_length = cnn_feature_length
        self.disable_sort = disable_sort
        self.weights = self.item_dict = self.params.get('weights', shape=(self.num_views,), init=initializer.Uniform(),
                                                        lr_mult=10)
        


        with self.name_scope():
            cnnresnet= get_resnet18_v1(1, num_layers, pretrained=True, ctx=ctx)
            self.features00=cnnresnet.features0
            self.features01 = cnnresnet.features1
            self.features02 = cnnresnet.features2
            self.features03 = cnnresnet.features3
            self.features04 = cnnresnet.features4
            # x = self.features0(x)  # (12, 64, 112, 112)
            # x = self.features1(x)  # (12, 64, 56, 56)
            # x = self.features2(x)  # (12, 128, 28, 28)
            # x = self.features3(x)  # (12, 256, 14, 14)
            # x = self.features4(x)  # (12, 512, 7, 7)
            # cnnvgg=vision.vgg11_bn(pretrained=True,ctx=ctx)
            # self.features1=cnnvgg.features1
 
            #self.max_pooling=nn.MaxPool2D(4,3,0)

            self.features1 = nn.HybridSequential()
            self.features1.add(nn.Dense(self.cnn_feature_length, activation='relu',
                                        weight_initializer='normal',
                                        bias_initializer='zeros'))
            self.features1.add(nn.Dropout(rate=0.5))
            self.features1.add(nn.Dense(self.cnn_feature_length, activation='relu',
                                        weight_initializer='normal',
                                        bias_initializer='zeros'))
            self.features1.add(nn.Dropout(rate=0.5)) 


            self.output = nn.Dense(in_units=self.cnn_feature_length, units=num_class)


    def hybrid_forward(self, F, x,index_7,index_14,weights, *args, **kwargs):
        # print('forward:', 1, datetime.datetime.now())
        x = x.reshape((-1, 3, 224, 224))  # [batch_size * num_views, 3, 224, 224]
        x = self.features00(x)  # [batch_size * num_views, c]--->[batch_size * num_views, 512,7,7]
        x = self.features01(x)#(12, 64, 56, 56)
        x = self.features02(x)#(12, 128, 28, 28)

        """ x = x.reshape(-1, 12, 128, 28, 28)  # (B, 12, 512, 7, 7)
        index_28=nd.array(index_28,dtype=np.int)
        fusion_28=Fusion_28()
        x = fusion_28(x, index_28)
        x=nd.array(x)
        x = x.reshape((-1, 128, 28, 28)) """

        x = self.features03(x)#(12, 256, 14, 14)

        # x = x.reshape(-1,self.select_views, 256, 14, 14)  # (B, 12, 512, 7, 7)
        # index_14 = nd.array(index_14, dtype=np.int)
        # fusion_14 = Fusion_14()
        # x = fusion_14(x, index_14)
        # x = nd.array(x)
        # x = x.reshape((-1, 256, 14, 14)) 

        x = self.features04(x)#(12, 512, 7, 7)

        x1 = x.reshape(-1,self.num_views, 512, 7, 7)  # (B, 12, 512, 7, 7)
        x2=x
        index_7=nd.array(index_7,dtype=np.int)
        fusion_7=Fusion_7()
        x1 = fusion_7(x1, index_7)
        x1=nd.array(x1)
        x1 = x1.reshape((-1, 512, 7, 7))

        x=nd.concat(x1,x2,dim=1)# (B*12, 512*2, 7, 7)
        # x=self.max_pooling(x)# (B*12, 512*2, 2, 2)
        x = self.features1(x)  # [batch_size * num_views, 1024]
        x = x.reshape((-1, self.num_views, self.cnn_feature_length))  # [batch_size, num_views, c]

        if not self.disable_sort:
            x = F.sort(x, axis=1)
        weights = F.softmax(weights).reshape((1, self.num_views, 1))
        x = F.broadcast_mul(weights, x).sum(axis=1)
        # x = F.max(x, axis=1)  # [batch_size, c]
        x = self.output(x)

        # print('forward:', 4, datetime.datetime.now())
        return x  # [batch_size, num_class]

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False, force_reinit=False):
        self.output.initialize(init, ctx=ctx)
        if self.pretrained:
            a=1
            self.weights.initialize(initializer.Uniform(), ctx)
        else:
            super().initialize(init, ctx, verbose, force_reinit)

    def get_info(self):
        return 'Architecture: %s_mvcnn' % (self.arch_name)





class Fusion_7(autograd.Function):
    # @staticmethod
    def forward(self, pic, index):
        B, N, C, H, W = pic.shape
        pic = pic.asnumpy()
        index = index.asnumpy()
        out = pic*1.0
        k_num = 3  # 7->3/14->6/28->9
        n_num = np.ones((B, N, C, H, W))
        

        @jit(nopython=True)
        def sss0(B, N, H, W, C, index, pic, out, n_num,k_num):
            for batch in range(B):
                for view in range(N):
                    for x in range(H):
                        for y in range(W):
                            if index[batch, view, x, y, 5, 1] != 0:
                                for k in range(k_num):
                                    # if index[batch, view, x, y, k, 0] != view:
                                    v = index[batch, view, x, y, k, 0]
                                    i = index[batch, view, x, y, k, 1]
                                    j = index[batch, view, x, y, k, 2]
                                    for l in range(C):
                                        if out[batch, view, l, x, y]<pic[batch, v, l, i, j]:
                                            out[batch, view, l, x, y] = pic[batch, v, l, i, j]
                                            n_num[batch, view, l, x, y] = 0.0

            return out, n_num

        out, n_num = sss0(B, N, H, W, C, index, pic, out, n_num,k_num)
        n_num = nd.array(n_num)
        self.save_for_backward(n_num)
        out = nd.array(out)
        return out

    def backward(self, dy):
        n, = self.saved_tensors
        # n=n.reshape((-1, 512, 7, 7))
        # n=(2, 12, 512, 7, 7)
        # dy=(2, 12, 512, 7, 7)
        # return_grad = dy * n / 12.0
        return_grad = dy * n
        zero = nd.zeros(shape=(2, 12, 7, 7, 12, 3))
        return return_grad, zero



class Fusion_14(autograd.Function):
    # @staticmethod
    def forward(self, pic, index):
        B, N, C, H, W = pic.shape
        pic = pic.asnumpy()
        index = index.asnumpy()
        out = pic * 1.0
        k_num = 6  # 7->3/14->6/28->9
        n_num = np.ones((B, N, C, H, W))


        @jit(nopython=True)
        def sss0(B, N, H, W, C, index, pic, out, n_num, k_num):
            for batch in range(B):
                for view in range(N):
                    for x in range(H):
                        for y in range(W):
                            if index[batch, view, x, y, 5, 1] != 0:
                                for l in range(C):
                                    iiii = 1
                                    for k in range(1, k_num):
                                        # if index[batch, view, x, y, k, 0] != view:
                                        v = index[batch, view, x, y, k, 0]
                                        i = index[batch, view, x, y, k, 1]
                                        j = index[batch, view, x, y, k, 2]
                                        if pic[batch, v, l, i, j] != 0.0:
                                            out[batch, view, l, x, y] = pic[batch, v, l, i, j] + out[
                                                batch, view, l, x, y]
                                            iiii += 1
                                    out[batch, view, l, x, y] = out[batch, view, l, x, y] / iiii
                                    n_num[batch, view, l, x, y] = 1 / iiii

            return out, n_num

        out, n_num = sss0(B, N, H, W, C, index, pic, out, n_num, k_num)
        n_num = nd.array(n_num)
        self.save_for_backward(n_num)
        out = nd.array(out)
        return out

    def backward(self, dy):
        n, = self.saved_tensors
        # n=n.reshape((-1, 512, 7, 7))
        # n=(2, 12, 512, 7, 7)
        # dy=(2, 12, 512, 7, 7)
        # return_grad = dy * n / 12.0
        return_grad = dy * n
        zero = nd.zeros(shape=(2, 12, 7, 7, 12, 3))
        return return_grad, zero


