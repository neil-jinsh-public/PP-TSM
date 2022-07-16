import os
import sys
import cv2
import time
import math
import copy
import paddle
import random
import traceback
import itertools

import numpy as np
import os.path as osp
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as init

from PIL import Image
from tqdm import tqdm
from paddle import ParamAttr
from paddle.optimizer.lr import *
from collections import OrderedDict
from collections.abc import Sequence
from paddle.regularizer import L2Decay
from paddle.nn import (Conv2D, BatchNorm2D, Linear, Dropout, MaxPool2D,
                       AvgPool2D, AdaptiveAvgPool2D)


class ConvBNLayer(nn.Layer):
    """Conv2D and BatchNorm2D layer.
    Args:
        in_channels (int): Number of channels for the input.
        out_channels (int): Number of channels for the output.
        kernel_size (int): Kernel size.
        stride (int): Stride in the Conv2D layer. Default: 1.
        groups (int): Groups in the Conv2D, Default: 1.
        is_tweaks_mode (bool): switch for tweaks. Default: False.
        act (str): Indicate activation after BatchNorm2D layer.
        name (str): the name of an instance of ConvBNLayer.
    Note: weight and bias initialization include initialize values and name the restored parameters, values initialization are explicit declared in the ```init_weights``` method.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 is_tweaks_mode=False,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.is_tweaks_mode = is_tweaks_mode
        # ResNet-D 1/2:add a 2×2 average pooling layer with a stride of 2 before the convolution,
        #             whose stride is changed to 1, works well in practice.
        self._pool2d_avg = AvgPool2D(kernel_size=2,
                                     stride=2,
                                     padding=0,
                                     ceil_mode=True)

        self._conv = Conv2D(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size - 1) // 2,
                            groups=groups,
                            weight_attr=ParamAttr(name=name + "_weights"),
                            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        self._act = act

        self._batch_norm = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(name=bn_name + "_scale",
                                  regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(bn_name + "_offset", regularizer=L2Decay(0.0)))

    def forward(self, inputs):
        if self.is_tweaks_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self._act:
            y = getattr(paddle.nn.functional, self._act)(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 num_seg=8,
                 name=None):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 act="leaky_relu",
                                 name=name + "_branch2a")
        self.conv1 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 act="leaky_relu",
                                 name=name + "_branch2b")

        self.conv2 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels * 4,
                                 kernel_size=1,
                                 act=None,
                                 name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=
                1,  # ResNet-D 2/2:add a 2×2 average pooling layer with a stride of 2 before the convolution,
                #             whose stride is changed to 1, works well in practice.
                is_tweaks_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut
        self.num_seg = num_seg

    def forward(self, inputs):
        shifts = paddle.fluid.layers.temporal_shift(inputs, self.num_seg,
                                                    1.0 / self.num_seg)
        y = self.conv0(shifts)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        return F.leaky_relu(y)


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 filter_size=3,
                                 stride=stride,
                                 act="leaky_relu",
                                 name=name + "_branch2a")
        self.conv1 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 filter_size=3,
                                 act=None,
                                 name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=out_channels,
                                     filter_size=1,
                                     stride=stride,
                                     name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(short, conv1)
        y = F.leaky_relu(y)
        return y


class ResNetTweaksTSM(nn.Layer):
    """ResNet TSM backbone.
    Args:
        depth (int): Depth of resnet model.
        pretrained (str): pretrained model. Default: None.
    """

    def __init__(self, depth, num_seg=8, pretrained=None):
        super(ResNetTweaksTSM, self).__init__()
        self.pretrained = pretrained
        self.layers = depth
        self.num_seg = num_seg

        supported_layers = [18, 34, 50, 101, 152]
        assert self.layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, self.layers)

        if self.layers == 18:
            depth = [2, 2, 2, 2]
        elif self.layers == 34 or self.layers == 50:
            depth = [3, 4, 6, 3]
        elif self.layers == 101:
            depth = [3, 4, 23, 3]
        elif self.layers == 152:
            depth = [3, 8, 36, 3]

        in_channels = 64
        out_channels = [64, 128, 256, 512]

        # ResNet-C: use three 3x3 conv, replace, one 7x7 conv
        self.conv1_1 = ConvBNLayer(in_channels=3,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=2,
                                   act='leaky_relu',
                                   name="conv1_1")
        self.conv1_2 = ConvBNLayer(in_channels=32,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   act='leaky_relu',
                                   name="conv1_2")
        self.conv1_3 = ConvBNLayer(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   act='leaky_relu',
                                   name="conv1_3")
        self.pool2D_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        if self.layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if self.layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' %
                        (block, i),  # same with PaddleClas, for loading pretrain
                        BottleneckBlock(
                            in_channels=in_channels
                            if i == 0 else out_channels[block] * 4,
                            out_channels=out_channels[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            num_seg=self.num_seg,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    in_channels = out_channels[block] * 4
                    self.block_list.append(bottleneck_block)
                    shortcut = True
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        conv_name,
                        BasicBlock(in_channels=in_channels[block]
                        if i == 0 else out_channels[block],
                                   out_channels=out_channels[block],
                                   stride=2 if i == 0 and block != 0 else 1,
                                   shortcut=shortcut,
                                   name=conv_name))
                    self.block_list.append(basic_block)
                    shortcut = True

    def init_weights(self):
        """Initiate the parameters.
        Note:
            1. when indicate pretrained loading path, will load it to initiate backbone.
            2. when not indicating pretrained loading path, will follow specific initialization initiate backbone. Always, Conv2D layer will be initiated by KaimingNormal function, and BatchNorm2d will be initiated by Constant function.
            Please refer to https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/initializer/kaiming/KaimingNormal_en.html
        """
        # XXX: check bias!!! check pretrained!!!

        if isinstance(self.pretrained, str) and self.pretrained.strip() != "":
            load_ckpt(self, self.pretrained)
        elif self.pretrained is None or self.pretrained.strip() == "":
            for layer in self.sublayers():
                if isinstance(layer, nn.Conv2D):
                    # XXX: no bias
                    weight_init_(layer, 'KaimingNormal')
                elif isinstance(layer, nn.BatchNorm2D):
                    weight_init_(layer, 'Constant', value=1)

    def forward(self, inputs):
        """Define how the backbone is going to run.
        """
        # NOTE: Already merge axis 0(batches) and axis 1(channels) before extracting feature phase,
        # please refer to paddlevideo/modeling/framework/recognizers/recognizer2d.py#L27
        # y = paddle.reshape(
        #    inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])

        #### ResNet-C: use three 3x3 conv, replace, one 7x7 conv
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)

        y = self.pool2D_max(y)
        for block in self.block_list:
            y = block(y)
        return y


class CrossEntropyLoss(nn.Layer):
    """Cross Entropy Loss."""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def _forward(self, score, labels, **kwargs):
        """Forward function.
        Args:
            score (paddle.Tensor): The class score.
            labels (paddle.Tensor): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        Returns:
            loss (paddle.Tensor): The returned CrossEntropy loss.
        """
        loss = F.cross_entropy(score, labels, **kwargs)
        return loss

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.
        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.
        Returns:
            paddle.Tensor: The calculated loss.
        """
        return self._forward(*args, **kwargs) * self.loss_weight


class ppTSMHead(nn.Layer):
    """ ppTSM Head
    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channles in input feature.
        loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
        drop_ratio(float): drop ratio. Default: 0.8.
        std(float): Std(Scale) value in normal initilizar. Default: 0.001.
        kwargs (dict, optional): Any keyword argument to initialize.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 drop_ratio=0.8,
                 std=0.01,
                 data_format="NCHW",
                 ls_eps=0.):

        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.drop_ratio = drop_ratio
        self.ls_eps = ls_eps

        self.fc = Linear(self.in_channels,
                         self.num_classes,
                         weight_attr=ParamAttr(learning_rate=5.0,
                                               regularizer=L2Decay(1e-4)),
                         bias_attr=ParamAttr(learning_rate=10.0,
                                             regularizer=L2Decay(0.0)))
        self.stdv = std
        self.loss_func = CrossEntropyLoss()
        self.avgpool2d = AdaptiveAvgPool2D((1, 1), data_format=data_format)
        self.dropout = Dropout(p=self.drop_ratio)

    def init_weights(self):
        """Initiate the FC layer parameters"""
        weight_init_(self.fc, 'Normal', 'fc_0.w_0', 'fc_0.b_0', std=self.stdv)

    def forward(self, x, seg_num):
        """Define how the head is going to run.
        Args:
            x (paddle.Tensor): The input data.
            num_segs (int): Number of segments.
        Returns:
            score: (paddle.Tensor) The classification scores for input samples.
        """

        # XXX: check dropout location!
        # [N * num_segs, in_channels, 7, 7]
        x = self.avgpool2d(x)
        # [N * num_segs, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N * seg_num, in_channels, 1, 1]
        x = paddle.reshape(x, [-1, seg_num, x.shape[1]])
        # [N, seg_num, in_channels]
        x = paddle.mean(x, axis=1)
        # [N, in_channels]
        x = paddle.reshape(x, shape=[-1, self.in_channels])
        # [N, in_channels]
        score = self.fc(x)
        # [N, num_class]
        return score

    def loss(self, scores, labels, valid_mode=False, **kwargs):
        """Calculate the loss accroding to the model output ```scores```,
           and the target ```labels```.
        Args:
            scores (paddle.Tensor): The output of the model.
            labels (paddle.Tensor): The target output of the model.
        Returns:
            losses (dict): A dict containing field 'loss'(mandatory) and 'top1_acc', 'top5_acc'(optional).
        """
        if len(labels) == 1:  # commonly case
            labels = labels[0]
            losses = dict()
            if self.ls_eps != 0. and not valid_mode:  # label_smooth
                loss = self.label_smooth_loss(scores, labels, **kwargs)
            else:
                loss = self.loss_func(scores, labels, **kwargs)

            top1, top5 = self.get_acc(scores, labels, valid_mode)
            losses['top1'] = top1
            losses['top5'] = top5
            losses['loss'] = loss
            return losses
        elif len(labels) == 3:  # mix_up
            labels_a, labels_b, lam = labels
            lam = lam[0]  # get lam value
            losses = dict()

            if self.ls_eps != 0:
                loss_a = self.label_smooth_loss(scores, labels_a, **kwargs)
                loss_b = self.label_smooth_loss(scores, labels_b, **kwargs)
            else:
                loss_a = self.loss_func(scores, labels_a, **kwargs)
                loss_b = self.loss_func(scores, labels_b, **kwargs)
            loss = lam * loss_a + (1 - lam) * loss_b
            top1a, top5a = self.get_acc(scores, labels_a, valid_mode)
            top1b, top5b = self.get_acc(scores, labels_b, valid_mode)
            top1 = lam * top1a + (1 - lam) * top1b
            top5 = lam * top5a + (1 - lam) * top5b
            losses['top1'] = top1
            losses['top5'] = top5
            losses['loss'] = loss
            return losses
        else:
            raise NotImplemented

    def label_smooth_loss(self, scores, labels, **kwargs):
        labels = F.one_hot(labels, self.num_classes)
        labels = F.label_smooth(labels, epsilon=self.ls_eps)
        labels = paddle.squeeze(labels, axis=1)
        loss = self.loss_func(scores, labels, soft_label=True, **kwargs)
        return loss

    def get_acc(self, scores, labels, valid_mode):
        top1 = paddle.metric.accuracy(input=scores, label=labels, k=1)
        top5 = paddle.metric.accuracy(input=scores, label=labels, k=5)
        return top1, top5


# Recognizer2D
#
# 将主干网络和头部分封装。

class Recognizer2D(nn.Layer):
    """2D recognizer model framework."""

    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.backbone.init_weights()
        self.head = head
        self.head.init_weights()

    def forward_net(self, imgs):
        # NOTE: As the num_segs is an attribute of dataset phase, and didn't pass to build_head phase,
        # should obtain it from imgs(paddle.Tensor) now, then call self.head method.
        imgs = paddle.Tensor(imgs)
        num_segs = imgs.shape[1]  # imgs.shape=[N,T,C,H,W], for most commonly case
        imgs = paddle.reshape_(imgs, [-1] + list(imgs.shape[2:]))
        feature = self.backbone(imgs)
        cls_score = self.head(feature, num_segs)
        return cls_score

    def train_step(self, data_batch):
        """Define how the model is going to train_video, from input to output.
        """
        imgs = data_batch[0]
        labels = data_batch[1:]
        cls_score = self.forward_net(imgs)
        loss_metrics = self.head.loss(cls_score, labels)
        return loss_metrics

    def val_step(self, data_batch):
        imgs = data_batch[0]
        labels = data_batch[1:]
        cls_score = self.forward_net(imgs)
        loss_metrics = self.head.loss(cls_score, labels, valid_mode=True)
        return loss_metrics

    def test_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        # NOTE: (shipping) when testing, the net won't call head.loss,
        # we deal with the test processing in /paddlevideo/metrics
        imgs = data_batch[0]
        cls_score = self.forward_net(imgs)
        return cls_score

    def infer_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        # imgs = data_batch[0]
        cls_score = self.forward_net(data_batch)
        return cls_score


def load_ckpt(model, weight_path):
    """
    1. Load pre-trained model parameters
    2. Extract and convert from the pre-trained model to the parameters
    required by the existing model
    3. Load the converted parameters of the existing model
    """
    # model.set_state_dict(state_dict)

    if not osp.isfile(weight_path):
        raise IOError(f'{weight_path} is not a checkpoint file')
    # state_dicts = load(weight_path)

    state_dicts = paddle.load(weight_path)
    tmp = {}
    total_len = len(model.state_dict())
    with tqdm(total=total_len, position=1, bar_format='{desc}', desc="Loading weights") as desc:
        for item in tqdm(model.state_dict(), total=total_len, position=0):
            name = item
            desc.set_description('Loading %s' % name)
            if name not in state_dicts:  # Convert from non-parallel model
                if str('backbone.' + name) in state_dicts:
                    tmp[name] = state_dicts['backbone.' + name]
            else:  # Convert from parallel model
                tmp[name] = state_dicts[name]
            time.sleep(0.01)
    ret_str = "loading {:<20d} weights completed.".format(len(model.state_dict()))
    desc.set_description(ret_str)
    model.set_state_dict(tmp)


def weight_init_(layer,
                 func,
                 weight_name=None,
                 bias_name=None,
                 bias_value=0.0,
                 **kwargs):
    """
    In-place params init function.
    Usage:
    .. code-block:: python
        import paddle
        import numpy as np
        data = np.ones([3, 4], dtype='float32')
        linear = paddle.nn.Linear(4, 4)
        input = paddle.to_tensor(data)
        print(linear.weight)
        linear(input)
        weight_init_(linear, 'Normal', 'fc_w0', 'fc_b0', std=0.01, mean=0.1)
        print(linear.weight)
    """

    if hasattr(layer, 'weight') and layer.weight is not None:
        getattr(init, func)(**kwargs)(layer.weight)
        if weight_name is not None:
            # override weight name
            layer.weight.name = weight_name

    if hasattr(layer, 'bias') and layer.bias is not None:
        init.Constant(bias_value)(layer.bias)
        if bias_name is not None:
            # override bias name
            layer.bias.name = bias_name
