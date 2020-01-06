#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 00:04:18 2020

@author: peppazhang
"""

import tensorflow as tf
import Layers as ly


class ShuffleV2Block:
    """
    ShuffleV2Block. the data will be split into two path. If stride == 1: we just use branch_main to deal with data_path. 
    Else: we will use branch_main and branch_proj to deal with data_path and data_proj respectively
    """
    def __init__(self, inp, oup, mid_channels, ksize, stride, name = 'ShuffleV2Block'):
        """
        param initialization and predefinition.
        
        Args:
            inp: the dim of the input
            oup: the dim of the output
            mid_channels: the dim of the mid_result
            ksize: the size of kernel used in depth-wise
            stride: the stride of the kernel used in depth-wise
            
        """
        self.strides = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.name = name
        self.outputs = oup - inp//2
    
    def _build_graph(self):
        """
        build the computation graph.
        
        """
        with tf.variable_scope(name_or_scope = self.name, reuse = self.reuse):
            if self.strides == 1:
                data_proj, data_path = ly.channel_shuffle(self.input)
                # deal data_path with branch_main
                with tf.variable_scope(name_or_scope = 'branch_main_s1', reuse = self.reuse):
                    data_path = ly.conv_bn_activation(data_path, self.mid_channels, 1, self.strides, data_format = self.data_format, is_training = self.is_training, _use_bias = False)
                    data_path = ly.depthwise_conv_layer(data_path, self.mid_channels, self.ksize, [1, self.strides, self.strides, 1], data_format = self.data_format)
                    data_path = ly._bn(data_path, self.data_format, self.is_training)
                    data_path = ly.conv_bn_activation(data_path, self.outputs, 1, self.strides, data_format = self.data_format, is_training = self.is_training, _use_bias = False)
                return tf.concat((data_proj, data_path), axis = -1)
            else:
                data_proj = self.input
                data_path = self.input
                with tf.variable_scope(name_or_scope = 'branch_main_s2', reuse = self.reuse):
                    data_path = ly.conv_bn_activation(data_path, self.mid_channels, 1, 1, data_format = self.data_format, is_training = self.is_training, _use_bias = False)
                    data_path = ly.depthwise_conv_layer(data_path, self.mid_channels, self.ksize, [1, self.strides, self.strides, 1], data_format = self.data_format)
                    data_path = ly._bn(data_path, self.data_format, self.is_training)
                    data_path = ly.conv_bn_activation(data_path, self.outputs, 1, 1, data_format = self.data_format, is_training = self.is_training, _use_bias = False)
                with tf.variable_scope(name_or_scope = 'branch_proj_s2', reuse = self.reuse):
                    data_proj = ly.depthwise_conv_layer(data_proj, self.inp, self.ksize, [1, self.strides, self.strides, 1], data_format = self.data_format)
                    data_proj = ly._bn(data_proj, self.data_format, self.is_training)
                    data_proj = ly.conv_bn_activation(data_proj, self.inp, 1, 1, data_format = self.data_format, is_training = self.is_training, _use_bias = False)
                return tf.concat((data_proj, data_path), axis = -1)
    def forward(self, input_, is_training = True, reuse = False, data_format = 'channels_last'):
        """
        compute the output of the shufflev2block.
        Args:
            input_:the input of block.
            is_training: True or False
            reuse: True or False
            data_format: channels_last or channels_first.
        
        """
        self.input = input_
        self.reuse = reuse
        self.is_training = is_training
        self.data_format = data_format
        outp = self._build_graph()
        return outp

class Bottleneck:
    def __init__(self,inp,oup,name = 'Bottleneck'):
        
        self.inp = inp
        self.outp = oup
        self.name = name
    
    def _build_graph(self):
        
        with tf.variable_scope(name_or_scope = self.name, reuse = self.reuse):
            pw1 = ly.conv_bn_activation(self.input, self.outp//4, 1, 1, data_format = self.data_format, is_training = self.is_training, _use_bias = False)
            dw1 = ly.depthwise_conv_layer(pw1, self.outp//4, 3, [1, 2, 2, 1], data_format = self.data_format)
            dw1 = ly._bn(dw1, self.data_format, self.is_training)
            out = ly.conv_bn_activation(dw1, self.outp, 1, 1, data_format = self.data_format, is_training = self.is_training, _use_bias = False)
        
        return out

    def forward(self, input_,is_training = True, reuse = False, data_format = 'channels_last'):
        self.input = input_
        self.reuse = reuse
        self.is_training = is_training
        self.data_format = data_format
        out = self._build_graph()
        return out



class Res2Block:
    def __init__(self,inp,scales=4, name = 'Res2Block'):
        self.name = name
        self.inp = inp
        self.scale_channel = inp // scales
        self.output_channels = inp
        
    def _build_graph(self):
        with tf.variable_scope(name_or_scope = self.name, reuse = self.reuse):
            pw1 = ly.conv_bn_activation(self.input, self.scale_channel, 1, 1, data_format = self.data_format, is_training = self.is_training, _use_bias = False)
            x1,x2,x3,x4 = tf.split(pw1, 4, -1)
            #branch-1
            group1 = ly.depthwise_conv_layer(x2, self.scale_channel//4, 3, [1, 1, 1, 1], data_format = self.data_format)
            #branch-2
            inp_2 = group1 + x3
            group2 = ly.depthwise_conv_layer(inp_2, self.scale_channel//4, 3, [1, 1, 1, 1], data_format = self.data_format)
            #branch-3
            inp_3 = group2 + x4
            group3 = ly.depthwise_conv_layer(inp_3, self.scale_channel//4, 3, [1, 1, 1, 1], data_format = self.data_format)
            outp = tf.concat([x1,group1,group2,group3], axis = -1)
            # pw2
            outp = ly.conv_bn_activation(outp, self.inp, 1, 1, data_format = self.data_format, is_training = self.is_training, _use_bias = False)
            outp = outp + self.input
        
        return outp

    def forward(self, input_,is_training = True, reuse = False, data_format = 'channels_last'):
        self.input = input_
        self.reuse = reuse
        self.is_training = is_training
        self.data_format = data_format
        out = self._build_graph()
        
        return out


"""   
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)        

"""