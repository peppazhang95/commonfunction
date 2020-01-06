#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:34:47 2020

@author: peppazhang
"""
import tensorflow as tf
import Layers as ly
import modules as md

def Res2NetLite(input_, base_channel = 32, is_training = True, data_format = 'channels_last', reuse = False, c = 72, stage_number =[3,7,3], name = 'Res2NetLite'):
    """
    the backbone of the detection model.
    
    """
    with tf.variable_scope(name_or_scope = name, reuse = reuse):
        layers = []
        #stage-0: 224,224
        bn1 = ly._bn(input_, data_format, is_training)
        #112,112,32
        conv1 = ly.conv_bn_activation(bn1, 32, 3, 2)
        #56,56,32
        maxp1 = ly.max_pooling(conv1, 3, 2)
        
        #stage-1
        #28,28,4c
        bottle1_s1 = md.Bottleneck(32, 4*c, name ='Bottleneck_stage1')
        out_s1 = bottle1_s1.forward(maxp1)
        for i in range(stage_number[0]):
            _name = 'res2block_'+str(i)+'_stage1'
            layers.append(md.Res2Block(4*c, name = _name))
            out_s1 = layers[-1].forward(out_s1)
        
        #stage-2
        #14,14,8c
        bottle1_s2 = md.Bottleneck(4*c, 8*c, name ='Bottleneck_stage2')
        out_s2 = bottle1_s2.forward(out_s1)
        for i in range(stage_number[1]):
            _name = 'res2block_'+str(i)+'_stage2'
            layers.append(md.Res2Block(8*c, name = _name))
            out_s2 = layers[-1].forward(out_s2)
        
        #stage-3
        #7,7,16c
        bottle1_s3 = md.Bottleneck(8*c, 16*c, name ='Bottleneck_stage3')
        out_s3 = bottle1_s3.forward(out_s2)
        for i in range(stage_number[2]):
            _name = 'res2block_'+str(i)+'_stage3'
            layers.append(md.Res2Block(16*c, name = _name))
            out_s3 = layers[-1].forward(out_s3)
        
        #stage-4
        out = ly.avg_pooling(out_s3, 7, 7)
        print(out)
        out = ly.conv_bn_activation(out, 1000, 1, 1)
        print(conv1,maxp1,out_s1,out_s2,out_s3)
        return out

    
    