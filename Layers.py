#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:16:51 2020

@author: peppazhang
"""

import tensorflow as tf


def _bn(bottom, data_format='channels_last', is_training=True):
    """
     Batch Normalization.
     Args:
         bottom: the input of data.
         data_format: the localization of channels of data; channel_last: [H,W,channels]
         is_training: the mode of model, train or test
     Returns:
         bn:the result of batch norm
    """
    bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if data_format == 'channels_last' else 1,
            training= is_training
        )
    return bn

def conv_bn_activation(bottom, filters, kernel_size, strides, data_format = 'channels_last', is_training = True, pad = 'same', activation=tf.nn.relu, _use_bias = True, kernel_initializer= None):
    """
        The common conv layer with bn and activation
        Args:
            bottom: the input of layer
            filters: the number of kernels. like 32
            kernel_size: the size of kernel. like kernel_size = 7
            strides: the stride of kernel. like strides = 1
            activation: the method of activation. like tf.nn.relu
            data_format: the localization of channels of data; channel_last: [H,W,channels]
            is_training: the mode of model, train or test
        Returns:
            outp: the output of conv layer
    """
    if kernel_initializer == None:
        conv = tf.layers.conv2d(
                inputs=bottom,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding= pad,
                data_format=data_format,
                use_bias = _use_bias
            )
    else:
        conv = tf.layers.conv2d(
                inputs=bottom,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding= pad,
                data_format=data_format,
                use_bias = _use_bias,
                kernel_initializer = kernel_initializer
            )
    bn = _bn(conv, data_format, is_training)
    if activation is not None:
        outp = activation(bn)
        return outp
    else:
        outp = bn
        return outp

def dconv_bn_activation(bottom, filters, kernel_size, strides, data_format = 'channels_last', is_training = True, activation=tf.nn.relu):
    """
        The common deconv layer with bn and activation
        Args:
            bottom: the input of layer
            filters: the number of kernels. like 32
            kernel_size: the size of kernel. like kernel_size = 7
            strides: the stride of kernel. like strides = 1
            activation: the method of activation. like tf.nn.relu
            data_format: the localization of channels of data; channel_last: [H,W,channels]
            is_training: the mode of model, train or test
        Returns:
            outp: the output of deconv layer
    """
    conv = tf.layers.conv2d_transpose(
        inputs=bottom,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        data_format=data_format,
    )
    bn = _bn(conv, data_format, is_training)
    if activation is not None:
        outp = activation(bn)
    else:
        outp = bn
    return outp

def depthwise_conv_layer(bottom, channels, kernel_size, _strides, data_format = 'channels_last', pad = 'SAME'):
    if data_format == 'channels_last':
        data_format = "NHWC"
    else:
        data_format = "NCHW"
    filters = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, channels, 1], stddev=0.1, dtype=tf.float32))
    outp = tf.nn.depthwise_conv2d(bottom, filter = filters, strides = _strides, rate = [1,1], padding = pad, data_format = data_format)
    
    return outp

def group_conv_layer(bottom, channels_in, channels_out, group_num, kernel_size, _strides, data_format = 'channels_last', pad = 'SAME', use_bias = True):
    
    data_list = tf.split(bottom, group_num, axis=-1)
    outp_list = [tf.layers.conv2d(
            inputs=data_list[i],
            filters=channels_out//group_num,
            kernel_size=kernel_size,
            strides=_strides,
            padding= pad,
            data_format=data_format,
            use_bias = use_bias
        ) for i in range(group_num)]
    
    outp = tf.concat(outp_list, axis = -1)
    return outp
    
def separable_conv_layer(bottom, filters, kernel_size, strides, data_format, is_training, activation=tf.nn.relu):
    """
        The common separable layer with bn and activation
        Args:
            bottom: the input of layer
            filters: the number of kernels. like 32
            kernel_size: the size of kernel. like kernel_size = 7
            strides: the stride of kernel. like strides = 1
            activation: the method of activation. like tf.nn.relu
            data_format: the localization of channels of data; channel_last: [H,W,channels]
            is_training: the mode of model, train or test
        Returns:
            outp: the output of deconv layer
    """
    conv = tf.layers.separable_conv2d(
        inputs=bottom,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        data_format=data_format,
        use_bias=False,
    )
    bn = _bn(conv, data_format, is_training)
    if activation is not None:
        outp = activation(bn)
    else:
        outp = bn
    return outp

    

def max_pooling(bottom, pool_size, strides, data_format ='channels_last', name=None):
    """
     max pooling layer.
     
     Args:
         bottom:the input
         pool_size: the size of the kernel
         strides: the size of the kernel
         name: the name of pooling
         data_format: the localization of channels of data; channel_last: [H,W,channels]
     Returns:
         the outp of maxPooling
    """
    return tf.layers.max_pooling2d(
        inputs=bottom,
        pool_size=pool_size,
        strides=strides,
        padding='same',
        data_format=data_format,
        name=name
    )

def avg_pooling(bottom, pool_size, strides, data_format='channels_last', name=None):
    """
     avg pooling layer.
     
     Args:
         bottom:the input
         pool_size: the size of the kernel
         strides: the size of the kernel
         name: the name of pooling
         data_format: the localization of channels of data; channel_last: [H,W,channels]
     Returns:
         the outp of avgPooling
    """
    return tf.layers.average_pooling2d(
        inputs=bottom,
        pool_size=pool_size,
        strides=strides,
        padding='same',
        data_format=data_format,
        name=name
    )

def dropout(bottom, name, prob, is_training):
    """
     the dropout layer.
     Args:
         bottom: the input of layer
         name: the name of layer
         prob: the prob of dropout
     Returns:
         dropout result
    """
    return tf.layers.dropout(
        inputs=bottom,
        rate=prob,
        training=is_training,
        name=name
    )

def channel_shuffle(x):
    """
    shuffle the channel of x with group = 2 proposed in shufflenetv2
    
    Args:
        x : the input of shuffle layer. x = [n, h,w, c]
    Returns:
        x_proj: the input which will be fed into the branch_proj
        x: the input which will be fed into the branch_main
    """
    g = 2
    n,h,w,c = x.shape
    x = tf.reshape(x, [n,h,w,g,c//g])
    x = tf.transpose(x, [0,1,2,4,3])
    x = tf.reshape(x,[n,h,w,-1])
    x_proj = x[:,:,:,:c//g]
    x = x[:,:,:,c//g:]
    return x_proj,x

def _basic_block(bottom, filters, data_format = 'channels_last', is_training = True, pad = 'same', activation=tf.nn.relu, _use_bias = True, kernel_initializer= None):
    conv = conv_bn_activation(bottom, filters, 3, 1, is_training = is_training)
    conv = conv_bn_activation(conv, filters, 3, 1, is_training = is_training)
    axis = 3 if data_format == 'channels_last' else 1
    input_channels = tf.shape(bottom)[axis]
    shutcut = tf.cond(
        tf.equal(input_channels, filters),
        lambda: bottom,
        lambda: conv_bn_activation(bottom, filters, 1, 1)
    )
    return conv + shutcut

def Res2BlockLayer(bottom, inp, scales=4, reuse = False, name = 'Res2Block'):
    scale_channel = inp//scales
    #with tf.variable_scope(name_or_scope = name, reuse = reuse):
    #dim reduce --> computation reduce
    pw1 = conv_bn_activation(bottom, scale_channel, 1, 1)
    #group process
    x1,x2,x3,x4 = tf.split(pw1, 4, -1)
    #branch-1
    group1 = depthwise_conv_layer(x2, scale_channel//4, 3, [1, 1, 1, 1])
    #branch-2
    inp_2 = group1 + x3
    group2 = depthwise_conv_layer(inp_2, scale_channel//4, 3, [1, 1, 1, 1])
    #branch-3
    inp_3 = group2 + x4
    group3 = depthwise_conv_layer(inp_3, scale_channel//4, 3, [1, 1, 1, 1])
    outp = tf.concat([x1,group1,group2,group3], axis = -1)
    # pw2
    outp = conv_bn_activation(outp, inp, 1, 1)
    outp = outp + bottom
        
    return outp

def bottlenecklayer(bottom,outp,is_training = True, reuse = False, data_format = 'channels_last'):
    pw1 = conv_bn_activation(bottom, outp//4, 1, 1)
    dw1 = depthwise_conv_layer(pw1, outp//4, 3, [1, 2, 2, 1], data_format = data_format)
    dw1 = _bn(dw1, data_format, is_training)
    out = conv_bn_activation(dw1, outp, 1, 1, data_format = data_format, is_training = is_training, _use_bias = False)
    return out

def _dla_generator(bottom, filters, levels, stack_block_fn, data_format = 'channels_last', is_training = True, pad = 'same', activation=tf.nn.relu, _use_bias = True, kernel_initializer= None):
    if levels == 1:
        block1 = stack_block_fn(bottom, filters)
        block2 = stack_block_fn(block1, filters)
        aggregation = block1 + block2
        aggregation = conv_bn_activation(aggregation, filters, 3, 1)
    else:
        block1 = _dla_generator(bottom, filters, levels-1, stack_block_fn)
        block2 = _dla_generator(block1, filters, levels-1, stack_block_fn)
        aggregation = block1 + block2
        aggregation = conv_bn_activation(aggregation, filters, 3, 1)
    return aggregation

def carafe_layer(input_tensor, kup=3, scale=2):
    """
    one kind of upsampling. It combine the advantages of both context-aware and loc-aware method.
    """
    #position-aware kernel
    N,H,W,C = input_tensor.shape
    context_encoder = conv_bn_activation(input_tensor, int((kup**2)*(scale**2)), 1, 1)
    context_encoder = tf.reshape(context_encoder, shape = [N,int(scale*H),int(scale*W),int(kup**2)])
    context_encoder = tf.nn.softmax(context_encoder, axis=-1)
    context_encoder = tf.reshape(context_encoder, shape = [N,H,W,int(kup**2),int(scale**2)])
    
    #extract patch from input tensor
    input_tensor = tf.pad(input_tensor, paddings=[[0,0],[int(kup//2),int(kup//2)],[int(kup//2),int(kup//2)],[0,0]],mode="CONSTANT")
    input_tensor_patch = tf.image.extract_image_patches(input_tensor, [1,kup,kup,1], [1,1,1,1], [1,1,1,1], 'VALID')
    input_tensor_patch = tf.reshape(input_tensor_patch,[N,H,W,C,int(kup**2)])
    
    #context-aware 
    out_tensor = tf.matmul(input_tensor_patch, context_encoder)
    out_tensor = tf.reshape(out_tensor, shape = [N,int(scale*H),int(scale*W),C])
    
    
    return out_tensor
    
    
def _gaussian_radius(height, width, min_overlap=0.7):
    """
    proposed in cornetNet to compute the radius of gaussian related to the size of box.    
    when we compute the soft label of gtbox center, we use the gaussian kernel and the hyperparameter about the radius related to the size of box.
    
    """
    a1 = 1.
    b1 = (height + width)
    c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
    sq1 = tf.sqrt(b1 ** 2. - 4. * a1 * c1)
    r1 = (b1 + sq1) / 2.
    a2 = 4.
    b2 = 2. * (height + width)
    c2 = (1. - min_overlap) * width * height
    sq2 = tf.sqrt(b2 ** 2. - 4. * a2 * c2)
    r2 = (b2 + sq2) / 2.
    a3 = 4. * min_overlap
    b3 = -2. * min_overlap * (height + width)
    c3 = (min_overlap - 1.) * width * height
    sq3 = tf.sqrt(b3 ** 2. - 4. * a3 * c3)
    r3 = (b3 + sq3) / 2.
    return tf.reduce_min([r1, r2, r3])    


"""
def save_weight(self, mode, path):
        assert (mode in ['latest', 'best'])
        if mode == 'latest':
            saver = self.saver
        else:
            saver = self.best_saver
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')
        saver.save(self.sess, path, global_step=self.global_step)
        print('save', mode, 'model in', path, 'successfully')

    def load_weight(self, path):
        self.saver.restore(self.sess, path)
        print('load weight', path, 'successfully')

    def load_pretrained_weight(self, path):
        self.pretrained_saver.restore(self.sess, path)
        print('load pretrained weight', path, 'successfully')

"""