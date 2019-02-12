"""
@file    Network.py
@author  rohithjayarajan
@date 2/11/2019

Template Credits: Nitin J Sanket and Chahatdeep Singh
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True


def ResNetBlockType1(Input, FilterSize, i):

    with tf.variable_scope("ResNetBlock"+str(i)):

        BlockConv1 = tf.layers.conv2d(
            inputs=Input,
            filters=FilterSize,
            kernel_size=[3, 3],
            padding="same")
        BlockBN1 = tf.contrib.layers.batch_norm(BlockConv1,
                                                center=True, scale=True,
                                                is_training=True)
        BlockZ1 = tf.nn.relu(BlockBN1, name='ReLUT11')

        BlockConv2 = tf.layers.conv2d(
            inputs=BlockZ1,
            filters=FilterSize,
            kernel_size=[3, 3],
            padding="same")
        BlockBN2 = tf.contrib.layers.batch_norm(BlockConv2,
                                                center=True, scale=True,
                                                is_training=True)

        BlockBN2 = tf.add(BlockBN2, Input, name="add1")
        BlockZ2 = tf.nn.relu(BlockBN2, name='ReLUT12')

        return BlockZ2


def ResNetBlockType2(Input, FilterSize, i):

    with tf.variable_scope("ResNetBlock"+str(i)):

        BlockConv1 = tf.layers.conv2d(
            inputs=Input,
            filters=FilterSize,
            kernel_size=[3, 3],
            strides=2,
            padding="same")
        BlockBN1 = tf.contrib.layers.batch_norm(BlockConv1,
                                                center=True, scale=True,
                                                is_training=True)
        BlockZ1 = tf.nn.relu(BlockBN1, name='ReLUT21')

        BlockConv2 = tf.layers.conv2d(
            inputs=BlockZ1,
            filters=FilterSize,
            kernel_size=[3, 3],
            padding="same")
        BlockBN2 = tf.contrib.layers.batch_norm(BlockConv2,
                                                center=True, scale=True,
                                                is_training=True)

        shape = tf.shape(Input)
        Input = tf.layers.max_pooling2d(
            Input,
            pool_size=[1, 1],
            strides=2,
            padding='valid',
            name='pool')

        ChanelsToAdd = FilterSize - shape[3]
        Input = tf.pad(Input, [[0, 0], [0, 0], [0, 0], [0, ChanelsToAdd]])

        BlockBN2 = tf.add(BlockBN2, Input, name="add2")
        BlockZ2 = tf.nn.relu(BlockBN2, name='ReLUT22')

        return BlockZ2


def ResNet34Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    conv1 = tf.layers.conv2d(
        inputs=Img,
        filters=64,
        kernel_size=[7, 7],
        padding="same",
        name='conv1')
    bn1 = tf.contrib.layers.batch_norm(conv1,
                                       center=True, scale=True,
                                       is_training=True,
                                       scope='bn1')
    z1 = tf.nn.relu(bn1, name='ReLU1')

    # pool1 = tf.layers.max_pooling2d(
    #     z1,
    #     pool_size=[3, 3],
    #     strides=2,
    #     padding='valid',
    #     name='pool1')

    Rz1 = ResNetBlockType1(z1, 64, 1)
    Rz2 = ResNetBlockType1(Rz1, 64, 2)
    Rz3 = ResNetBlockType1(Rz2, 64, 3)

    Rz4 = ResNetBlockType2(Rz3, 128, 4)
    Rz5 = ResNetBlockType1(Rz4, 128, 5)
    Rz6 = ResNetBlockType1(Rz5, 128, 6)
    Rz7 = ResNetBlockType1(Rz6, 128, 7)

    Rz8 = ResNetBlockType2(Rz7, 256, 8)
    Rz9 = ResNetBlockType1(Rz8, 256, 9)
    Rz10 = ResNetBlockType1(Rz9, 256, 10)
    Rz11 = ResNetBlockType1(Rz10, 256, 11)
    Rz12 = ResNetBlockType1(Rz11, 256, 12)
    Rz13 = ResNetBlockType1(Rz12, 256, 13)

    Rz14 = ResNetBlockType2(Rz13, 512, 14)
    Rz15 = ResNetBlockType1(Rz14, 512, 15)
    Rz16 = ResNetBlockType1(Rz15, 512, 16)

    # pool5 = tf.layers.average_pooling2d(
    #     Rz14,
    #     pool_size=[2, 2],
    #     strides=2,
    #     padding='valid',
    #     name='pool5')

    Rz17_flat = tf.contrib.layers.flatten(Rz16)
    # dense1 = tf.layers.dense(inputs=Rz17_flat, units=1000, activation=None)
    # bn9 = tf.contrib.layers.batch_norm(dense1,
    #                                    center=True, scale=True,
    #                                    is_training=True,
    #                                    scope='dense1')

    # Rz18 = tf.nn.relu(bn9, name='ReLUend')
    prLogits = tf.layers.dense(inputs=Rz17_flat, units=10)
    prSoftMax = tf.nn.softmax(prLogits, name="softmax_tensor")

    return prLogits, prSoftMax
