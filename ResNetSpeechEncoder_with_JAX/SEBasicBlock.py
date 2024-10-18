#!/usr/bin/env python
# coding: utf-8



import jax
import equinox as eqx
from SELayer import SELayer

# https://arxiv.org/pdf/1512.03385

class SEBasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm
    se: SELayer
    downsample: None

    def __init__(self, channels_in, channels_out, stride=1, downsample=None, key=None):
        key1, key3, key5 = jax.random.split(key, 3)

        # TODO Understand why bias isn't added.
        # TODO Do we want to have a state or simply do GroupNorm instead ?

        self.conv1 = eqx.nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=stride, padding=1, use_bias=False, key=key1)
        self.bn1 = eqx.nn.BatchNorm(channels_out, axis_name="batch")
        self.conv2 = eqx.nn.Conv2d(channels_out, channels_out, kernel_size=(3, 3), padding=1, use_bias=False, key=key3)
        self.bn2 = eqx.nn.BatchNorm(channels_out, axis_name="batch")

        self.se = SELayer(channels_out, key=key5)
        self.downsample = downsample

    def __call__(self, x, state):
        residual = x

        y = self.conv1(x)
        
        y = jax.nn.relu(y)
        y, state = self.bn1(y,  state)

        y = self.conv2(y)
        y, state = self.bn2(y,  state)

        y = self.se(y)

        if self.downsample is not None:
            residual, state = self.downsample(x, state)

        y = y + residual #Residual
        y = jax.nn.relu(y)

        return y, state
