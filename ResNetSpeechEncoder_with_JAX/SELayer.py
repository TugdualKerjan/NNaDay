#!/usr/bin/env python
# coding: utf-8



import equinox as eqx
import jax


class SELayer(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, channel, key, reduction=8):
        key1, key2 = jax.random.split(key, 2)
        self.fc1 = eqx.nn.Linear(channel, channel // reduction, use_bias=True, key=key1)
        self.fc2 = eqx.nn.Linear(channel // reduction, channel, use_bias=True, key=key2)

    def __call__(self, x):
        y = eqx.nn.AdaptiveAvgPool2d(1)(x)
        y = jax.numpy.squeeze(y)
        y = self.fc1(y)
        y = jax.nn.relu(y)
        y = self.fc2(y)
        y = jax.nn.sigmoid(y)
        y = jax.numpy.expand_dims(y, (1, 2))
        
        return x * y
