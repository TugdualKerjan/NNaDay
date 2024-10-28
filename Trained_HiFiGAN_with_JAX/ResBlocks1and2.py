#!/usr/bin/env python
# coding: utf-8



import jax
import equinox as eqx
import equinox.nn as nn

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

LRELU_SLOPE = 0.1

class ResBlock1(eqx.Module):
    layers: list

    def __init__(self, chan_in, kernel_size, dilations=[[1, 1], [3, 1], [5, 1]], key=None):
        self.layers = []

        for dilation in dilations:
            smol_layers = []
            for i in range(len(dilation)):
                key, grab = jax.random.split(key, 2)
                smol_layers.append(
                    nn.Conv1d(chan_in, chan_in, 
                            kernel_size=kernel_size, stride=1, 
                            dilation=dilation[i], padding=get_padding(kernel_size, dilation[i]), key=grab)
                )
            self.layers.append(smol_layers)
    
    def __call__(self, x):
        for smol_layer in self.layers:
            y = x
            for conv in smol_layer:
                y = jax.nn.leaky_relu(y, LRELU_SLOPE)
                y = conv(y)
            x = x + y

        return x