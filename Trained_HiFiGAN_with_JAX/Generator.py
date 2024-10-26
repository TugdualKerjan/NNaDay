#!/usr/bin/env python
# coding: utf-8



import jax
import equinox as eqx
import equinox.nn as nn
import ResBlocks1and2

LRELU_SLOPE = 0.1

class MRF(eqx.Module):
    resblocks: list

    def __init__(self, channel_in, kernel_sizes, dilations, key=None):
        self.resblocks = []

        for kernel_size in kernel_sizes:
            key, grab = jax.random.split(key, 2)
            self.resblocks.append(ResBlocks1and2.ResBlock1(channel_in, kernel_size, dilations, key=grab))
    
    def __call__(self, x):
        y = self.resblocks[0](x)
        for block in self.resblocks[1:]:
            y += block(y)

        return y / len(self.resblocks)

class Generator(eqx.Module):
    pre_magic: nn.Conv1d

    layers: list

    post_magic: nn.Conv1d

    def __init__(self, channels_in, channels_out, h_u=512, k_u = [16, 16, 4, 4], upsample_rate_decoder=[8,8,2,2], k_r = [3, 7, 11], dilations=[[1, 1], [3, 1], [5, 1]], key=None):

        key, grab = jax.random.split(key, 2)
        self.pre_magic = nn.Conv1d(channels_in, h_u, kernel_size=7, dilation=1, padding=3, key=grab)

        self.layers = []

        # This is where the magic happens. Upsample aggressively then more slowly. TODO could play around with this.
        # Then convolve one last time (Curious to see the weights to see if has good impact)
        for i, (k, u) in enumerate(zip(k_u, upsample_rate_decoder)):
            layer = []
            current_chans = int(h_u / (2 ** i))
            key, grab1, grab2 = jax.random.split(key, 3)

            # These upsample the mel by cutting channels by half but increasing by 16/2 = 8 with a transpose.

            padding = int((k-u)/2)


            layer.append(nn.ConvTranspose1d(current_chans, int(current_chans / 2), kernel_size=k, stride=u, padding=((padding - 2, padding + 2),), key=grab1))  # Ensure stride and padding are integers
            # layer.append(MRF(channel_in=int(current_chans / 2), kernel_sizes=k_r, dilations=dilations, key= grab2))
            self.layers.append(layer)
        
        self.post_magic = nn.Conv1d(int(current_chans / 2), channels_out, kernel_size=7, stride=1, padding=3, key=key)
        # self.post_magic = nn.WeightNorm(self.post_magic,

    def __call__(self, x):
        y = self.pre_magic(x)

        for layer in self.layers:
            y = jax.nn.leaky_relu(y, LRELU_SLOPE)
            y = layer[0](y) # Upsample
            # y = layer[1](y) # MRF
        y = jax.nn.leaky_relu(y, LRELU_SLOPE)

        y = self.post_magic(y)
        y = jax.nn.tanh(y)
        return y

    