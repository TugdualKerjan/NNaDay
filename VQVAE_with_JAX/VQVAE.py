#!/usr/bin/env python
# coding: utf-8



import jax
import equinox as eqx
import equinox.nn as nn

class ResBlock(eqx.Module):
    layers: list
    norm1: nn.BatchNorm
    norm2: nn.BatchNorm

    def __init__(self, dim, key):
        key1, key2 = jax.random.split(key, 2)

        self.layers = [
            nn.Conv2d(dim, dim, (5, 5), padding="SAME", key=key1),
            nn.Conv2d(dim, dim, (5, 5), padding="SAME", key=key2)
        ]
        self.norm1 = nn.BatchNorm(input_size=dim, axis_name="batch2", momentum=0.9, dtype=jax.numpy.float32)
        self.norm2 = nn.BatchNorm(input_size=dim, axis_name="batch2", momentum=0.9, dtype=jax.numpy.float32)

    def __call__(self, x, state):
        y = x

        y = self.layers[0](y)
        y, state = self.norm1(y, state)

        y = jax.nn.relu(y)

        y = self.layers[1](y)
        y, state = self.norm2(y, state)

        y = y + x
        y = jax.nn.relu(y)

        return y, state


class Encoder(eqx.Module):
    layers: list

    def __init__(self, dim, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.layers = [
            nn.Conv2d(1, 2, (4, 4), 2, padding="SAME", key=key1),
            nn.Conv2d(2, 4, (4, 4), 2, padding="SAME", key=key3),
            ResBlock(dim, key=key2),
            ResBlock(dim, key=key4)
        ]

    def __call__(self, x, state):
        y = x
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                y, state = layer(y, state)
            else:
                y = layer(y) 
    
        return y, state




class Decoder(eqx.Module):
    layers: list

    def __init__(self, dim, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.layers = [
            ResBlock(dim, key=key2),
            ResBlock(dim, key=key4),
            nn.ConvTranspose2d(4, 2, (4, 4), 2, padding="SAME", key=key1),
            nn.ConvTranspose2d(2, 1, (4, 4), 2, padding="SAME", key=key3),
        ]

    def __call__(self, x, state):
        y = x
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                y, state = layer(y, state)
            else:
                y = layer(y)

        y = jax.nn.sigmoid(y)
    
        return y, state




import jax
import equinox as eqx
import jax.numpy as np

class Quantizer(eqx.Module):
    K: int
    D: int
    codebook: np.array

    def __init__(self, num_vecs, num_dims, key):
        self.K = num_vecs
        self.D = num_dims

        # Init a matrix of vectors that will move with time

        self.codebook = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_in", distribution="uniform")(key, (self.K, self.D))

    def __call__(self, x):
        # X comes in as a N x D Matrix.
        flattened_x = jax.numpy.reshape(x, (-1, self.D))
        # Calculate dist
        # Nx1
        a_squared = np.sum(flattened_x**2, axis=-1, keepdims=True)
        # 1xK
        b_squared = np.transpose(np.sum(self.codebook**2, axis=-1, keepdims=True))
        distance = a_squared + b_squared - 2*np.matmul(flattened_x, np.transpose(self.codebook))


        encoding_indices = np.reshape(
            np.argmin(distance, axis=-1), x.shape[0]
        )


        z_q = self.codebook[encoding_indices]


        z_q = flattened_x + jax.lax.stop_gradient(z_q - flattened_x) # For the straight through estimation.
        z_q = jax.numpy.reshape(z_q, (4, 7, 7))

        return z_q

