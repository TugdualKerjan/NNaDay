#!/usr/bin/env python
# coding: utf-8



import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as np

class SwiGLU(eqx.Module):
    W: nn.Linear
    V: nn.Linear
    b: jax.Array
    c: jax.Array

    def __init__(self, input_dim, output_dim, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.W = nn.Linear(input_dim, output_dim, key=key1)
        self.V = nn.Linear(input_dim, output_dim, key=key2)
        self.b = jax.random.normal(key3, (output_dim))
        self.c = jax.random.normal(key4, (output_dim))
        

    def __call__(self, x):
        return jax.nn.swish((self.W(x) + self.b) * (self.V(x) + self.c))




from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 3
    n_head: int = 3
    n_embd: int = 200
    dropout: float = 0.0
    bias: bool = False  #




class MLP(eqx.Module):
    layers: list


    def __init__(self, config, key):

        key1, key2, key3 = jax.random.split(key, 3)

        self.layers = [
            nn.Linear(config.n_embd, 4 * config.n_embd, use_bias=config.bias, key=key1),
            SwiGLU( 4 * config.n_embd,  4 * config.n_embd, key=key2),
            nn.Linear(4 * config.n_embd, config.n_embd, use_bias=config.bias, key=key3),
            nn.Dropout(config.dropout)
        ]
# TODO: Interesting take on the fact that vmap should be applied here ?
    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)

        return y




import math

class CausalSelfAttention(eqx.Module):
    attnk: nn.Linear
    attnq: nn.Linear
    attnv: nn.Linear
    proj: nn.Linear
    
    resid_dropout: nn.Dropout
    attn_dropout: nn.Dropout

    mask: jax.Array = eqx.field(static=True)
    
    def __init__(self, config, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        self.attnk = nn.Linear(config.n_embd,  config.n_embd, use_bias=config.bias, key=key1)
        self.attnv = nn.Linear(config.n_embd, config.n_embd, use_bias=config.bias, key=key2)
        self.attnq = nn.Linear(config.n_embd,  config.n_embd, use_bias=config.bias, key=key3)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.proj = nn.Linear(config.n_embd,  config.n_embd, use_bias=config.bias, key=key4)

        self.mask = np.tril(np.ones((config.block_size, config.block_size)))

    # Could play arround with the different attention score calculations (Baidhu ?)
    # X is an embedding, it should self attend.  
    
    def __call__(self, x):
        # x = np.swapaxes(x, -1, -2)
        T, C = x.shape # Seq length and embedding dim.
        

        q = jax.vmap(self.attnq)(x)
        k = jax.vmap(self.attnk)(x)
        v = jax.vmap(self.attnv)(x)
        
        att = np.matmul(q, np.transpose(k)) / math.sqrt(np.shape(k)[-1])
        att = np.where(jax.numpy.equal(jax.lax.stop_gradient(self.mask[:T, :T]), 0), float('-inf'), att)
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att)

        y = np.matmul(att, v)

        y = jax.vmap(self.proj)(y)
        y = self.resid_dropout(y)
        return y




class Block(eqx.Module):
    norm: nn.LayerNorm
    attn: CausalSelfAttention
    mlp: MLP

    def __init__(self, config, key):
        key1, key2 = jax.random.split(key, 2)
        
        self.norm = nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = CausalSelfAttention(config, key=key1)
        self.mlp = MLP(config, key=key2)
        

    def __call__(self, x):
        y = jax.vmap(self.norm)(x)
        y = self.attn(y) # Can't vmap as the whole point is exchange info between tokens.
        x = y + x

        y = jax.vmap(self.norm)(x)
        y = jax.vmap(self.mlp)(y)
        x = y + x

        return x




class TransformerLayer(eqx.Module):
    wte: nn.Embedding # Token embeddings
    wpe: nn.Embedding # Positional embeddings

    drop: nn.Dropout

    layers: list
    norm: nn.LayerNorm

    def __init__(self, config, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd, key=key1)
        self.wpe = nn.Embedding(config.block_size, config.n_embd, key=key2)
        self.drop = nn.Dropout(config.dropout)

        self.layers = [Block(config, key) for _ in range(config.n_layer)]
        self.norm = nn.LayerNorm(config.n_embd, use_bias=config.bias)

    def __call__(self, token_ids):
        (t,)= token_ids.shape
        
        # Should use better positional embeddings with cos and sin.
        pos = np.arange(0, t, dtype=np.int64)

        tok_emb = jax.vmap(self.wte)(token_ids)
        pos_emb = jax.vmap(self.wpe)(pos)

        # Dropout at the first layer ? Seems a bit aggressive...
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.layers:
            x = block(x)
        x = jax.vmap(self.norm)(x)

        return x




class GPT(eqx.Module):
    transformer: TransformerLayer
    lm_head: nn.Linear

    def __init__(self, config, key):
        key1, key2 = jax.random.split(key, 2)

        self.transformer = TransformerLayer(config, key1)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, use_bias=False, key=key2)

    def __call__(self, token_ids):
        y = self.transformer(token_ids)
        logits = jax.vmap( self.lm_head)(y)
        return logits
        
