import jax
import jax.numpy as np
import equinox.nn as nn
import equinox as eqx
import typing as tp


class ResBlock(eqx.Module):
    """The residual unit contains two convolutions with kernel size 3 and a skip-connection.

    Args:
        eqx (_type_): _description_
    """

    conv1: nn.Conv1d
    conv2: nn.Conv1d
    activation: tp.Callable
    norm: eqx.Module

    def __init__(
        self,
        channels: int,
        activation: jax.Array = jax.nn.relu,
        norm: eqx.nn = eqx.nn.WeightNorm,
        norm_params: tp.Dict[str, tp.Any] = {},
        key=None,
    ):
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding="SAME",
            use_bias=True,
            key=key,
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding="SAME",
            use_bias=True,
            key=key,
        )
        self.activation = activation

        self.norm = norm

    # TODO Currently using trueskip.
    # TODO Currently don't have a hidden dim in the ResNet.
    # TODO Could have SEBlock !
    def __call__(self, x):
        y = self.conv1(x)
        y = self.activation(y)

        y = self.conv2(y)
        y = self.activation(y)

        return y + x


class EncoderLayer(eqx.Module):
    resblocks: list
    conv: nn.Conv1d
    activation: tp.Callable
    norm: eqx.Module

    def __init__(
        self,
        channels: int,
        remaining_shape: int,
        activation: jax.Array = jax.nn.relu,
        norm: eqx.nn = eqx.nn.WeightNorm,
        norm_params: tp.Dict[str, tp.Any] = {},
        n_res_layers: int = 2,
        ratio: int = 2,
        key=None,
    ):
        key1, key2 = jax.random.split(key, 2)

        keys = jax.random.split(key1, n_res_layers)
        self.resblocks = [
            ResBlock(
                channels=channels,
                activation=activation,
                norm=norm,
                norm_params=norm_params,
                key=key_i,
            )
            for key_i in keys
        ]
        self.conv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=ratio * 2,
            stride=ratio,
            padding="SAME",
            use_bias=True,
            key=key2,
        )
        self.activation = activation
        self.norm = nn.LayerNorm(shape=(channels * 2, remaining_shape))

    def __call__(self, x):
        y = x

        for res in self.resblocks:
            y = res(y)

        y = self.activation(y)
        y = self.conv(y)
        # y = self.norm(y)

        return y


class Encoder(eqx.Module):
    """SEANet encoder.
    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order
        activation (str): Activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
    """

    B_layers: list

    activation: tp.Callable

    first: nn.Conv1d
    last: nn.Conv1d

    norm_first: eqx.Module
    norm_last: eqx.Module

    # TODO Didn't implement LSTM for now
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_res_layers: int = 1,
        ratios: tp.List[int] = [2, 4, 5, 8],
        activation: jax.nn = jax.nn.relu,
        norm: nn = nn.WeightNorm,
        norm_params: tp.Dict[str, tp.Any] = {},
        lstm: int = 2,
        key=None,
    ):
        key1, key2, key3 = jax.random.split(key, 3)

        self.first = nn.Conv1d(
            channels, n_filters, kernel_size=7, padding="SAME", use_bias=True, key=key1
        )

        keys = jax.random.split(key2, len(ratios))
        self.B_layers = [
            EncoderLayer(
                channels=n_filters * (2**i),
                ratio=ratio,
                norm=norm,
                norm_params=norm_params,
                n_res_layers=n_res_layers,
                remaining_shape=int((24000 / np.cumprod(np.array(ratios)))[i].item()),
                key=key_i,
            )
            for i, (ratio, key_i) in enumerate(zip(ratios, keys))
        ]
        self.norm_first = nn.LayerNorm(shape=(n_filters, 24000))
        self.last = nn.Conv1d(
            n_filters * (2 ** len(ratios)),
            dimension,
            kernel_size=7,
            padding="SAME",
            use_bias=True,
            key=key3,
        )
        remain = int(24000 / np.cumprod(np.array(ratios))[-1].item())
        self.norm_last = nn.LayerNorm(shape=(dimension,remain))
        self.activation = activation

    def __call__(self, x):
        y = x

        y = self.first(y)
        y = self.norm_first(y)

        for layer in self.B_layers:
            y = layer(y)

        y = self.activation(y)
        y = self.last(y)
        y = self.norm_last(y)

        return y


class DecoderLayer(eqx.Module):
    resblocks: list
    conv: nn.ConvTranspose1d
    activation: tp.Callable
    norm: eqx.Module

    def __init__(
        self,
        channels: int,
        remaining_shape: int,
        activation: jax.Array = jax.nn.relu,
        norm: eqx.nn = eqx.nn.WeightNorm,
        norm_params: tp.Dict[str, tp.Any] = {},
        n_res_layers: int = 2,
        ratio: int = 2,
        key=None,
    ):
        key1, key2 = jax.random.split(key, 2)

        self.conv = nn.ConvTranspose1d(
            channels,
            channels // 2,
            kernel_size=ratio * 2,
            stride=ratio,
            padding="SAME",
            use_bias=True,
            key=key1,
        )

        keys = jax.random.split(key2, n_res_layers)
        self.resblocks = [
            ResBlock(
                channels=channels // 2,
                activation=activation,
                norm=norm,
                norm_params=norm_params,
                key=key_i,
            )
            for key_i in keys
        ]
        self.activation = activation
        self.norm = nn.LayerNorm(shape=(channels // 2, remaining_shape))

    def __call__(self, x):
        y = self.activation(x)
        y = self.conv(x)
        y = self.norm(y)

        for res in self.resblocks:
            y = res(y)

        return y


class Decoder(eqx.Module):

    B_layers: list

    final_activation: tp.Callable
    activation: tp.Callable

    first: nn.ConvTranspose1d
    last: nn.ConvTranspose1d

    norm_first: eqx.Module
    norm_last: eqx.Module

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_res_layers: int = 1,
        ratios: list = [8, 5, 4, 2],
        cum_ratios: tp.List[int] = [320, 40, 8, 2, 1],
        activation=jax.nn.relu,
        final_activation=None,
        norm: eqx.nn = eqx.nn.WeightNorm,
        norm_params: tp.Dict[str, tp.Any] = {},
        lstm: int = 2,
        key=None,
    ):
        key1, key2, key3 = jax.random.split(key, 3)

        self.first = nn.ConvTranspose1d(
            dimension,
            n_filters * (2 ** len(ratios)),
            kernel_size=7,
            padding="SAME",
            use_bias=True,
            key=key1,
        )
        
        self.norm_first = nn.LayerNorm(shape=(n_filters * (2 ** len(ratios)), 24000 // cum_ratios[0]))

        keys = jax.random.split(key2, len(ratios))
        self.B_layers = [
            DecoderLayer(
                n_filters * (2 ** (len(ratios) - i)),
                remaining_shape=24000 // cum_ratios[i+1],
                activation=activation,
                norm=norm,
                norm_params=norm_params,
                n_res_layers=n_res_layers,
                ratio=ratio,
                key=key_i,
            )
            for i, (ratio, key_i) in enumerate(zip(ratios, keys))
        ]

        self.last = nn.ConvTranspose1d(
            n_filters, channels, kernel_size=7, padding="SAME", use_bias=True, key=key3
        )
        self.norm_last = nn.LayerNorm(shape=(1, 24000))
        self.activation = activation
        self.final_activation = (
            nn.Identity() if final_activation is None else final_activation
        )

    def __call__(self, x):
        y = x

        y = self.activation(y)
        y = self.first(y)
        y = self.norm_first(y)

        for layer in self.B_layers:
            y = layer(y)

        y = self.activation(y)
        y = self.last(y)
        y = self.norm_last(y)

        y = self.final_activation(y)

        return y


# class Encodec(eqx.Module):
#     def __init__(self, channels=32, layers=4, strides=[2, 4, 5, 8], kernel_size=None, key=None):

# def test():
#     key = jax.random.PRNGKey(9)
#     encoder = Encoder(key=key)
#     decoder = Decoder(key=key)
#     x = jax.random.normal(key, shape=(1, 1, 24000))
#     print(x.shape)
#     z = jax.vmap(encoder)(x)
#     print(z.shape)
#     y = jax.vmap(decoder)(z)
#     print(y.shape)

# if __name__ == "__main__":
#     test()


class EncodecModel(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, activation: jax.Array, n_res_layers: int = 1, key=None):
        key1, key2 = jax.random.split(key, 2)

        self.encoder = Encoder(
            activation=activation,
            norm=nn.LayerNorm,
            n_res_layers=n_res_layers,
            key=key1,
        )
        self.decoder = Decoder(
            activation=activation,
            norm=nn.LayerNorm,
            n_res_layers=n_res_layers,
            key=key2,
        )

    # We take a frame as input TODO could define a segment length and return a list of frames having iterated through them.
    def __call__(self, x):
        # X comes in as a 2d vector of [1 x Samples] we only do mono for now TODO
        z = self.encoder(x)
        y = self.decoder(z)

        return y
