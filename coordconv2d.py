Array = jnp.ndarray
Padding = Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]

class CoordConv2D(nn.Module):
    """CoordConv2D (NHWC, coords in [-1, 1])."""
    features: int
    kernel_size: Tuple[int, int]
    with_r: bool = False
    strides: Tuple[int, int] = (1, 1)
    padding: Padding = "SAME"
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    feature_group_count: int = 1
    kernel_dilation: Tuple[int, int] = (1, 1)
    precision: Optional[jax.lax.Precision] = None

    @nn.compact
    def __call__(self, x: Array) -> Array:
        b, h, w, _ = x.shape  # NHWC

        # Coordinate grids in [-1, 1], shape (h, w)
        xs = jnp.linspace(-1.0, 1.0, w, dtype=x.dtype)
        ys = jnp.linspace(-1.0, 1.0, h, dtype=x.dtype)
        xv, yv = jnp.meshgrid(xs, ys, indexing="xy")  # (h, w), (h, w)

        # Stack coord channels -> (h, w, 2 or 3)
        if self.with_r:
            rv = jnp.sqrt(xv**2 + yv**2)
            coord = jnp.stack([xv, yv, rv], axis=-1)
        else:
            coord = jnp.stack([xv, yv], axis=-1)

        # Broadcast to batch and concatenate with input along channel axis
        coord = jnp.broadcast_to(coord, (b, h, w, coord.shape[-1]))
        x_aug = jnp.concatenate([x, coord], axis=-1)  # NHWC

        conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            feature_group_count=self.feature_group_count,
            kernel_dilation=self.kernel_dilation,
            precision=self.precision,
        )
        return conv(x_aug)
