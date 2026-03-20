# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal

from mava.networks.distributions import IdentityTransformation, TanhTransformedDistribution


class DiscreteActionHead(nn.Module):
    """Discrete Action Head"""

    action_dim: int

    @nn.compact
    def __call__(
        self,
        obs_embedding: chex.Array,
        action_mask: chex.Array,
    ) -> tfd.TransformedDistribution:
        """Action selection for distrete action space environments.

        Args:
        ----
            obs_embedding: Observation embedding from network torso.
            action_mask: Legal action mask for masked distributions.

        Returns:
        -------
            A transformed tfd.categorical distribution on the action space for action sampling.

        NOTE: We pass both the observation embedding and the observation object to the action head
        since the observation object contains the action mask and other potentially useful
        information.

        """
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(obs_embedding)

        masked_logits = jnp.where(
            action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )

        #  We transform this distribution with the `Identity()` transformation to
        # keep the API identical to the ContinuousActionHead.
        return IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))


# class ContinuousActionHead(nn.Module):
#     """ContinuousActionHead using a transformed Normal distribution.

#     Note: This network only handles the case where actions lie in the interval [-1, 1].
#     """

#     action_dim: int
#     min_scale: float = 1e-3
#     independent_std: bool = True  # whether or not the log_std is independent of the observation.

#     def setup(self) -> None:
#         self.mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))

#         if self.independent_std:
#             self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
#         else:
#             self.log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))

#     @nn.compact
#     def __call__(self, obs_embedding: chex.Array, action_mask: chex.Array) -> tfd.Independent:
#         """Action selection for continuous action space environments.

#         Args:
#         ----
#             obs_embedding: Observation embedding.
#             action_mask: Legal action mask for masked distributions. NOTE: In the
#                 continuous case, the action mask is not used but we still pass it in
#                 to keep the API consistent between the discrete and continuous cases.

#         Returns:
#         -------
#             tfd.Independent: Independent transformed distribution.

#         """
#         del action_mask
#         loc = self.mean(obs_embedding)

#         if self.independent_std:
#             scale = self.log_std * jnp.ones_like(loc)
#         else:
#             scale = self.log_std(obs_embedding)
#         scale = jax.nn.softplus(scale) + self.min_scale

#         distribution = tfd.Normal(loc=loc, scale=scale)

#         return tfd.Independent(
#             TanhTransformedDistribution(distribution),
#             reinterpreted_batch_ndims=1,
#         )

def safe_normalize(v: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)


class ContinuousActionHead(nn.Module):
    """Mava-compatible continuous action head with an equivariant mean.

    Expects obs_embedding = concat([h, v_r, v_u]) where:
      - h is scalar/context embedding (rotation-invariant content)
      - v_r, v_u are equivariant vector summaries in R^3 each

    Mean:
      loc = action_scale * sigmoid(mag(h)) * normalize( sigmoid(g_r,g_u)(h)[0]*v_r + ... )
    Std:
      - independent_std=True: single learned log_std per action dim (broadcast)
      - independent_std=False: log_std = Dense(h), depending only on h (keeps std rotation-stable)
    """

    action_dim: int
    min_scale: float = 1e-3
    independent_std: bool = True
    action_scale_accel: float = 2.0  # scale before tanh
    action_scale_att: float = 1.0

    def setup(self) -> None:
        self.gates = nn.Dense(2, kernel_init=orthogonal(0.01))    # g_r, g_u in (0,1)
        self.mag = nn.Dense(1, kernel_init=orthogonal(0.01))      # magnitude in (0,1)
        self.att_mean = nn.Dense(3, kernel_init=orthogonal(0.01)) # attitude update

        if self.independent_std:
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        else:
            self.log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))

    @nn.compact
    def __call__(self, obs_embedding: chex.Array, action_mask: chex.Array) -> tfd.Independent:
        del action_mask

        # Split embedding: last 6 dims are v_r (3) and v_u (3)
        h   = obs_embedding[..., :-6]
        v_r = obs_embedding[..., -6:-3]
        v_u = obs_embedding[..., -3:]

        g   = jax.nn.sigmoid(self.gates(h))
        g_r = g[..., 0:1]
        g_u = g[..., 1:2]

        v = g_r * v_r + g_u * v_u
        direction = safe_normalize(v)            # (..,3)
        magnitude = jax.nn.sigmoid(self.mag(h))  # (..,1)

        # 3D acceleration is rotation-equivariant, while attitude isn't geometric.
        loc_accel = self.action_scale_accel * magnitude * direction  # (..,3)
        loc_att = self.action_scale_att * self.att_mean(h)
        loc = jnp.concatenate([loc_accel, loc_att], axis=-1)         # (..,6)

        if self.independent_std:
            scale_raw = self.log_std * jnp.ones_like(loc)
        else:
            scale_raw = self.log_std(h)  # depend only on h (rotation-stable)
        scale_raw = jnp.clip(scale_raw, -20.0, 5.0)
        scale = jax.nn.softplus(scale_raw) + self.min_scale

        distribution = tfd.Normal(loc=loc, scale=scale)
        return tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )