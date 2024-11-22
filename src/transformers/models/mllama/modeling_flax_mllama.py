# coding=utf-8
# Copyright 2021 The OpenAI Team Authors, The Google Flax Team Authors and The HuggingFace Inc. team.
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
"""Flax Mllama model."""
from typing import Any, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_mllama import MllamaConfig, MllamaTextConfig, MllamaVisionConfig
import math

logger = logging.get_logger(__name__)

CLIP_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a
    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
    behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class FlaxMllamaVisionAttention(nn.Module):
  config: Union[MllamaTextConfig, MllamaVisionConfig]
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.embed_dim = self.config.hidden_size
    self.num_heads = self.config.attention_heads
    self.head_dim = self.embed_dim // self.num_heads
    if self.head_dim * self.num_heads != self.embed_dim:
      raise ValueError(
          f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
          f" {self.num_heads})."
      )
    self.scale = self.head_dim**-0.5

    self.k_proj = nn.Dense(
        self.embed_dim, dtype=self.dtype, param_dtype=self.weights_dtype, kernel_init=jax.nn.initializers.normal(0.01),
        use_bias=False
    )
    self.v_proj = nn.Dense(
        self.embed_dim, dtype=self.dtype, param_dtype=self.weights_dtype, kernel_init=jax.nn.initializers.normal(0.01),
        use_bias=False
    )
    self.q_proj = nn.Dense(
        self.embed_dim, dtype=self.dtype, param_dtype=self.weights_dtype, kernel_init=jax.nn.initializers.normal(0.01),
        use_bias=False
    )
    self.out_proj = nn.Dense(
        self.embed_dim, dtype=self.dtype, param_dtype=self.weights_dtype, kernel_init=jax.nn.initializers.normal(0.01),
        use_bias=False
    )

  def _split_heads(self, hidden_states):
    return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

  def _merge_heads(self, hidden_states):
    return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

  def __call__(
      self,
      hidden_states,
      attention_mask: Optional[jnp.array] = None,
      deterministic: bool = True,
      output_attentions: bool = None,
  ) -> jnp.array:
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query)
    key = self._split_heads(key)
    value = self._split_heads(value)

    if attention_mask is not None:# Ensure the attention mask matches the key sequence length
      causal_mask = attention_mask[:, :, :, : key.shape[-3]] # (batch_size, 1, q_seq_len, kv_seq_len)

    attn_weights = dot_product_attention_weights(
        query,
        key,
        mask=causal_mask,
        deterministic=deterministic,
        dtype=self.dtype,
    )

    attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
    attn_output = self._merge_heads(attn_output)
    attn_output = self.out_proj(attn_output)

    outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
    return outputs


class FlaxMllamaVisionMLP(nn.Module):
  config: Union[MllamaTextConfig, MllamaVisionConfig]
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.activation_fn = ACT2FN[self.config.hidden_act]
    self.fc1 = nn.Dense(
        self.config.intermediate_size,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        kernel_init=jax.nn.initializers.normal(0.01),
    )
    self.fc2 = nn.Dense(
        self.config.hidden_size,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        kernel_init=jax.nn.initializers.normal(0.01),
    )

  def __call__(self, hidden_states):
    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    hidden_states = self.fc2(hidden_states)
    return hidden_states


class FlaxMllamaVisionEncoderLayer(nn.Module):
  config: Union[MllamaTextConfig, MllamaVisionConfig]
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  is_gated: bool = False

  def setup(self):
    self.self_attn = FlaxMllamaVisionAttention(self.config, dtype=self.dtype, weights_dtype=self.weights_dtype)
    self.input_layernorm = nn.LayerNorm(epsilon=self.config.norm_eps, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.mlp = FlaxMllamaVisionMLP(self.config, dtype=self.dtype, weights_dtype=self.weights_dtype)
    self.post_attention_layernorm = nn.LayerNorm(epsilon=self.config.norm_eps, dtype=self.dtype, param_dtype=self.weights_dtype)

    if self.is_gated:
      self.gate_attn = self.param('gate_attn', nn.initializers.constant(math.pi / 4), (1,))
      self.gate_ffn = self.param('gate_ffn', nn.initializers.constant(math.pi / 4), (1,))

  def __call__(
      self,
      hidden_states,
      attention_mask,
      deterministic: bool = True,
      output_attentions: bool = False,
  ):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    output= self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        deterministic=deterministic,
        output_attentions=output_attentions,
    )
    # Apply residual connection with optional gating for attention
    if self.is_gated:
        hidden_states = jnp.tanh(self.gate_attn) * output[0]
    hidden_states = residual + output[0]

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if self.is_gated:
      hidden_states = jnp.tanh(self.gate_ffn) * hidden_states
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
      outputs += (output[1],)

    return outputs


class FlaxMllamaVisionEncoder(nn.Module):
  config: Union[MllamaTextConfig, MllamaVisionConfig]
  num_layers: 32
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.layers = [
        FlaxMllamaVisionEncoderLayer(self.config, name="layers."+str(i), dtype=self.dtype, weights_dtype=self.weights_dtype)
        for i in range(self.num_layers)
    ]

  def __call__(
      self,
      hidden_states,
      attention_mask=None,
      deterministic: bool = True,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
      return_dict: bool = True,
  ):
    all_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    for layer in self.layers:
      if output_hidden_states:
        all_hidden_states += (hidden_states,)

      layer_outputs = layer(hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions)
      hidden_states = layer_outputs[0]

      if output_attentions:
        all_attentions += (layer_outputs[1],)

    if output_hidden_states:
      all_hidden_states += (hidden_states,)

    if not return_dict:
      return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

    return FlaxBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

class FlaxMllamaPrecomputedPositionEmbedding(nn.Module):
    """
    FlaxMllamaPrecomputedPositionEmbedding is a neural network module that computes position embeddings for input hidden states.
    It uses precomputed position embeddings and tile position embeddings based on aspect ratio IDs.

    Args:
        config: Configuration object containing model hyperparameters.
    """
    config: MllamaVisionConfig
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_state: jnp.ndarray, aspect_ratio_ids: jnp.ndarray) -> jnp.ndarray:
        max_num_tiles = self.config.max_num_tiles  # 4
        max_aspect_ratio_id = self.config.max_aspect_ratio_id  # 8
        num_patches = (self.config.image_size // self.config.patch_size) ** 2 + 1  # 1025
        hidden_size = self.config.hidden_size  # 1280
        scale = hidden_size ** -0.5

        # Learnable gate parameter
        gate = self.param("gate", jax.nn.initializers.zeros, (1,))

        # Position embedding
        position_embedding = self.param(
            "embedding", 
            lambda key, shape: scale * jax.random.normal(key, shape), 
            (num_patches, hidden_size)
        )

        # Tile position embedding
        tile_embedding = nn.Embed(
            name="tile_embedding",
            num_embeddings=max_aspect_ratio_id + 1, 
            features=max_num_tiles * num_patches * hidden_size,
            dtype=self.dtype, param_dtype=self.weights_dtype,
        )
        # Apply gated position embedding
        gated_position_embedding = (1 - jnp.tanh(gate)) * position_embedding
        hidden_state = hidden_state + gated_position_embedding.reshape(1, 1, num_patches, hidden_size)

        # Precomputed tile position embeddings
        tile_position_embedding = tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, max_num_tiles, num_patches, hidden_size
        )
        gated_tile_position_embedding = jnp.tanh(gate) * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state

class FlaxMllamaVisionEmbeddings(nn.Module):
  config: MllamaVisionConfig
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    embed_dim = self.config.hidden_size
    image_size = self.config.image_size
    patch_size = self.config.patch_size

    self.class_embedding = self.param("class_embedding", jax.nn.initializers.normal(stddev=0.02), (embed_dim,))
    self.patch_embedding = nn.Conv(
        embed_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="VALID",
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        kernel_init=jax.nn.initializers.normal(),
    )

    self.num_patches = (image_size // patch_size) ** 2
    num_positions = self.num_patches + 1
    self.position_embedding = nn.Embed(
        num_positions,
        embed_dim,
        embedding_init=jax.nn.initializers.normal(),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )
    self.position_ids = jnp.expand_dims(jnp.arange(0, num_positions, dtype="i4"), axis=0)

  def __call__(self, pixel_values):
    patch_embeds = self.patch_embedding(pixel_values)
    batch_size, height, width, channels = patch_embeds.shape
    patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))

    class_embeds = jnp.expand_dims(self.class_embedding, axis=(0, 1))
    class_embeds = jnp.tile(class_embeds, (batch_size, 1, 1))
    embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings

class FlaxMllamaPrecomputedAspectRatioEmbedding(nn.Module):
    config: MllamaVisionConfig
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    is_gated: bool = True

    def setup(self):
        self.max_num_tiles = self.config.max_num_tiles
        self.hidden_size = self.config.hidden_size
        self.max_aspect_ratio_id = self.config.max_aspect_ratio_id

        self.embedding = nn.Embed(
            num_embeddings=self.max_aspect_ratio_id + 1,
            features=self.max_num_tiles * self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
        )
        if self.is_gated:
            self.gate = self.param("gate", jax.nn.initializers.zeros, (1,))

    def __call__(self, hidden_state, aspect_ratio_ids):
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * jnp.tanh(self.gate)

        hidden_state = hidden_state + embeddings
        return hidden_state
    

def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: jnp.ndarray,
    num_patches: int,
    target_length: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = jnp.expand_dims(aspect_ratio_mask, (2, 3)).astype(dtype)  # (batch_size, max_num_tiles, 1, 1)
    attention_mask = jnp.tile(attention_mask, (1, 1, target_length, 1))  # (batch_size, max_num_tiles, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    if pad_patches > 0:
        attention_mask = attention_mask.at[:, :, -pad_patches:].set(0)

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = jnp.matmul(attention_mask, attention_mask.swapaxes(2, 1)) * jnp.finfo(dtype).min
    attention_mask = jnp.expand_dims(attention_mask, axis=1)

    return attention_mask

class FlaxMllamaVisionTransformer(nn.Module):
  config: MllamaVisionConfig
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.image_size = self.config.image_size #448
    self.patch_size = self.config.patch_size #14
    self.max_num_tiles = self.config.max_num_tiles #4
    self.hidden_size = self.config.hidden_size #1280
    self.num_channels = self.config.num_channels #3
    self.intermediate_layers_indices = self.config.intermediate_layers_indices #[3, 7, 15, 23, 30]

    self.num_patches = (self.image_size // self.patch_size) ** 2 + 1 #1025
    self.scale = self.config.hidden_size**-0.5
  
    self.patch_embedding = nn.Conv(
        self.hidden_size,
        kernel_size=(self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        padding="VALID",
        use_bias=False,
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.normal(),
    )

    self.class_embedding = self.param("class_embedding", jax.nn.initializers.normal(stddev=0.02), (self.config.hidden_size,))
    self.gated_positional_embedding = FlaxMllamaPrecomputedPositionEmbedding(self.config, dtype=self.dtype, weights_dtype=self.weights_dtype)

    self.pre_tile_positional_embedding = FlaxMllamaPrecomputedAspectRatioEmbedding(self.config, is_gated=True, dtype=self.dtype, weights_dtype=self.weights_dtype)
    self.post_tile_positional_embedding = FlaxMllamaPrecomputedAspectRatioEmbedding(self.config, is_gated=True, dtype=self.dtype, weights_dtype=self.weights_dtype)

    self.layernorm_pre = nn.LayerNorm(epsilon=self.config.norm_eps, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.layernorm_post = nn.LayerNorm(epsilon=self.config.norm_eps, dtype=self.dtype, param_dtype=self.weights_dtype)
    
    self.transformer = FlaxMllamaVisionEncoder(self.config, self.config.num_hidden_layers, dtype=self.dtype, weights_dtype=self.weights_dtype)
    self.global_transformer = FlaxMllamaVisionEncoder(self.config, self.config.num_global_layers, dtype=self.dtype, weights_dtype=self.weights_dtype)

  def get_input_embeddings(self):
      """
      This function is used to fetch the first embedding layer to activate grads on inputs.
      """
      return self.patch_embedding
  
  def apply_class_embedding(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
      batch_size, _, hidden_size = hidden_state.shape
      class_embedding = jnp.expand_dims(self.class_embedding, axis=(0, 1))
      class_embedding = jnp.tile(class_embedding, (batch_size, 1, 1))
      hidden_state = jnp.concatenate([class_embedding, hidden_state], axis=1)
      return hidden_state
  
  def __call__(
      self,
      pixel_values: jnp.ndarray,
      aspect_ratio_ids: jnp.ndarray,
      aspect_ratio_mask: jnp.ndarray,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
      return_dict: bool = True,
  ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape((batch_size * num_concurrent_media * num_tiles, num_channels, height, width))
        aspect_ratio_ids = aspect_ratio_ids.reshape((batch_size * num_concurrent_media, -1))

        # Patch embedding
        print("5", pixel_values.shape)
        patch_embeds = self.patch_embedding(pixel_values.transpose((0, 2, 3, 1)))
        patch_embeds = patch_embeds.transpose((0, 3, 1, 2))
        print("6", patch_embeds.shape)
        hidden_state = patch_embeds.reshape(batch_size * num_concurrent_media * num_tiles, self.hidden_size, -1).swapaxes(1, 2)
        print("4", hidden_state.shape)
        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles, -1, dim))
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # Add cls token
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media * num_tiles, num_patches, dim))
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles, num_patches, dim))
        print("3", hidden_state.shape)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        padding = [(0, 0), (0, 0), (0, num_padding_patches), (0, 0)]
        hidden_state = jnp.pad(hidden_state, padding, mode="constant", constant_values=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape((batch_size * num_concurrent_media, -1))
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
            dtype=hidden_state.dtype,
        )

        # Apply encoder
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, -1, dim))
        print("2", hidden_state.shape)
        output = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
        hidden_state = output[0]
        print("1", hidden_state.shape)

        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim))
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles * (num_patches + num_padding_patches), dim))
        global_output = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_state = global_output[0]

        # Remove padding from hidden state
        hidden_state = hidden_state.reshape((batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, dim))
        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape((batch_size, num_concurrent_media, num_tiles, num_patches, dim))

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = output[1]
        intermediate_hidden_states = jnp.stack(all_intermediate_hidden_states, axis=-1)
        intermediate_hidden_states = intermediate_hidden_states[..., self.intermediate_layers_indices]

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape((batch_size * num_concurrent_media, num_tiles, num_patches + num_padding_patches, -1))
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape((batch_size, num_concurrent_media, num_tiles, num_patches, -1))

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = jnp.concatenate([hidden_state, intermediate_hidden_states], axis=-1)

        if output_hidden_states:
            hidden_states = tuple(all_intermediate_hidden_states) + tuple(global_output[1])
        else:
            hidden_states = None

        if output_attentions:
            global_attn = tuple(global_output[2]) if output_hidden_states else tuple(global_output[1])
            attentions = tuple(output[2]) + global_attn
        else:
            attentions = None

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states, attentions] if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )

class FlaxMllamaVisionPreTrainedModel(FlaxPreTrainedModel):
  config_class = MllamaVisionConfig
#   main_input_name = "pixel_values"
  module_class: nn.Module = None

  def __init__(
      self,
      config: MllamaVisionConfig,
      input_shape: Optional[Tuple] = None,
      seed: int = 0,
      dtype: jnp.dtype = jnp.float32,
      _do_init: bool = True,
      **kwargs,
  ):
    if input_shape is None:
      input_shape = (1, 1, config.max_num_tiles, config.num_channels, config.image_size, config.image_size,)
    module = self.module_class(config=config, dtype=dtype, **kwargs)
    super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

  def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
    # init input tensor
    pixel_values = jax.random.normal(rng, input_shape)
    aspect_ratio_ids = jnp.zeros(input_shape[:2], dtype="i4")
    aspect_ratio_mask = jnp.ones(input_shape[:3])

    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {"params": params_rng, "dropout": dropout_rng}

    random_params = self.module.init(rngs, pixel_values, aspect_ratio_ids, aspect_ratio_mask)["params"]

    if params is not None:
      random_params = flatten_dict(unfreeze(random_params))
      params = flatten_dict(unfreeze(params))
      for missing_key in self._missing_keys:
        params[missing_key] = random_params[missing_key]
      self._missing_keys = set()
      return freeze(unflatten_dict(params))
    else:
      return random_params

  def __call__(
      self,
      pixel_values,
      aspect_ratio_ids: jnp.ndarray,
      aspect_ratio_mask: jnp.ndarray,
      params: dict = None,
      dropout_rng: jax.random.PRNGKey = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
  ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.return_dict

    # Handle any PRNG if needed
    rngs = {}
    if dropout_rng is not None:
      rngs["dropout"] = dropout_rng

    return self.module.apply(
        {"params": params or self.params},
        jnp.array(pixel_values, dtype=jnp.float32),
        aspect_ratio_ids,
        aspect_ratio_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        rngs=rngs,
    )

class FlaxMllamaVisionModule(nn.Module):
  config: MllamaVisionConfig
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.vision_model = FlaxMllamaVisionTransformer(self.config, dtype=self.dtype)

  def __call__(
      self,
      pixel_values: jnp.ndarray,
      aspect_ratio_ids: jnp.ndarray,
      aspect_ratio_mask: jnp.ndarray,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
      return_dict: bool = True,
  ):
    return self.vision_model(
        pixel_values=pixel_values,
        aspect_ratio_ids=aspect_ratio_ids, # a list of aspect ratio ids for the images in the data
        aspect_ratio_mask=aspect_ratio_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


class FlaxMllamaVisionModel(FlaxMllamaVisionPreTrainedModel):
  module_class = FlaxMllamaVisionModule
  base_model_prefix = "vision_model"


FLAX_Mllama_VISION_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, FlaxMllamaVisionModel

    >>> model = FlaxMllamaVisionModel.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
    >>> processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooler_output = outputs.pooler_output  # pooled CLS states
    ```
"""

overwrite_call_docstring(FlaxMllamaVisionModel, CLIP_VISION_INPUTS_DOCSTRING + FLAX_Mllama_VISION_MODEL_DOCSTRING)
append_replace_return_docstrings(
    FlaxMllamaVisionModel, output_type=FlaxBaseModelOutputWithPooling, config_class=MllamaVisionConfig
)
