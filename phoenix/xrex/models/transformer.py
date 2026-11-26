# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from dataclasses import (
    dataclass,
    replace,
)
from functools import partial
from typing import Callable, Optional, Tuple, TypeAlias, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.xla_metadata import set_xla_metadata
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from xai_configlib import Config, configclass
from xrex.data.recsys.sequence_packing import SequencePackedLayout, packed_prefix_positions
from xrex.models.attention import AttentionConfig
from xrex.models.fp8 import Fp8Dot, Fp8Gemms
from xrex.models.layers import (
    CandidateOnlyAttention,
    Linear,
    MultiHeadAttention,
    MultiHeadAttentionOutput,
)
from xrex.models.normalization import RMSNorm
from xrex.models.remat import (
    ATTN_FP8_T,
    DENSE_GELU_FP8_T,
    NORM_STATS,
    POST_ATTN_RESIDUAL,
    RematType,
    checkpoint_as_bits,
    custom_remat_policy,
)
from xrex.models.scaling import ScaleConfig
from xrex.models.sharding_context import NamedShape, ShardingContext
from xrex.utils import gpu, layer_stack
from xrex.utils.gpu import peak_tflops
from xrex.utils.sharding import with_sharding_constraint_unless_manual
from xrex.utils.utils import (
    dump_block_outputs,
    ffn_size,
    gelu_fn,
    remat_less,
    summarize,
    unroll_layer_dumps,
)

DecoderInput: TypeAlias = jax.Array


@chex.dataclass(kw_only=True)
class TransformerModelOutput:
    output: jax.Array


@configclass
class FeedForwardConfig(Config):
    widening_factor: float
    gated_mlp: bool = False
    with_bias: bool = False
    force_ffn_base_size: bool = True
    base_ffn_size: int = 32
    merge_gate_up: bool = False


@configclass
class TransformerConfig(Config):
    emb_size: int
    num_layers: int
    sequence_len: int
    ffn_config: FeedForwardConfig
    attn_config: Optional[AttentionConfig] = None
    scale_config: ScaleConfig = ScaleConfig()
    pre_norm: bool = True
    primer_norm: bool = False
    layer_norm_eps: float = 1e-5
    output_vocab_size: Optional[int] = None

    use_layer_stack: bool = False
    unroll_layer_stack: bool = False
    num_candidate_only_layers: int = 0
    candidate_only_layers_update_user_token: bool = False

    use_remat: bool = True
    remat_policy: RematType = RematType.WHOLE
    fp8_gemms: Fp8Gemms = Fp8Gemms.OFF
    prevent_cse: bool = True
    remat_boundary: str = "block"

    data_axis: Union[str, Tuple[str, ...]] = ("expert", "replica", "data")

    debug_tensor_dump_output_folder: Optional[str] = None
    debug_tensor_dump_only_target_logprobs: bool = False
    debug_tensor_dump_n_tokens: Optional[int] = None
    debug_tensor_dump_n_layers: Optional[int] = None
    debug_tensor_dump_num_dumps: int = 1
    debug_tensor_dump_start_pos: int = 0

    def __post_init__(self):
        if isinstance(self.data_axis, list):
            self.data_axis = tuple(self.data_axis)
        self.fp8_gemms = Fp8Gemms(self.fp8_gemms)

    @property
    def ffn_dim(self):
        return int(self.ffn_config.widening_factor * self.emb_size)

    def compute_forward_flops_per_sec(self, num_seq_per_sec: float, S: int) -> tuple[float, float]:
        m = self

        def attn_flops():
            num_q_heads, num_kv_heads = m.attn_config.num_q_heads, m.attn_config.num_kv_heads
            key_size, value_size = m.attn_config.key_size, m.attn_config.key_size
            q = 2 * S * m.emb_size / 1e12 * num_q_heads * key_size
            kv = 2 * S * m.emb_size / 1e12 * num_kv_heads * (key_size + value_size)
            qk_score = num_q_heads * (2 * S * S / 2 * key_size) / 1e12
            attn_dot_value = 2 * S * S / 2 * value_size * num_q_heads / 1e12
            proj = 2 * S * num_q_heads * value_size * m.emb_size / 1e12
            num_layers = m.num_layers
            attn = (q + kv + qk_score + attn_dot_value + proj) * num_layers * num_seq_per_sec
            return attn

        ffn_dim = m.ffn_dim

        z1 = 2 * S * m.emb_size / 1e12 * ffn_dim
        z2 = 2 * S * ffn_dim / 1e12 * m.emb_size

        o = 2 * S * m.emb_size / 1e12 * m.output_vocab_size

        attn = attn_flops()
        ffn = ((z1 + z2) * m.num_layers + o) * num_seq_per_sec

        return attn, ffn

    def compute_tflops(self, num_seq_per_sec: float, S: int) -> float:
        attn, ffn = self.compute_forward_flops_per_sec(num_seq_per_sec, S)
        total_flops_per_second = (attn + ffn) * 3
        return round(total_flops_per_second, 2)

    def compute_mfu(
        self, num_seq_per_sec: float, S: int, peak_tflops_override: Optional[float] = None
    ) -> float:
        if peak_tflops_override is None:
            peak_tflops_override = peak_tflops()
        return self.compute_tflops(num_seq_per_sec, S) / peak_tflops_override

    def make(self, sharding_context: ShardingContext) -> hk.Module:
        return Transformer(
            config=self,
            sharding_context=sharding_context,
        )


@dataclass
class MHABlock(hk.Module):
    config: AttentionConfig
    scale_config: ScaleConfig
    sharding_context: ShardingContext
    fp8_gemms: Fp8Gemms = Fp8Gemms.OFF
    fp8_remat_saves: bool = False

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        segment_ids: jax.Array,
        segment_ids_k: Optional[jax.Array] = None,
        positions: jax.Array | None = None,
        seqpack_layout: SequencePackedLayout | None = None,
        history_states: jax.Array | None = None,
    ) -> MultiHeadAttentionOutput:
        _, _, model_size = inputs.shape
        initializer = hk.initializers.VarianceScaling(self.scale_config.attn_init_scale**2)

        def attn_block(
            input_to_query: jax.Array,
            input_to_key: jax.Array,
            input_to_value: jax.Array,
            mask: jax.Array,
            segment_ids: jax.Array,
            segment_ids_k: Optional[jax.Array] = None,
        ) -> MultiHeadAttentionOutput:
            def softmax_attn() -> MultiHeadAttentionOutput:
                return MultiHeadAttention(
                    model_size=model_size,
                    config=self.config,
                    scale_config=self.scale_config,
                    w_init=initializer,
                    sharding_context=self.sharding_context,
                    fp8_gemms=self.fp8_gemms,
                    out_proj_fp8_input_name=ATTN_FP8_T if self.fp8_remat_saves else None,
                )(
                    input_to_query,
                    input_to_key,
                    input_to_value,
                    mask,
                    segment_ids,
                    segment_ids_k=segment_ids_k,
                    positions=positions,
                    seqpack_layout=seqpack_layout,
                )

            return softmax_attn()

        attn_input_named_shape = NamedShape(
            inputs.shape, names=("batch", "activation_seq", "replicated")
        )
        activation_pspecs = self.sharding_context.sharding_specs(
            attn_input_named_shape, namespace=self.config.attn_sharding_namespace
        )
        inputs = with_sharding_constraint_unless_manual(
            inputs,
            activation_pspecs,
        )
        if history_states is None:
            assert mask.ndim == 4, f"shape: {mask.shape}"
            assert mask.shape[2] in {1, inputs.shape[1]}, str(mask.shape)
            assert mask.shape[3] in {1, inputs.shape[1]}, str(mask.shape)
            attn_output = attn_block(inputs, inputs, inputs, mask, segment_ids, segment_ids_k)
        else:
            history_states = with_sharding_constraint_unless_manual(
                history_states, activation_pspecs
            )
            attn_output = CandidateOnlyAttention(
                model_size=model_size,
                config=self.config,
                scale_config=self.scale_config,
                w_init=initializer,
                sharding_context=self.sharding_context,
                name="multi_head_attention",
            )(inputs, history_states, segment_ids, positions, seqpack_layout)

        attn_output = replace(
            attn_output,
            output=with_sharding_constraint_unless_manual(attn_output.output, activation_pspecs),
        )

        return attn_output


@dataclass
class DenseBlock(hk.Module):
    config: FeedForwardConfig
    scale_config: ScaleConfig
    data_axis: Union[str, Tuple[str, ...]]
    widening_factor: float = 1
    sharding_context: ShardingContext = None
    checkpoint_name_prefix: str = "dense_"
    suffix_name: str = ""
    fp8_gemms: Fp8Gemms = Fp8Gemms.OFF
    fp8_remat_saves: bool = False

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,
        is_training: bool = True,
    ) -> jax.Array:
        batch_size, seq_len, model_size = inputs.shape
        initializer = hk.initializers.VarianceScaling(self.scale_config.ffn_init_scale**2)
        ffn_size_ = ffn_size(
            emb_size=model_size,
            widening_factor=self.widening_factor,
            gated_mlp=self.config.gated_mlp,
            force_ffn_base_size=self.config.force_ffn_base_size,
            base_size=self.config.base_ffn_size,
        )

        input_spec = P(
            self.data_axis,
            self.sharding_context.logical_axis_to_physical("sequence"),
            self.sharding_context.logical_axis_to_physical("replicated"),
        )
        input_sharding = NamedSharding(self.sharding_context.mesh, input_spec)
        inputs = with_sharding_constraint_unless_manual(inputs, input_sharding)

        weights_pspec = P(None, None)
        activation_spec = P(
            self.data_axis,
            self.sharding_context.logical_axis_to_physical("dense_activation_seq"),
            self.sharding_context.logical_axis_to_physical("dense_activation_model"),
        )
        activation_sharding = NamedSharding(self.sharding_context.mesh, activation_spec)

        if self.config.gated_mlp:
            fp8_dot = Fp8Dot(self.fp8_gemms)
            if self.config.merge_gate_up:
                h = Linear(
                    ffn_size_ * 2,
                    w_init=initializer,
                    with_bias=self.config.with_bias,
                    sharding_context=self.sharding_context,
                    pspec=weights_pspec,
                    lr_multiplier=self.scale_config.hidden_lr_multiplier(model_size)
                    * self.scale_config.transformer_matrix_lr_scale,
                    init_scale=self.scale_config.ffn_init_scale,
                    name="gate_up_proj" + self.suffix_name,
                    dot=fp8_dot,
                )(inputs)

                h = jnp.reshape(h, (batch_size, seq_len, -1, 2))
                h = with_sharding_constraint_unless_manual(h, activation_sharding)
                h = checkpoint_name(h, "gate_up_proj")
                h_up, h_gate = jnp.split(h, 2, axis=-1)
                h_up = jnp.reshape(h_up, (batch_size, seq_len, -1))
                h_gate = h_gate.reshape(batch_size, seq_len, -1)
            else:
                h_up = Linear(
                    ffn_size_,
                    w_init=initializer,
                    with_bias=self.config.with_bias,
                    sharding_context=self.sharding_context,
                    pspec=weights_pspec,
                    lr_multiplier=self.scale_config.hidden_lr_multiplier(model_size)
                    * self.scale_config.transformer_matrix_lr_scale,
                    init_scale=self.scale_config.ffn_init_scale,
                    name="up_proj" + self.suffix_name,
                    dot=fp8_dot,
                )(inputs)
                h_gate = Linear(
                    ffn_size_,
                    w_init=initializer,
                    with_bias=self.config.with_bias,
                    sharding_context=self.sharding_context,
                    pspec=weights_pspec,
                    lr_multiplier=self.scale_config.hidden_lr_multiplier(model_size)
                    * self.scale_config.transformer_matrix_lr_scale,
                    init_scale=self.scale_config.ffn_init_scale,
                    name="gate_proj" + self.suffix_name,
                    dot=fp8_dot,
                )(inputs)
                h_up = checkpoint_name(h_up, f"{self.checkpoint_name_prefix}up_proj")
                h_gate = checkpoint_name(h_gate, f"{self.checkpoint_name_prefix}gate_proj")
            combined = h_up * gelu_fn(h_gate, approximate=True)
            combined = with_sharding_constraint_unless_manual(combined, activation_sharding)

            h_dense = Linear(
                model_size,
                w_init=initializer,
                with_bias=self.config.with_bias,
                pspec=weights_pspec,
                lr_multiplier=self.scale_config.ffn_output_lr_multiplier(ffn_size_, gated_mlp=True)
                * self.scale_config.transformer_matrix_lr_scale,
                init_scale=self.scale_config.ffn_init_scale,
                sharding_context=self.sharding_context,
                name="down_proj" + self.suffix_name,
                dot=fp8_dot,
            )(combined)
        else:
            gelu_fp8_name = DENSE_GELU_FP8_T if self.fp8_remat_saves else None
            down_dot = Fp8Dot(self.fp8_gemms, transposed_input_checkpoint_name=gelu_fp8_name)
            up_dot = Fp8Dot(self.fp8_gemms, fuse_gradient_amax=self.fp8_remat_saves)
            h_up = Linear(
                ffn_size_,
                w_init=initializer,
                with_bias=self.config.with_bias,
                sharding_context=self.sharding_context,
                pspec=weights_pspec,
                lr_multiplier=self.scale_config.hidden_lr_multiplier(model_size)
                * self.scale_config.transformer_matrix_lr_scale,
                name="up_proj" + self.suffix_name,
                dot=up_dot,
            )(inputs)
            checkpoint_fn = checkpoint_as_bits if self.fp8_remat_saves else checkpoint_name
            h_up = checkpoint_fn(h_up, f"{self.checkpoint_name_prefix}up_proj")
            h_up = gelu_fn(h_up, approximate=True)
            h_dense = Linear(
                model_size,
                w_init=initializer,
                with_bias=self.config.with_bias,
                pspec=weights_pspec,
                lr_multiplier=self.scale_config.ffn_output_lr_multiplier(ffn_size_, gated_mlp=True)
                * self.scale_config.transformer_matrix_lr_scale,
                sharding_context=self.sharding_context,
                name="down_proj" + self.suffix_name,
                dot=down_dot,
            )(h_up)

        h_dense = checkpoint_name(h_dense, f"{self.checkpoint_name_prefix}outputs_individual")
        h_dense = with_sharding_constraint_unless_manual(h_dense, input_sharding)
        return h_dense


@chex.dataclass(kw_only=True)
class DecoderOutput:
    output: jax.Array
    attn_loss: float | jax.Array = 0.0
    ffn_loss: float | jax.Array = 0.0
    layer_dumps: tuple[jax.Array] | None = None


@chex.dataclass(kw_only=True)
class FFNLayerOutput:
    output: jax.Array


@dataclass
class FFNLayer(hk.Module):
    ffn_config: FeedForwardConfig
    scale_config: ScaleConfig
    sharding_context: ShardingContext = None
    fp8_gemms: Fp8Gemms = Fp8Gemms.OFF
    fp8_remat_saves: bool = False

    @hk.transparent
    def __call__(
        self,
        h_ln: jax.Array,
        is_training: bool = True,
    ) -> FFNLayerOutput:
        def base_dense_block(
            h,
            widening_factor=self.ffn_config.widening_factor,
        ):
            data_axis = self.sharding_context.logical_axis_to_physical("batch")
            h = DenseBlock(
                config=self.ffn_config,
                scale_config=self.scale_config,
                widening_factor=widening_factor,
                data_axis=data_axis,
                sharding_context=self.sharding_context,
                checkpoint_name_prefix="dense_",
                fp8_gemms=self.fp8_gemms,
                fp8_remat_saves=self.fp8_remat_saves,
            )(h, is_training)
            return h

        h_dense = base_dense_block(h_ln, widening_factor=self.ffn_config.widening_factor)

        h_dense = checkpoint_name(h_dense, "dense_outputs")

        return FFNLayerOutput(
            output=h_dense,
        )


@dataclass
class DecoderLayer(hk.Module):
    attn_config: AttentionConfig
    ffn_config: FeedForwardConfig
    scale_config: ScaleConfig
    pre_norm: bool = True
    primer_norm: bool = False
    sharding_context: ShardingContext = None
    name: Optional[str] = None
    debug_tensor_dump_output_folder: Optional[str] = None
    debug_tensor_dump_n_tokens: Optional[int] = None
    layer_norm_eps: float = 1e-5
    num_layers: int = None
    pre_attn_norm_name: str = "pre_attn_norm"
    pre_ffn_norm_name: str = "pre_ffn_norm"
    fp8_gemms: Fp8Gemms = Fp8Gemms.OFF
    fp8_remat_saves: bool = False

    def __call__(
        self,
        inputs: DecoderInput,
        mask: jax.Array,
        segment_ids: jax.Array,
        segment_ids_k: Optional[jax.Array] = None,
        is_training: bool = True,
        global_layer_index: Optional[jax.Array] = None,
        positions: jax.Array | None = None,
        seqpack_layout: SequencePackedLayout | None = None,
        remat_fn: Optional[Callable[[Callable], Callable]] = None,
        padding_mask: Optional[jax.Array] = None,
        history_states: jax.Array | None = None,
    ) -> DecoderOutput:
        layer_norm = partial(
            RMSNorm,
            axis=-1,
            lr_multiplier=self.scale_config.ln_lr_multiplier(),
            eps=self.layer_norm_eps,
            weight_decay_mask=self.scale_config.ln_weight_decay_mask,
        )
        if remat_fn is None:
            remat_fn = remat_less
        norm_stats_name = NORM_STATS if self.fp8_remat_saves else None

        assert inputs.ndim == 3, f"Expected 3D array but got shape {inputs.shape=}"
        activation_pspec = self.sharding_context.sharding_specs(
            NamedShape(inputs.shape, names=("batch", "sequence", "replicated"))
        )
        residual = with_sharding_constraint_unless_manual(inputs, activation_pspec)

        h_attn_in = residual
        residual = with_sharding_constraint_unless_manual(residual, activation_pspec)

        def run_attn_layer(h_attn_in: jax.Array, history_states: jax.Array | None):
            if self.pre_norm:
                pre_attn_norm = layer_norm(name=self.pre_attn_norm_name)
                h_ln = pre_attn_norm(h_attn_in, stats_name=norm_stats_name)
                if history_states is not None:
                    history_states = pre_attn_norm(history_states)
            else:
                h_ln = h_attn_in
            h_ln = with_sharding_constraint_unless_manual(h_ln, activation_pspec)

            attn_input = h_ln
            attn_outputs: MultiHeadAttentionOutput = MHABlock(
                config=self.attn_config,
                scale_config=self.scale_config,
                sharding_context=self.sharding_context,
                fp8_gemms=self.fp8_gemms,
                fp8_remat_saves=self.fp8_remat_saves,
            )(
                h_ln,
                mask,
                segment_ids,
                segment_ids_k=segment_ids_k,
                positions=positions,
                seqpack_layout=seqpack_layout,
                history_states=history_states,
            )
            checkpoint_fn = checkpoint_as_bits if self.fp8_remat_saves else checkpoint_name
            h_attn = checkpoint_fn(attn_outputs.output, "attn_outputs")

            with jax.named_scope("attn_loss"), set_xla_metadata(_xla_collective_group="metrics"):
                if padding_mask is None:
                    attn_loss = jnp.mean(jnp.mean(h_attn**2, axis=-1))
                else:
                    attn_loss = jnp.mean(jnp.mean(h_attn**2, axis=-1) * padding_mask)
                attn_loss = checkpoint_name(attn_loss, "scalar_stats")

            if self.primer_norm:
                h_attn = with_sharding_constraint_unless_manual(h_attn, activation_pspec)
                h_attn = layer_norm(name="post_attn_norm")(h_attn)

            return h_attn, attn_input, attn_outputs, attn_loss

        h_attn, attn_input, attn_outputs, attn_loss = remat_fn(run_attn_layer)(
            h_attn_in, history_states
        )

        residual = residual + h_attn
        if self.fp8_remat_saves and gpu.bf16_arithmetic_is_native(jax.devices()[0]):
            residual = checkpoint_as_bits(residual, POST_ATTN_RESIDUAL, barrier=False)
        residual = with_sharding_constraint_unless_manual(residual, activation_pspec)

        attn_post_ln_output = residual

        h_ffn_in = residual
        residual = with_sharding_constraint_unless_manual(residual, activation_pspec)

        def run_ffn_layer(h_ffn_in: jax.Array):
            if self.pre_norm:
                h_ln = layer_norm(name=self.pre_ffn_norm_name)(h_ffn_in, stats_name=norm_stats_name)
            else:
                h_ln = h_ffn_in
            h_ln = with_sharding_constraint_unless_manual(h_ln, activation_pspec)

            ffn_input = h_ln
            ffn_result: FFNLayerOutput = FFNLayer(
                ffn_config=self.ffn_config,
                scale_config=self.scale_config,
                sharding_context=self.sharding_context,
                fp8_gemms=self.fp8_gemms,
                fp8_remat_saves=self.fp8_remat_saves,
            )(
                h_ln=h_ln,
                is_training=is_training,
            )
            h_ffn = ffn_result.output
            ffn_output = h_ffn

            with jax.named_scope("ffn_loss"), set_xla_metadata(_xla_collective_group="loss"):
                if padding_mask is None:
                    ffn_loss = jnp.mean(jnp.mean(h_ffn**2, axis=-1))
                else:
                    ffn_loss = jnp.mean(jnp.mean(h_ffn**2, axis=-1) * padding_mask)
                ffn_loss = checkpoint_name(ffn_loss, "scalar_stats")

            if self.primer_norm:
                h_ffn = with_sharding_constraint_unless_manual(h_ffn, activation_pspec)
                h_ffn = layer_norm(name="post_ffn_norm")(h_ffn)

            return h_ffn, ffn_input, ffn_output, ffn_loss

        h_ffn, ffn_input, ffn_output, ffn_loss = remat_fn(run_ffn_layer)(h_ffn_in)

        residual = residual + h_ffn
        residual = with_sharding_constraint_unless_manual(residual, activation_pspec)

        ffn_post_ln_output = residual

        if self.debug_tensor_dump_output_folder:
            layer_dumps = (
                attn_input,
                attn_outputs.query_heads,
                attn_outputs.key_heads,
                attn_outputs.value_heads,
                attn_outputs.attn_output,
                attn_outputs.output,
                attn_post_ln_output,
                ffn_input,
                ffn_output,
                ffn_post_ln_output,
            )

            slice_tokens = lambda x: (
                x[:, : self.debug_tensor_dump_n_tokens] if x is not None else None
            )
            layer_dumps = jax.tree.map(slice_tokens, layer_dumps)
        else:
            layer_dumps = (0,)

        return DecoderOutput(
            output=residual,
            attn_loss=attn_loss,
            ffn_loss=ffn_loss,
            layer_dumps=layer_dumps,
        )


def layer_stack_block(
    h: jax.Array,
    global_layer_index: jax.Array | None = None,
    layer_index_offset: int = 0,
    block: Callable[..., DecoderOutput] | None = None,
    mask: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    segment_ids_k: jax.Array | None = None,
    padding_mask: jax.Array | None = None,
    debug_tensor_dump_output_folder: str | None = None,
    seqpack_layout: SequencePackedLayout | None = None,
    name_prefix: str = "decoder_layer",
) -> tuple[jax.Array, DecoderOutput]:
    if global_layer_index is None:
        global_layer_index = jnp.zeros([], dtype=int)

    if debug_tensor_dump_output_folder is not None:
        layer_dumps = []
    else:
        layer_dumps = None

    d = block(
        h,
        mask=mask,
        segment_ids=segment_ids,
        segment_ids_k=segment_ids_k,
        padding_mask=padding_mask,
        name=f"{name_prefix}_0",
        global_layer_index=global_layer_index + layer_index_offset,
        seqpack_layout=seqpack_layout,
    )
    h = d.output
    attn_loss = jnp.stack([d.attn_loss], axis=0)
    ffn_loss = jnp.stack([d.ffn_loss], axis=0)
    if debug_tensor_dump_output_folder is not None:
        layer_dumps.append(d.layer_dumps)

    if debug_tensor_dump_output_folder is not None:
        num_elements = len(layer_dumps[0])

        concatenated_dumps = []
        for i in range(num_elements):
            dumps = [dump[i] for dump in layer_dumps]
            if any(dump is None for dump in dumps):
                non_none_dump = next((dump for dump in dumps if dump is not None), None)
                if non_none_dump is None:
                    raise ValueError("All dumps are None")
                dummy_tensor = jnp.zeros_like(non_none_dump)
                dumps = [dump if dump is not None else dummy_tensor for dump in dumps]
            concatenated_dumps.append(jnp.stack(dumps, axis=0))
        layer_dumps = tuple(concatenated_dumps)

    return h, DecoderOutput(
        output=jnp.zeros(()),
        attn_loss=attn_loss,
        ffn_loss=ffn_loss,
        layer_dumps=layer_dumps,
    )


@dataclass
class Transformer(hk.Module):
    config: TransformerConfig
    sharding_context: ShardingContext
    name: Optional[str] = None

    summarizer_prefix: str = ""

    @property
    def data_axis(self):
        return self.config.data_axis

    def __call__(
        self,
        embeddings: jax.Array,
        mask: jax.Array,
        segment_ids: Optional[jax.Array] = None,
        segment_ids_k: Optional[jax.Array] = None,
        *,
        is_training: bool = True,
        decoding: bool = False,
        padding_mask: jax.Array | None = None,
        positions: jax.Array | None = None,
        seqpack_layout: SequencePackedLayout | None = None,
        subset_index: jax.Array | None = None,
        act_l2_mask: jax.Array | None = None,
        candidate_start: int | None = None,
    ) -> TransformerModelOutput:
        config = self.config

        assert not decoding, "KV-cache decoding is not supported by this transformer stack."

        if padding_mask is None:
            padding_mask = mask.copy()

        B, T, H = embeddings.shape
        del subset_index

        if len(mask.shape) != 4:
            mask = mask[:, None, None, :]
        else:
            assert mask.shape == (B, 1, T, T), (
                f"Since mask is rank 4, expected shape {(B, 1, T, T)}, but got {mask.shape} instead"
            )

        hidden_states = jnp.zeros((B, T, H), dtype=jnp.bfloat16) + embeddings

        assert config.remat_boundary in ("block", "layer"), (
            f"Invalid remat boundary: {config.remat_boundary}"
        )
        block_remat_fn = partial(
            hk.remat if config.use_remat and config.remat_boundary == "block" else remat_less,
            static_argnums=(4, 8),
            policy=custom_remat_policy(config.remat_policy),
            prevent_cse=config.prevent_cse,
        )
        layer_remat_fn = partial(
            hk.remat if config.use_remat and config.remat_boundary == "layer" else remat_less,
            policy=custom_remat_policy(config.remat_policy),
            prevent_cse=config.prevent_cse,
        )
        fp8_remat_saves = config.fp8_gemms == Fp8Gemms.ALL and (
            config.use_remat and config.remat_policy == RematType.SAVE_H100_RECSYS
        )

        def run_decoder_layer(
            hidden_states,
            mask,
            segment_ids,
            segment_ids_k: Optional[jax.Array] = None,
            padding_mask: Optional[jax.Array] = None,
            name: Optional[str] = None,
            global_layer_index: Optional[jax.Array] = None,
            seqpack_layout: SequencePackedLayout | None = None,
            history_states: jax.Array | None = None,
            fp8_gemms: Fp8Gemms = config.fp8_gemms,
            positions: jax.Array | None = positions,
        ) -> DecoderOutput:
            return block_remat_fn(
                DecoderLayer(
                    pre_norm=config.pre_norm,
                    primer_norm=config.primer_norm,
                    sharding_context=self.sharding_context,
                    attn_config=config.attn_config,
                    ffn_config=config.ffn_config,
                    scale_config=config.scale_config,
                    name=name,
                    debug_tensor_dump_output_folder=self.config.debug_tensor_dump_output_folder,
                    debug_tensor_dump_n_tokens=self.config.debug_tensor_dump_n_tokens,
                    layer_norm_eps=config.layer_norm_eps,
                    num_layers=config.num_layers,
                    fp8_gemms=fp8_gemms,
                    fp8_remat_saves=fp8_remat_saves,
                )
            )(
                hidden_states,
                mask,
                segment_ids,
                segment_ids_k,
                is_training,
                global_layer_index,
                positions,
                seqpack_layout,
                layer_remat_fn,
                padding_mask,
                history_states,
            )

        num_candidate_only_layers = config.num_candidate_only_layers
        candidate_only_attn_impl = (
            config.attn_config.candidate_only_attn_impl if config.attn_config else None
        )
        assert 0 <= num_candidate_only_layers <= config.num_layers
        assert num_candidate_only_layers or candidate_only_attn_impl is None

        if num_candidate_only_layers:
            assert config.use_layer_stack, "Candidate-only layers need use_layer_stack=True"
            assert not config.debug_tensor_dump_output_folder, "No dumps in candidate-only layers"

            is_candidate = segment_ids == -1
            use_candidate_only_kernel = candidate_only_attn_impl is not None
            if use_candidate_only_kernel and candidate_start is None:
                raise ValueError(
                    "candidate_only_attn_impl with use_seqpack needs seqpack_candidate_packing. "
                    "Otherwise set candidate_only_attn_impl=None."
                )

            update_user_token = config.candidate_only_layers_update_user_token
            if update_user_token:
                if seqpack_layout is not None:
                    user_positions = jnp.asarray(
                        packed_prefix_positions(seqpack_layout), dtype=jnp.int32
                    )
                elif use_candidate_only_kernel:
                    raise ValueError(
                        "candidate_only_layers_update_user_token with candidate_only_attn_impl "
                        "needs seqpack_candidate_packing."
                    )
                else:
                    user_positions = jnp.zeros((B, 1), dtype=jnp.int32)
                rows = jnp.arange(B)[:, None]
                is_candidate = is_candidate | jnp.zeros((B, T), bool).at[rows, user_positions].set(
                    True
                )

                def take_user_rows(x: jax.Array) -> jax.Array:
                    return jnp.take_along_axis(x, user_positions[:, :, None], axis=1)

            def split_history_and_candidates(
                hidden_states: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                if use_candidate_only_kernel and update_user_token:
                    history = hidden_states[:, :candidate_start]
                    candidates = hidden_states[:, candidate_start:]
                    return history, jnp.concatenate([candidates, take_user_rows(history)], axis=1)
                if use_candidate_only_kernel:
                    return hidden_states[:, :candidate_start], hidden_states[:, candidate_start:]
                return hidden_states, hidden_states

            def run_candidate_only_layer(
                candidate_states: jax.Array, *, history_states: jax.Array, **layer_kwargs
            ) -> DecoderOutput:
                run_layer = partial(run_decoder_layer, fp8_gemms=Fp8Gemms.OFF, **layer_kwargs)
                if use_candidate_only_kernel:
                    if update_user_token and positions is not None:
                        user_token_positions = take_user_rows(positions)
                        run_layer = partial(
                            run_layer,
                            positions=jnp.concatenate([positions, user_token_positions], axis=1),
                        )
                    return run_layer(candidate_states, history_states=history_states)

                decoder_output = run_layer(candidate_states)
                return replace(
                    decoder_output,
                    output=jnp.where(
                        is_candidate[..., None], decoder_output.output, history_states
                    ),
                )

            if use_candidate_only_kernel:
                num_candidates = T - candidate_start
                candidate_only_stats_mask = padding_mask[:, candidate_start:] * (num_candidates / T)
                if update_user_token:
                    user_mask = take_user_rows(padding_mask[..., None])[..., 0]
                    row_mask = jnp.concatenate(
                        [padding_mask[:, candidate_start:], user_mask], axis=1
                    )
                    candidate_only_stats_mask = row_mask * (row_mask.shape[1] / T)
            else:
                candidate_only_stats_mask = padding_mask * is_candidate

        if not config.use_layer_stack:
            attn_losses, ffn_losses = [], []
            for i in range(config.num_layers):
                decoder_output = run_decoder_layer(
                    hidden_states,
                    mask,
                    segment_ids,
                    segment_ids_k,
                    padding_mask=padding_mask,
                    name=f"decoder_layer_{i}",
                    global_layer_index=i,
                    seqpack_layout=seqpack_layout,
                )
                hidden_states = decoder_output.output
                attn_losses.append(decoder_output.attn_loss)
                ffn_losses.append(decoder_output.ffn_loss)
            decoder_output = replace(
                decoder_output, attn_loss=jnp.stack(attn_losses), ffn_loss=jnp.stack(ffn_losses)
            )

        else:
            num_main_layers = config.num_layers

            main_layer_block = partial(
                layer_stack_block,
                block=run_decoder_layer,
                mask=mask,
                segment_ids=segment_ids,
                segment_ids_k=segment_ids_k,
                padding_mask=padding_mask,
                layer_index_offset=1,
                debug_tensor_dump_output_folder=self.config.debug_tensor_dump_output_folder,
                seqpack_layout=seqpack_layout,
            )

            candidate_only_tail = None
            if config.num_candidate_only_layers:

                def candidate_only_stack_layer(candidate_states, *, tail_input, **stack_kwargs):
                    return main_layer_block(
                        candidate_states,
                        block=partial(run_candidate_only_layer, history_states=tail_input),
                        padding_mask=candidate_only_stats_mask,
                        **stack_kwargs,
                    )

                candidate_only_tail = layer_stack.LayerStackTail(
                    num_layers=config.num_candidate_only_layers,
                    layer_fn=candidate_only_stack_layer,
                    split_inputs=split_history_and_candidates,
                )
            main_stack = layer_stack.layer_stack(
                num_main_layers,
                with_per_layer_inputs=True,
                unroll=config.unroll_layer_stack,
                scan_impl=jax.lax.scan,
                tail=candidate_only_tail,
            )
            hidden_states, decoder_output = main_stack(main_layer_block)(hidden_states)
            if config.num_candidate_only_layers:
                history_states, candidate_states = hidden_states

                if use_candidate_only_kernel:
                    if update_user_token:
                        candidate_states, user_states = jnp.split(
                            candidate_states, [T - candidate_start], axis=1
                        )
                        history_states = history_states.at[rows, user_positions].set(user_states)
                    hidden_states = jnp.concatenate([history_states, candidate_states], axis=1)
                else:
                    hidden_states = candidate_states

            decoder_output = replace(
                decoder_output,
                output=hidden_states,
                attn_loss=decoder_output.attn_loss.flatten(),
                ffn_loss=decoder_output.ffn_loss.flatten(),
            )

            if self.config.debug_tensor_dump_output_folder:
                layer_dumps = unroll_layer_dumps(decoder_output.layer_dumps)

                total_num_layers = config.num_layers
                dump_block_outputs(
                    self.config,
                    layer_dumps,
                    total_num_layers,
                    self.config.debug_tensor_dump_n_layers,
                )

        final_hidden_states = decoder_output.output

        history_packed_token_scale = 1.0
        if seqpack_layout is not None and seqpack_layout.candidate_token_to_slot is not None:
            num_tokens = final_hidden_states.shape[1]
            history_packed_num_tokens = (
                num_tokens
                - seqpack_layout.candidate_token_to_slot.shape[-1]
                + seqpack_layout.candidate_positions.shape[-1]
            )
            history_packed_token_scale = jnp.float32(num_tokens / history_packed_num_tokens)

        summarize(
            f"{self.summarizer_prefix}attn-loss",
            jnp.sum(decoder_output.attn_loss) * history_packed_token_scale,
        )
        summarize(
            f"{self.summarizer_prefix}ffn-loss",
            jnp.sum(decoder_output.ffn_loss) * history_packed_token_scale,
        )

        act_l2_mask = padding_mask if act_l2_mask is None else act_l2_mask
        if config.num_candidate_only_layers:
            act_l2_mask = act_l2_mask * is_candidate
        summarize(
            f"{self.summarizer_prefix}act-l2-loss",
            jnp.mean(jnp.mean(final_hidden_states**2, axis=-1) * act_l2_mask)
            * history_packed_token_scale,
        )

        return TransformerModelOutput(output=final_hidden_states)
