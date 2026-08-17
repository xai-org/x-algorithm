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
from jax.lax import with_sharding_constraint
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from xai_configlib import Config, configclass
from xrex.data.recsys.sequence_packing import SequencePackedLayout
from xrex.models.attention import AttentionConfig
from xrex.models.layers import (
    Linear,
    MultiHeadAttention,
    MultiHeadAttentionOutput,
)
from xrex.models.normalization import rms_norm_fn
from xrex.models.remat import (
    RematType,
    custom_remat_policy,
)
from xrex.models.scaling import ScaleConfig
from xrex.models.sharding_context import NamedShape, ShardingContext
from xrex.utils import layer_stack
from xrex.utils.gpu import peak_tflops
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

    use_remat: bool = True
    remat_policy: RematType = RematType.WHOLE
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

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        segment_ids: jax.Array,
        segment_ids_k: Optional[jax.Array] = None,
        positions: jax.Array | None = None,
        seqpack_layout: SequencePackedLayout | None = None,
    ) -> MultiHeadAttentionOutput:
        _, _, model_size = inputs.shape
        initializer = hk.initializers.VarianceScaling(self.scale_config.attn_init_scale**2)
        assert mask.ndim == 4, f"shape: {mask.shape}"
        assert mask.shape[2] in {1, inputs.shape[1]}, str(mask.shape)
        assert mask.shape[3] in {1, inputs.shape[1]}, str(mask.shape)

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
        inputs = with_sharding_constraint(
            inputs,
            activation_pspecs,
        )
        attn_output = attn_block(inputs, inputs, inputs, mask, segment_ids, segment_ids_k)
        attn_output = replace(
            attn_output,
            output=with_sharding_constraint(attn_output.output, activation_pspecs),
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
        inputs = with_sharding_constraint(inputs, input_sharding)

        weights_pspec = P(None, None)
        activation_spec = P(
            self.data_axis,
            self.sharding_context.logical_axis_to_physical("dense_activation_seq"),
            self.sharding_context.logical_axis_to_physical("dense_activation_model"),
        )
        activation_sharding = NamedSharding(self.sharding_context.mesh, activation_spec)

        if self.config.gated_mlp:
            if self.config.merge_gate_up:
                h = Linear(
                    ffn_size_ * 2,
                    w_init=initializer,
                    with_bias=self.config.with_bias,
                    sharding_context=self.sharding_context,
                    pspec=weights_pspec,
                    lr_multiplier=self.scale_config.hidden_lr_multiplier(model_size),
                    init_scale=self.scale_config.ffn_init_scale,
                    name="gate_up_proj" + self.suffix_name,
                )(inputs)

                h = jnp.reshape(h, (batch_size, seq_len, -1, 2))
                h = with_sharding_constraint(h, activation_sharding)
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
                    lr_multiplier=self.scale_config.hidden_lr_multiplier(model_size),
                    init_scale=self.scale_config.ffn_init_scale,
                    name="up_proj" + self.suffix_name,
                )(inputs)
                h_gate = Linear(
                    ffn_size_,
                    w_init=initializer,
                    with_bias=self.config.with_bias,
                    sharding_context=self.sharding_context,
                    pspec=weights_pspec,
                    lr_multiplier=self.scale_config.hidden_lr_multiplier(model_size),
                    init_scale=self.scale_config.ffn_init_scale,
                    name="gate_proj" + self.suffix_name,
                )(inputs)
                h_up = checkpoint_name(h_up, f"{self.checkpoint_name_prefix}up_proj")
                h_gate = checkpoint_name(h_gate, f"{self.checkpoint_name_prefix}gate_proj")
            combined = h_up * gelu_fn(h_gate, approximate=True)
            combined = with_sharding_constraint(combined, activation_sharding)

            h_dense = Linear(
                model_size,
                w_init=initializer,
                with_bias=self.config.with_bias,
                pspec=weights_pspec,
                lr_multiplier=self.scale_config.ffn_output_lr_multiplier(ffn_size_, gated_mlp=True),
                init_scale=self.scale_config.ffn_init_scale,
                sharding_context=self.sharding_context,
                name="down_proj" + self.suffix_name,
            )(combined)
        else:
            h_up = Linear(
                ffn_size_,
                w_init=initializer,
                with_bias=self.config.with_bias,
                sharding_context=self.sharding_context,
                pspec=weights_pspec,
                lr_multiplier=self.scale_config.hidden_lr_multiplier(model_size),
                name="up_proj" + self.suffix_name,
            )(inputs)
            h_up = checkpoint_name(h_up, f"{self.checkpoint_name_prefix}up_proj")
            h_up = gelu_fn(h_up, approximate=True)
            h_dense = Linear(
                model_size,
                w_init=initializer,
                with_bias=self.config.with_bias,
                pspec=weights_pspec,
                lr_multiplier=self.scale_config.ffn_output_lr_multiplier(ffn_size_, gated_mlp=True),
                sharding_context=self.sharding_context,
                name="down_proj" + self.suffix_name,
            )(h_up)

        h_dense = checkpoint_name(h_dense, f"{self.checkpoint_name_prefix}outputs_individual")
        h_dense = with_sharding_constraint(h_dense, input_sharding)
        return h_dense


@chex.dataclass(kw_only=True)
class DecoderOutput:
    output: jax.Array
    layer_dumps: tuple[jax.Array] | None = None


@chex.dataclass(kw_only=True)
class FFNLayerOutput:
    output: jax.Array


@dataclass
class FFNLayer(hk.Module):
    ffn_config: FeedForwardConfig
    scale_config: ScaleConfig
    sharding_context: ShardingContext = None

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
    ) -> DecoderOutput:
        layer_norm = partial(
            rms_norm_fn,
            lr_multiplier=self.scale_config.ln_lr_multiplier(),
            eps=self.layer_norm_eps,
            weight_decay_mask=self.scale_config.ln_weight_decay_mask,
        )
        if remat_fn is None:
            remat_fn = remat_less

        assert inputs.ndim == 3, f"Expected 3D array but got shape {inputs.shape=}"
        activation_pspec = self.sharding_context.sharding_specs(
            NamedShape(inputs.shape, names=("batch", "sequence", "replicated"))
        )
        residual = with_sharding_constraint(inputs, activation_pspec)

        h_attn_in = residual
        residual = with_sharding_constraint(residual, activation_pspec)

        def run_attn_layer(h_attn_in: jax.Array):
            if self.pre_norm:
                h_ln = layer_norm(h_attn_in, self.pre_attn_norm_name)
            else:
                h_ln = h_attn_in
            h_ln = with_sharding_constraint(h_ln, activation_pspec)

            attn_input = h_ln
            attn_outputs: MultiHeadAttentionOutput = MHABlock(
                config=self.attn_config,
                scale_config=self.scale_config,
                sharding_context=self.sharding_context,
            )(
                h_ln,
                mask,
                segment_ids,
                segment_ids_k=segment_ids_k,
                positions=positions,
                seqpack_layout=seqpack_layout,
            )
            h_attn = checkpoint_name(attn_outputs.output, "attn_outputs")

            if self.primer_norm:
                h_attn = with_sharding_constraint(h_attn, activation_pspec)
                h_attn = layer_norm(h_attn, "post_attn_norm")

            return h_attn, attn_input, attn_outputs

        h_attn, attn_input, attn_outputs = remat_fn(run_attn_layer)(h_attn_in)

        residual = residual + h_attn
        residual = with_sharding_constraint(residual, activation_pspec)

        attn_post_ln_output = residual

        h_ffn_in = residual
        residual = with_sharding_constraint(residual, activation_pspec)

        def run_ffn_layer(h_ffn_in: jax.Array):
            if self.pre_norm:
                h_ln = layer_norm(h_ffn_in, self.pre_ffn_norm_name)
            else:
                h_ln = h_ffn_in
            h_ln = with_sharding_constraint(h_ln, activation_pspec)

            ffn_input = h_ln
            ffn_result: FFNLayerOutput = FFNLayer(
                ffn_config=self.ffn_config,
                scale_config=self.scale_config,
                sharding_context=self.sharding_context,
            )(
                h_ln=h_ln,
                is_training=is_training,
            )
            h_ffn = ffn_result.output
            ffn_output = h_ffn

            if self.primer_norm:
                h_ffn = with_sharding_constraint(h_ffn, activation_pspec)
                h_ffn = layer_norm(h_ffn, "post_ffn_norm")

            return h_ffn, ffn_input, ffn_output

        h_ffn, ffn_input, ffn_output = remat_fn(run_ffn_layer)(h_ffn_in)

        residual = residual + h_ffn
        residual = with_sharding_constraint(residual, activation_pspec)

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
        name=f"{name_prefix}_0",
        global_layer_index=global_layer_index + layer_index_offset,
        seqpack_layout=seqpack_layout,
    )
    h = d.output
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

        h = jnp.zeros((B, T, H), dtype=jnp.bfloat16) + embeddings

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

        def block(
            h,
            mask,
            segment_ids,
            segment_ids_k: Optional[jax.Array] = None,
            name: Optional[str] = None,
            global_layer_index: Optional[jax.Array] = None,
            seqpack_layout: SequencePackedLayout | None = None,
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
                )
            )(
                h,
                mask,
                segment_ids,
                segment_ids_k,
                is_training,
                global_layer_index,
                positions,
                seqpack_layout,
                layer_remat_fn,
            )

        if not config.use_layer_stack:
            for i in range(config.num_layers):
                d = block(
                    h,
                    mask,
                    segment_ids,
                    segment_ids_k,
                    name=f"decoder_layer_{i}",
                    global_layer_index=i,
                    seqpack_layout=seqpack_layout,
                )
                h = d.output

        else:
            num_main_layers = config.num_layers

            main_stack = layer_stack.layer_stack(
                num_main_layers,
                with_per_layer_inputs=True,
                unroll=config.unroll_layer_stack,
                scan_impl=jax.lax.scan,
            )
            main_layer_block = partial(
                layer_stack_block,
                block=block,
                mask=mask,
                segment_ids=segment_ids,
                segment_ids_k=segment_ids_k,
                layer_index_offset=1,
                debug_tensor_dump_output_folder=self.config.debug_tensor_dump_output_folder,
                seqpack_layout=seqpack_layout,
            )
            h, d = main_stack(main_layer_block)(h)
            d = replace(d, output=h)

            if self.config.debug_tensor_dump_output_folder:
                layer_dumps = unroll_layer_dumps(d.layer_dumps)

                total_num_layers = config.num_layers
                dump_block_outputs(
                    self.config,
                    layer_dumps,
                    total_num_layers,
                    self.config.debug_tensor_dump_n_layers,
                )

        h_final = d.output

        summarize(
            f"{self.summarizer_prefix}act-l2-loss",
            jnp.mean(jnp.mean(h_final**2, axis=-1) * padding_mask),
        )

        return TransformerModelOutput(output=h_final)
