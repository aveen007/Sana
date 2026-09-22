"""Read-only diagnostics for SANA text conditioning.

The debugger calls the model's existing caption projection, normalization,
cross-attention K/V projections, and denoiser forward methods.  It does not
replace modules or alter inference outputs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _as_token_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return conditioning as [batch, tokens, channels]."""
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(
            "Expected conditioning shaped [B, L, C] or [B, 1, L, C], "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


def _as_mask(mask: torch.Tensor, batch: int, tokens: int) -> torch.Tensor:
    while mask.ndim > 2 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.shape != (batch, tokens):
        raise ValueError(f"Expected mask shape {(batch, tokens)}, got {tuple(mask.shape)}")
    return mask.bool()


def _compare_tensors(
    native: torch.Tensor,
    mapped: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    include_per_token: bool = False,
) -> dict[str, Any]:
    if native.shape != mapped.shape:
        raise ValueError(
            f"Cannot compare different shapes: native={tuple(native.shape)}, "
            f"mapped={tuple(mapped.shape)}"
        )
    native = native.detach().float()
    mapped = mapped.detach().float()
    original_shape = list(native.shape)

    if mask is not None:
        if native.ndim != 3 or mask.shape != native.shape[:2]:
            raise ValueError(
                f"Mask {tuple(mask.shape)} does not match token tensor {tuple(native.shape)}"
            )
        native_vectors = native[mask]
        mapped_vectors = mapped[mask]
    elif native.ndim == 1:
        native_vectors = native.reshape(1, -1)
        mapped_vectors = mapped.reshape(1, -1)
    else:
        native_vectors = native.reshape(-1, native.shape[-1])
        mapped_vectors = mapped.reshape(-1, mapped.shape[-1])

    if native_vectors.numel() == 0:
        raise ValueError("No vectors remain after applying the comparison mask")
    if not torch.isfinite(native_vectors).all() or not torch.isfinite(mapped_vectors).all():
        raise ValueError("Conditioning comparison contains NaN or infinite values")

    difference = native_vectors - mapped_vectors
    native_norms = native_vectors.norm(dim=-1)
    mapped_norms = mapped_vectors.norm(dim=-1)
    cosines = F.cosine_similarity(native_vectors, mapped_vectors, dim=-1, eps=1e-12)
    l2 = difference.norm(dim=-1)
    native_norm = native_norms.mean()
    mapped_norm = mapped_norms.mean()
    global_native_norm = native_vectors.norm()
    global_l2 = difference.norm()
    result: dict[str, Any] = {
        "tensor_shape": original_shape,
        "compared_vectors": int(native_vectors.shape[0]),
        "cosine_similarity": float(cosines.mean()),
        "minimum_vector_cosine": float(cosines.min()),
        "maximum_vector_cosine": float(cosines.max()),
        "mean_l2_distance": float(l2.mean()),
        "global_l2_distance": float(global_l2),
        "relative_l2_error": float(global_l2 / global_native_norm.clamp_min(1e-12)),
        "native_norm": float(native_norm),
        "mapped_norm": float(mapped_norm),
        "norm_ratio": float(mapped_norm / native_norm.clamp_min(1e-12)),
        "relative_norm_difference": float(
            (mapped_norm - native_norm).abs() / native_norm.clamp_min(1e-12)
        ),
        "mean_absolute_error": float(difference.abs().mean()),
        "maximum_absolute_error": float(difference.abs().max()),
    }
    if include_per_token:
        result["per_vector_cosine"] = cosines.cpu().tolist()
    return result


def _sign_metrics(
    native: torch.Tensor,
    mapped: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Measure sign agreement, weighting serious sign errors by native magnitude."""
    if native.shape != mapped.shape:
        raise ValueError("Sign comparison requires tensors with identical shapes")
    native = native.detach().float()
    mapped = mapped.detach().float()
    if mask is not None:
        if native.ndim != 3 or mask.shape != native.shape[:2]:
            raise ValueError("Sign-comparison mask does not match token tensor")
        native = native[mask]
        mapped = mapped[mask]
    native = native.reshape(-1)
    mapped = mapped.reshape(-1)
    if native.numel() == 0:
        raise ValueError("No coordinates remain for sign comparison")

    native_negative = torch.signbit(native)
    mapped_negative = torch.signbit(mapped)
    agreement = native_negative == mapped_negative
    disagreement = ~agreement
    positive_to_negative = (~native_negative) & mapped_negative
    negative_to_positive = native_negative & (~mapped_negative)
    native_magnitude = native.abs()
    magnitude_total = native_magnitude.sum().clamp_min(1e-12)
    weighted_error = (native_magnitude * disagreement).sum() / magnitude_total

    threshold_agreement: dict[str, Any] = {}
    for threshold in (0.01, 0.05, 0.1, 0.5, 1.0):
        selected = native_magnitude > threshold
        threshold_agreement[f"{threshold:g}"] = {
            "selected_coordinates": int(selected.sum()),
            "agreement_fraction": (
                float(agreement[selected].float().mean()) if selected.any() else None
            ),
        }

    return {
        "total_coordinates": int(native.numel()),
        "sign_agreement_fraction": float(agreement.float().mean()),
        "sign_disagreement_fraction": float(disagreement.float().mean()),
        "native_positive_mapped_negative_fraction": float(
            positive_to_negative.float().mean()
        ),
        "native_negative_mapped_positive_fraction": float(
            negative_to_positive.float().mean()
        ),
        "magnitude_weighted_sign_error": float(weighted_error),
        "magnitude_weighted_sign_agreement": float(1.0 - weighted_error),
        "threshold_sign_agreement": threshold_agreement,
    }


def _gelu_analysis(
    native_before: torch.Tensor,
    mapped_before: torch.Tensor,
    native_after: torch.Tensor,
    mapped_after: torch.Tensor,
    *,
    mask: torch.Tensor,
    top_n: int,
) -> dict[str, Any]:
    """Describe which fc1-coordinate errors GELU suppresses or amplifies."""
    before = _compare_tensors(native_before, mapped_before, mask=mask)
    after = _compare_tensors(native_after, mapped_after, mask=mask)
    signs = _sign_metrics(native_before, mapped_before, mask=mask)
    active_positions = torch.nonzero(mask, as_tuple=False).cpu()
    native_before_active = native_before.detach().float()[mask].cpu()
    mapped_before_active = mapped_before.detach().float()[mask].cpu()
    native_after_active = native_after.detach().float()[mask].cpu()
    mapped_after_active = mapped_after.detach().float()[mask].cpu()
    before_error = (native_before_active - mapped_before_active).abs()
    after_error = (native_after_active - mapped_after_active).abs()
    amplified = after_error > before_error

    flat_error = after_error.flatten()
    count = min(max(int(top_n), 0), flat_error.numel())
    top_coordinates: list[dict[str, Any]] = []
    if count:
        for flat_index in torch.topk(flat_error, k=count).indices.tolist():
            active_row = flat_index // after_error.shape[1]
            dimension = flat_index % after_error.shape[1]
            batch_index, token_index = active_positions[active_row].tolist()
            native_pre = native_before_active[active_row, dimension]
            mapped_pre = mapped_before_active[active_row, dimension]
            native_post = native_after_active[active_row, dimension]
            mapped_post = mapped_after_active[active_row, dimension]
            top_coordinates.append(
                {
                    "batch_index": int(batch_index),
                    "token_index": int(token_index),
                    "dimension": int(dimension),
                    "native_before": float(native_pre),
                    "mapped_before": float(mapped_pre),
                    "native_after": float(native_post),
                    "mapped_after": float(mapped_post),
                    "sign_match_before": bool(
                        torch.signbit(native_pre) == torch.signbit(mapped_pre)
                    ),
                    "absolute_error_before": float(before_error[active_row, dimension]),
                    "absolute_error_after": float(after_error[active_row, dimension]),
                }
            )

    return {
        "before": before,
        "after": after,
        "signs_before": signs,
        "mismatch_amplified_fraction": float(amplified.float().mean()),
        "mean_absolute_error_before": float(before_error.mean()),
        "mean_absolute_error_after": float(after_error.mean()),
        "top_post_gelu_error_coordinates": top_coordinates,
    }


def _projection_dimension_analysis(
    native_input: torch.Tensor,
    mapped_input: torch.Tensor,
    *,
    mask: torch.Tensor,
    kv_linear,
    top_n: int,
) -> dict[str, Any]:
    """Rank input dimensions by error and by their possible V-error contribution."""
    native_active = native_input.detach().float()[mask]
    mapped_active = mapped_input.detach().float()[mask]
    delta = native_active - mapped_active
    weight = kv_linear.weight.detach().float()
    output_channels = weight.shape[0] // 2
    if weight.shape[0] != output_channels * 2 or weight.shape[1] != delta.shape[-1]:
        raise ValueError("Unexpected SANA kv_linear dimensions")
    key_weight = weight[:output_channels]
    value_weight = weight[output_channels:]

    input_mean_abs_error = delta.abs().mean(dim=0)
    input_rms_error = delta.square().mean(dim=0).sqrt()
    input_l2_error = delta.norm(dim=0)
    key_weight_l2 = key_weight.norm(dim=0)
    value_weight_l2 = value_weight.norm(dim=0)
    key_contribution_l2 = input_l2_error * key_weight_l2
    value_contribution_l2 = input_l2_error * value_weight_l2

    bias = kv_linear.bias.detach().float() if kv_linear.bias is not None else None
    value_bias = bias[output_channels:] if bias is not None else None
    actual_value_error = F.linear(native_active, value_weight, value_bias) - F.linear(
        mapped_active, value_weight, value_bias
    )
    reconstructed_value_error = F.linear(delta, value_weight, None)
    reconstruction_residual = (actual_value_error - reconstructed_value_error).abs()

    def row(dimension: int) -> dict[str, Any]:
        return {
            "dimension": int(dimension),
            "input_mean_absolute_error": float(input_mean_abs_error[dimension]),
            "input_rms_error": float(input_rms_error[dimension]),
            "input_l2_error": float(input_l2_error[dimension]),
            "wk_column_l2": float(key_weight_l2[dimension]),
            "wv_column_l2": float(value_weight_l2[dimension]),
            "wv_to_wk_weight_ratio": float(
                value_weight_l2[dimension] / key_weight_l2[dimension].clamp_min(1e-12)
            ),
            "estimated_k_contribution_l2": float(key_contribution_l2[dimension]),
            "estimated_v_contribution_l2": float(value_contribution_l2[dimension]),
        }

    count = min(max(int(top_n), 0), delta.shape[-1])
    input_indices = (
        torch.topk(input_l2_error, k=count).indices.tolist() if count else []
    )
    value_indices = (
        torch.topk(value_contribution_l2, k=count).indices.tolist() if count else []
    )
    return {
        "top_by_input_error": [row(index) for index in input_indices],
        "top_by_estimated_v_contribution": [row(index) for index in value_indices],
        "value_error_reconstruction_max_abs_residual": float(
            reconstruction_residual.max()
        ),
        "value_error_reconstruction_mean_abs_residual": float(
            reconstruction_residual.mean()
        ),
    }


def _center_active_tokens(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Remove each sample's mean active-token vector while preserving shape."""
    centered = tensor.clone()
    for batch_index in range(tensor.shape[0]):
        active = mask[batch_index]
        if not active.any():
            raise ValueError("Cannot center a sequence with no active tokens")
        mean = tensor[batch_index, active].mean(dim=0, keepdim=True)
        centered[batch_index] = tensor[batch_index] - mean
    return centered


def _fixed_query_attention_metrics(
    native_query: torch.Tensor,
    native_key: torch.Tensor,
    mapped_key: torch.Tensor,
) -> dict[str, Any]:
    """Compare attention after replacing only K while holding native Q fixed."""
    native_query = native_query.detach().float()
    native_key = native_key.detach().float()
    mapped_key = mapped_key.detach().float()
    if native_key.shape != mapped_key.shape:
        raise ValueError("Fixed-Q comparison requires matching K shapes")
    if native_query.shape[-1] != native_key.shape[-1]:
        raise ValueError("Fixed-Q comparison has incompatible Q/K head dimensions")

    scale = math.sqrt(native_query.shape[-1])
    native_logits = torch.matmul(native_query, native_key.transpose(-2, -1)) / scale
    mapped_logits = torch.matmul(native_query, mapped_key.transpose(-2, -1)) / scale
    native_attention = torch.softmax(native_logits, dim=-1)
    mapped_attention = torch.softmax(mapped_logits, dim=-1)
    comparison = _compare_tensors(native_attention, mapped_attention)

    eps = 1e-12
    midpoint = 0.5 * (native_attention + mapped_attention)
    js = 0.5 * (
        (native_attention * (native_attention.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(-1)
        + (mapped_attention * (mapped_attention.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(-1)
    )
    native_entropy = -(
        native_attention * native_attention.clamp_min(eps).log()
    ).sum(-1)
    mapped_entropy = -(
        mapped_attention * mapped_attention.clamp_min(eps).log()
    ).sum(-1)
    argmax_agreement = native_attention.argmax(-1) == mapped_attention.argmax(-1)
    top_k = min(3, native_attention.shape[-1])
    native_top = native_attention.topk(top_k, dim=-1).indices
    mapped_top = mapped_attention.topk(top_k, dim=-1).indices
    overlap = (
        native_top.unsqueeze(-1) == mapped_top.unsqueeze(-2)
    ).any(dim=-1).float().sum(dim=-1) / max(top_k, 1)

    return {
        "attention": comparison,
        "logits": _compare_tensors(native_logits, mapped_logits),
        "jensen_shannon_divergence": float(js.mean()),
        "argmax_token_agreement_fraction": float(argmax_agreement.float().mean()),
        "top3_token_overlap_fraction": float(overlap.mean()),
        "top_k_used": int(top_k),
        "native_attention_entropy": float(native_entropy.mean()),
        "mapped_attention_entropy": float(mapped_entropy.mean()),
    }


def _per_token_statistics(
    *,
    native_post_gelu: torch.Tensor,
    mapped_post_gelu: torch.Tensor,
    native_post_norm: torch.Tensor,
    mapped_post_norm: torch.Tensor,
    native_pre_gelu: torch.Tensor,
    mapped_pre_gelu: torch.Tensor,
    native_values: list[torch.Tensor],
    mapped_values: list[torch.Tensor],
    mask: torch.Tensor,
    token_labels: list[list[str]] | None,
    worst_count: int,
) -> dict[str, Any]:
    positions = torch.nonzero(mask, as_tuple=False).cpu().tolist()
    tokens: list[dict[str, Any]] = []
    for batch_index, token_index in positions:
        post_gelu_cosine = F.cosine_similarity(
            native_post_gelu[batch_index, token_index].float(),
            mapped_post_gelu[batch_index, token_index].float(),
            dim=0,
            eps=1e-12,
        )
        post_norm_cosine = F.cosine_similarity(
            native_post_norm[batch_index, token_index].float(),
            mapped_post_norm[batch_index, token_index].float(),
            dim=0,
            eps=1e-12,
        )
        token_signs = _sign_metrics(
            native_pre_gelu[batch_index, token_index],
            mapped_pre_gelu[batch_index, token_index],
        )
        value_cosines = [
            float(
                F.cosine_similarity(
                    native_value[batch_index, token_index].float(),
                    mapped_value[batch_index, token_index].float(),
                    dim=0,
                    eps=1e-12,
                )
            )
            for native_value, mapped_value in zip(native_values, mapped_values)
        ]
        worst_block = min(range(len(value_cosines)), key=value_cosines.__getitem__)
        label = None
        if token_labels is not None:
            label = token_labels[batch_index][token_index]
        tokens.append(
            {
                "batch_index": int(batch_index),
                "token_index": int(token_index),
                "token_label": label,
                "post_gelu_cosine": float(post_gelu_cosine),
                "post_rmsnorm_cosine": float(post_norm_cosine),
                "pre_gelu_sign_agreement_fraction": token_signs[
                    "sign_agreement_fraction"
                ],
                "pre_gelu_magnitude_weighted_sign_error": token_signs[
                    "magnitude_weighted_sign_error"
                ],
                "mean_value_cosine": float(sum(value_cosines) / len(value_cosines)),
                "minimum_value_cosine": float(value_cosines[worst_block]),
                "minimum_value_cosine_block": int(worst_block),
                "value_cosine_by_block": value_cosines,
            }
        )

    count = min(max(int(worst_count), 0), len(tokens))

    def worst(metric: str) -> list[dict[str, Any]]:
        selected = sorted(tokens, key=lambda row: row[metric])[:count]
        return [
            {
                "batch_index": row["batch_index"],
                "token_index": row["token_index"],
                "token_label": row["token_label"],
                "value": row[metric],
            }
            for row in selected
        ]

    return {
        "tokens": tokens,
        "worst_tokens": {
            "post_gelu_cosine": worst("post_gelu_cosine"),
            "post_rmsnorm_cosine": worst("post_rmsnorm_cosine"),
            "minimum_value_cosine": worst("minimum_value_cosine"),
            "sign_agreement": worst("pre_gelu_sign_agreement_fraction"),
        },
    }


def _single_vector_metrics(
    native: torch.Tensor, mapped: torch.Tensor
) -> dict[str, float]:
    """Return mapping metrics for one token vector."""
    native = native.detach().float().cpu()
    mapped = mapped.detach().float().cpu()
    delta = native - mapped
    native_norm = native.norm()
    mapped_norm = mapped.norm()
    l2 = delta.norm()
    return {
        "cosine": float(F.cosine_similarity(native, mapped, dim=0, eps=1e-12)),
        "l2_error": float(l2),
        "relative_l2_error": float(l2 / native_norm.clamp_min(1e-12)),
        "native_norm": float(native_norm),
        "mapped_norm": float(mapped_norm),
        "norm_ratio": float(mapped_norm / native_norm.clamp_min(1e-12)),
        "mean_absolute_error": float(delta.abs().mean()),
        "maximum_absolute_error": float(delta.abs().max()),
    }


def _coordinate_error_rows(
    native: torch.Tensor,
    mapped: torch.Tensor,
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Rank a token's coordinates by absolute native-minus-mapped error."""
    native = native.detach().float().cpu().flatten()
    mapped = mapped.detach().float().cpu().flatten()
    delta = native - mapped
    count = min(max(int(top_n), 0), delta.numel())
    if not count:
        return []
    rows: list[dict[str, Any]] = []
    for dimension in torch.topk(delta.abs(), k=count).indices.tolist():
        native_value = native[dimension]
        mapped_value = mapped[dimension]
        absolute_error = delta[dimension].abs()
        rows.append(
            {
                "dimension": int(dimension),
                "native_value": float(native_value),
                "mapped_value": float(mapped_value),
                "absolute_error": float(absolute_error),
                "relative_error": float(
                    absolute_error / native_value.abs().clamp_min(1e-12)
                ),
                "sign_match": bool(
                    torch.signbit(native_value) == torch.signbit(mapped_value)
                ),
            }
        )
    return rows


def _rms_dimension_attribution(
    native: torch.Tensor,
    mapped: torch.Tensor,
    *,
    blocks,
    top_n: int,
) -> dict[str, Any]:
    """Trace post-RMS coordinate errors through every learned Wk/Wv matrix.

    The propagated values are exact single-coordinate contributions before
    key normalization: ``||W[:, d] * (native[d] - mapped[d])||``.
    """
    native = native.detach().float().cpu().flatten()
    mapped = mapped.detach().float().cpu().flatten()
    delta = native - mapped
    absolute_delta = delta.abs()
    wk_norms: list[torch.Tensor] = []
    wv_norms: list[torch.Tensor] = []
    for block in blocks:
        weight = block.cross_attn.kv_linear.weight.detach().float().cpu()
        channels = weight.shape[0] // 2
        if weight.shape[0] != 2 * channels or weight.shape[1] != delta.numel():
            raise ValueError("Unexpected SANA kv_linear dimensions during token attribution")
        wk_norms.append(weight[:channels].norm(dim=0))
        wv_norms.append(weight[channels:].norm(dim=0))
    wk = torch.stack(wk_norms)
    wv = torch.stack(wv_norms)
    propagated_k = wk * absolute_delta.unsqueeze(0)
    propagated_v = wv * absolute_delta.unsqueeze(0)
    mean_propagated_k = propagated_k.mean(dim=0)
    mean_propagated_v = propagated_v.mean(dim=0)

    def row(dimension: int) -> dict[str, Any]:
        per_block = []
        for block_index in range(len(blocks)):
            per_block.append(
                {
                    "block": int(block_index),
                    "wk_column_l2": float(wk[block_index, dimension]),
                    "wv_column_l2": float(wv[block_index, dimension]),
                    "propagated_k_error": float(propagated_k[block_index, dimension]),
                    "propagated_v_error": float(propagated_v[block_index, dimension]),
                }
            )
        native_value = native[dimension]
        mapped_value = mapped[dimension]
        return {
            "dimension": int(dimension),
            "native_value": float(native_value),
            "mapped_value": float(mapped_value),
            "input_error": float(absolute_delta[dimension]),
            "relative_error": float(
                absolute_delta[dimension] / native_value.abs().clamp_min(1e-12)
            ),
            "sign_match": bool(
                torch.signbit(native_value) == torch.signbit(mapped_value)
            ),
            "mean_wk_column_l2": float(wk[:, dimension].mean()),
            "maximum_wk_column_l2": float(wk[:, dimension].max()),
            "mean_wv_column_l2": float(wv[:, dimension].mean()),
            "maximum_wv_column_l2": float(wv[:, dimension].max()),
            "mean_propagated_k_error": float(mean_propagated_k[dimension]),
            "maximum_propagated_k_error": float(propagated_k[:, dimension].max()),
            "mean_propagated_v_error": float(mean_propagated_v[dimension]),
            "maximum_propagated_v_error": float(propagated_v[:, dimension].max()),
            "per_block": per_block,
        }

    count = min(max(int(top_n), 0), delta.numel())
    if not count:
        return {
            "definition": "Attribution is measured before key normalization.",
            "top_by_input_error": [],
            "top_by_propagated_k_error": [],
            "top_by_propagated_v_error": [],
        }
    return {
        "definition": (
            "propagated_{k,v}_error = ||W{k,v}[:, d] * "
            "(native[d] - mapped[d])|| before key normalization"
        ),
        "top_by_input_error": [
            row(index)
            for index in torch.topk(absolute_delta, k=count).indices.tolist()
        ],
        "top_by_propagated_k_error": [
            row(index)
            for index in torch.topk(mean_propagated_k, k=count).indices.tolist()
        ],
        "top_by_propagated_v_error": [
            row(index)
            for index in torch.topk(mean_propagated_v, k=count).indices.tolist()
        ],
    }


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _token_attention_block_rows(
    *,
    native_records: list[dict[str, torch.Tensor]],
    mapped_records: list[dict[str, torch.Tensor]],
    active_token_count: int,
    top_k: int = 3,
) -> list[list[dict[str, Any]]]:
    """Return fixed-native-Q attention and contribution metrics per token/block."""
    rows_by_token: list[list[dict[str, Any]]] = [
        [] for _ in range(active_token_count)
    ]
    for block_index, (native_record, mapped_record) in enumerate(
        zip(native_records, mapped_records)
    ):
        query = native_record["query"].float()
        native_key = native_record["key_without_bias"].float()
        mapped_key = mapped_record["key_without_bias"].float()
        native_value = native_record["value"].float()
        mapped_value = mapped_record["value"].float()
        if native_key.shape[-2] != active_token_count:
            raise ValueError(
                "Captured cross-attention token count does not match common active tokens: "
                f"{native_key.shape[-2]} vs {active_token_count}"
            )
        scale = math.sqrt(query.shape[-1])
        native_logits = torch.matmul(query, native_key.transpose(-2, -1)) / scale
        mapped_logits = torch.matmul(query, mapped_key.transpose(-2, -1)) / scale
        native_attention = torch.softmax(native_logits, dim=-1)
        mapped_attention = torch.softmax(mapped_logits, dim=-1)
        native_argmax = native_attention.argmax(dim=-1)
        mapped_argmax = mapped_attention.argmax(dim=-1)
        selected_top_k = min(max(int(top_k), 1), active_token_count)
        native_top = native_attention.topk(selected_top_k, dim=-1).indices
        mapped_top = mapped_attention.topk(selected_top_k, dim=-1).indices

        for token_offset in range(active_token_count):
            native_token_logits = native_logits[..., token_offset]
            mapped_token_logits = mapped_logits[..., token_offset]
            native_probability = native_attention[..., token_offset]
            mapped_probability = mapped_attention[..., token_offset]
            probability_difference = mapped_probability - native_probability
            logit_difference = mapped_token_logits - native_token_logits
            query_head_instances = native_probability.numel()
            native_argmax_count = int((native_argmax == token_offset).sum())
            mapped_argmax_count = int((mapped_argmax == token_offset).sum())
            native_top_k_count = int(
                (native_top == token_offset).any(dim=-1).sum()
            )
            mapped_top_k_count = int(
                (mapped_top == token_offset).any(dim=-1).sum()
            )

            native_contribution = (
                native_probability.unsqueeze(-1)
                * native_value[:, :, token_offset, :].unsqueeze(-2)
            )
            mapped_contribution = (
                mapped_probability.unsqueeze(-1)
                * mapped_value[:, :, token_offset, :].unsqueeze(-2)
            )
            contribution = _compare_tensors(
                native_contribution, mapped_contribution
            )
            value = _compare_tensors(
                native_value[:, :, token_offset, :],
                mapped_value[:, :, token_offset, :],
            )
            rows_by_token[token_offset].append(
                {
                    "block": int(block_index),
                    "native_logit_mean": float(native_token_logits.mean()),
                    "mapped_logit_mean": float(mapped_token_logits.mean()),
                    "mean_logit_difference": float(logit_difference.mean()),
                    "mean_absolute_logit_difference": float(
                        logit_difference.abs().mean()
                    ),
                    "maximum_absolute_logit_difference": float(
                        logit_difference.abs().max()
                    ),
                    "native_attention_probability_mean": float(
                        native_probability.mean()
                    ),
                    "native_attention_probability_maximum": float(
                        native_probability.max()
                    ),
                    "mapped_attention_probability_mean": float(
                        mapped_probability.mean()
                    ),
                    "mapped_attention_probability_maximum": float(
                        mapped_probability.max()
                    ),
                    "mean_attention_probability_difference": float(
                        probability_difference.mean()
                    ),
                    "mean_absolute_attention_probability_difference": float(
                        probability_difference.abs().mean()
                    ),
                    "maximum_absolute_attention_probability_difference": float(
                        probability_difference.abs().max()
                    ),
                    "native_argmax_frequency": float(
                        native_argmax_count / query_head_instances
                    ),
                    "native_argmax_count": native_argmax_count,
                    "mapped_argmax_frequency": float(
                        mapped_argmax_count / query_head_instances
                    ),
                    "mapped_argmax_count": mapped_argmax_count,
                    "native_top_k_frequency": float(
                        native_top_k_count / query_head_instances
                    ),
                    "native_top_k_count": native_top_k_count,
                    "mapped_top_k_frequency": float(
                        mapped_top_k_count / query_head_instances
                    ),
                    "mapped_top_k_count": mapped_top_k_count,
                    "query_head_instances": int(query_head_instances),
                    "top_k": int(selected_top_k),
                    "value_cosine": value["cosine_similarity"],
                    "value_relative_l2_error": value["relative_l2_error"],
                    "native_contribution_norm": contribution["native_norm"],
                    "mapped_contribution_norm": contribution["mapped_norm"],
                    "contribution_cosine": contribution["cosine_similarity"],
                    "contribution_relative_l2_error": contribution[
                        "relative_l2_error"
                    ],
                }
            )
    return rows_by_token


def _cohort_summary(
    name: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        selected: list[float] = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            if value is not None:
                selected.append(float(value))
        return selected

    return {
        "cohort": name,
        "token_count": len(rows),
        "token_indices": [row["token_index"] for row in rows],
        "mean_raw_cosine": _mean(values(("raw_cosine",))),
        "mean_post_gelu_cosine": _mean(values(("post_gelu_cosine",))),
        "mean_post_rmsnorm_cosine": _mean(values(("post_rmsnorm_cosine",))),
        "mean_k_cosine_without_bias": _mean(
            values(("mean_k_cosine_without_bias",))
        ),
        "mean_v_cosine": _mean(values(("mean_v_cosine",))),
        "mean_relative_l2_error": _mean(values(("relative_l2_error",))),
        "mean_top_dimension_absolute_error": _mean(
            values(("top_dimension_summary", "mean_input_error"))
        ),
        "mean_wk_importance_of_top_error_dimensions": _mean(
            values(("top_dimension_summary", "mean_wk_column_l2"))
        ),
        "mean_wv_importance_of_top_error_dimensions": _mean(
            values(("top_dimension_summary", "mean_wv_column_l2"))
        ),
        "mean_propagated_k_error": _mean(
            values(("top_dimension_summary", "mean_propagated_k_error"))
        ),
        "mean_propagated_v_error": _mean(
            values(("top_dimension_summary", "mean_propagated_v_error"))
        ),
        "mean_native_attention_probability": _mean(
            values(("attention", "native_probability_mean_across_blocks"))
        ),
        "mean_native_argmax_frequency": _mean(
            values(("attention", "native_argmax_frequency_across_blocks"))
        ),
        "mean_contribution_relative_l2_error": _mean(
            values(("attention", "contribution_relative_l2_error_across_blocks"))
        ),
    }


def _build_token_importance_diagnostics(
    *,
    native_stages: dict[str, torch.Tensor],
    mapped_stages: dict[str, torch.Tensor],
    native_keys_without_bias: list[torch.Tensor],
    mapped_keys_without_bias: list[torch.Tensor],
    native_values: list[torch.Tensor],
    mapped_values: list[torch.Tensor],
    native_records: list[dict[str, torch.Tensor]],
    mapped_records: list[dict[str, torch.Tensor]],
    common_mask: torch.Tensor,
    token_labels: list[list[str]] | None,
    blocks,
    top_n_dimensions: int,
    worst_token_count: int,
) -> dict[str, Any]:
    """Join mapping errors, learned K/V sensitivity, and actual attention use."""
    if common_mask.shape[0] != 1:
        return {
            "skipped": True,
            "reason": (
                "Token-level functional attribution currently requires the single-prompt "
                "debug mode because SANA packs multi-sample text tokens."
            ),
        }
    active_positions = torch.nonzero(common_mask[0], as_tuple=False).flatten().tolist()
    functional_rows = _token_attention_block_rows(
        native_records=native_records,
        mapped_records=mapped_records,
        active_token_count=len(active_positions),
    )
    tokens: list[dict[str, Any]] = []
    for active_offset, token_index in enumerate(active_positions):
        stage_metrics = {
            name: _single_vector_metrics(
                native_stages[name][0, token_index],
                mapped_stages[name][0, token_index],
            )
            for name in native_stages
        }
        key_metrics = [
            _single_vector_metrics(native[0, token_index], mapped[0, token_index])
            for native, mapped in zip(
                native_keys_without_bias, mapped_keys_without_bias
            )
        ]
        value_metrics = [
            _single_vector_metrics(native[0, token_index], mapped[0, token_index])
            for native, mapped in zip(native_values, mapped_values)
        ]
        dimensions = {
            "before_gelu": {
                "direct_wk_wv_attribution_applicable": False,
                "top_by_absolute_error": _coordinate_error_rows(
                    native_stages["after_fc1"][0, token_index],
                    mapped_stages["after_fc1"][0, token_index],
                    top_n=top_n_dimensions,
                ),
            },
            "after_gelu": {
                "direct_wk_wv_attribution_applicable": False,
                "top_by_absolute_error": _coordinate_error_rows(
                    native_stages["after_gelu"][0, token_index],
                    mapped_stages["after_gelu"][0, token_index],
                    top_n=top_n_dimensions,
                ),
            },
            "after_rmsnorm": _rms_dimension_attribution(
                native_stages["after_rmsnorm"][0, token_index],
                mapped_stages["after_rmsnorm"][0, token_index],
                blocks=blocks,
                top_n=top_n_dimensions,
            ),
        }
        top_dimension_rows = dimensions["after_rmsnorm"]["top_by_input_error"]
        block_attention = functional_rows[active_offset]
        strongest_attention_change = max(
            block_attention,
            key=lambda row: row[
                "mean_absolute_attention_probability_difference"
            ],
        )
        label = (
            token_labels[0][token_index] if token_labels is not None else None
        )
        token = {
            "batch_index": 0,
            "token_index": int(token_index),
            "active_token_offset": int(active_offset),
            "decoded_token": label,
            "stage_metrics": stage_metrics,
            "raw_cosine": stage_metrics["raw_embedding"]["cosine"],
            "after_fc1_cosine": stage_metrics["after_fc1"]["cosine"],
            "post_gelu_cosine": stage_metrics["after_gelu"]["cosine"],
            "after_fc2_cosine": stage_metrics["after_fc2"]["cosine"],
            "post_rmsnorm_cosine": stage_metrics["after_rmsnorm"]["cosine"],
            "relative_l2_error": stage_metrics["after_rmsnorm"][
                "relative_l2_error"
            ],
            "mean_k_cosine_without_bias": _mean(
                [metric["cosine"] for metric in key_metrics]
            ),
            "minimum_k_cosine_without_bias": min(
                metric["cosine"] for metric in key_metrics
            ),
            "mean_v_cosine": _mean(
                [metric["cosine"] for metric in value_metrics]
            ),
            "minimum_v_cosine": min(metric["cosine"] for metric in value_metrics),
            "k_without_bias_by_block": [
                {"block": index, **metric}
                for index, metric in enumerate(key_metrics)
            ],
            "v_by_block": [
                {"block": index, **metric}
                for index, metric in enumerate(value_metrics)
            ],
            "dimension_errors": dimensions,
            "top_dimension_summary": {
                "dimension_count": len(top_dimension_rows),
                "mean_input_error": _mean(
                    [row["input_error"] for row in top_dimension_rows]
                ),
                "mean_wk_column_l2": _mean(
                    [row["mean_wk_column_l2"] for row in top_dimension_rows]
                ),
                "mean_wv_column_l2": _mean(
                    [row["mean_wv_column_l2"] for row in top_dimension_rows]
                ),
                "mean_propagated_k_error": _mean(
                    [
                        row["mean_propagated_k_error"]
                        for row in top_dimension_rows
                    ]
                ),
                "mean_propagated_v_error": _mean(
                    [
                        row["mean_propagated_v_error"]
                        for row in top_dimension_rows
                    ]
                ),
            },
            "attention": {
                "native_probability_mean_across_blocks": _mean(
                    [
                        row["native_attention_probability_mean"]
                        for row in block_attention
                    ]
                ),
                "native_probability_maximum_across_blocks": max(
                    row["native_attention_probability_maximum"]
                    for row in block_attention
                ),
                "mapped_probability_mean_across_blocks": _mean(
                    [
                        row["mapped_attention_probability_mean"]
                        for row in block_attention
                    ]
                ),
                "native_argmax_frequency_across_blocks": _mean(
                    [row["native_argmax_frequency"] for row in block_attention]
                ),
                "native_top_k_frequency_across_blocks": _mean(
                    [row["native_top_k_frequency"] for row in block_attention]
                ),
                "mean_absolute_probability_change_across_blocks": _mean(
                    [
                        row["mean_absolute_attention_probability_difference"]
                        for row in block_attention
                    ]
                ),
                "contribution_cosine_across_blocks": _mean(
                    [row["contribution_cosine"] for row in block_attention]
                ),
                "contribution_relative_l2_error_across_blocks": _mean(
                    [
                        row["contribution_relative_l2_error"]
                        for row in block_attention
                    ]
                ),
                "strongest_attention_change": strongest_attention_change,
                "per_block": block_attention,
            },
        }
        tokens.append(token)

    ranked = sorted(tokens, key=lambda row: row["post_rmsnorm_cosine"])
    for rank, row in enumerate(ranked, start=1):
        row["rank_worst_to_best"] = rank

    cohort_size = max(1, math.ceil(0.1 * len(ranked)))
    worst = ranked[:cohort_size]
    best_start = max(cohort_size, len(ranked) - cohort_size)
    middle = ranked[cohort_size:best_start]
    best = ranked[best_start:]
    cohorts = [
        _cohort_summary("worst_10_percent", worst),
        _cohort_summary("middle_80_percent", middle),
        _cohort_summary("best_10_percent", best),
    ]

    repeated: list[dict[str, Any]] = []
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in tokens:
        if row["decoded_token"] is not None:
            by_label.setdefault(row["decoded_token"], []).append(row)
    for label, occurrences in by_label.items():
        if len(occurrences) < 2:
            continue
        occurrence_rows = []
        for row in sorted(occurrences, key=lambda item: item["token_index"]):
            occurrence_rows.append(
                {
                    "position": row["token_index"],
                    "raw_cosine": row["raw_cosine"],
                    "post_rmsnorm_cosine": row["post_rmsnorm_cosine"],
                    "mean_k_cosine_without_bias": row[
                        "mean_k_cosine_without_bias"
                    ],
                    "mean_v_cosine": row["mean_v_cosine"],
                    "native_attention_probability": row["attention"][
                        "native_probability_mean_across_blocks"
                    ],
                    "mapped_attention_probability": row["attention"][
                        "mapped_probability_mean_across_blocks"
                    ],
                }
            )
        pairwise = []
        for left_index in range(len(occurrences)):
            for right_index in range(left_index + 1, len(occurrences)):
                left = occurrences[left_index]["token_index"]
                right = occurrences[right_index]["token_index"]
                pairwise.append(
                    {
                        "positions": [left, right],
                        "native_raw_occurrence_difference": _single_vector_metrics(
                            native_stages["raw_embedding"][0, left],
                            native_stages["raw_embedding"][0, right],
                        ),
                        "mapped_raw_occurrence_difference": _single_vector_metrics(
                            mapped_stages["raw_embedding"][0, left],
                            mapped_stages["raw_embedding"][0, right],
                        ),
                        "native_rms_occurrence_difference": _single_vector_metrics(
                            native_stages["after_rmsnorm"][0, left],
                            native_stages["after_rmsnorm"][0, right],
                        ),
                        "mapped_rms_occurrence_difference": _single_vector_metrics(
                            mapped_stages["after_rmsnorm"][0, left],
                            mapped_stages["after_rmsnorm"][0, right],
                        ),
                    }
                )
        repeated.append(
            {
                "decoded_token": label,
                "occurrences": occurrence_rows,
                "pairwise_context_difference": pairwise,
            }
        )

    def normalized_rank(values: list[float], value: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        # Keep the weakest factor non-zero so the combined score remains
        # useful even when a prompt contains only a handful of active tokens.
        return (ordered.index(value) + 1) / len(ordered)

    mapping_values = [row["relative_l2_error"] for row in tokens]
    sensitivity_values = [
        row["top_dimension_summary"]["mean_propagated_k_error"]
        + row["top_dimension_summary"]["mean_propagated_v_error"]
        for row in tokens
    ]
    attention_values = [
        row["attention"]["native_probability_mean_across_blocks"]
        for row in tokens
    ]
    damaging: list[dict[str, Any]] = []
    for row, sensitivity in zip(tokens, sensitivity_values):
        severity_rank = normalized_rank(mapping_values, row["relative_l2_error"])
        sensitivity_rank = normalized_rank(sensitivity_values, sensitivity)
        attention_rank = normalized_rank(
            attention_values,
            row["attention"]["native_probability_mean_across_blocks"],
        )
        score = (severity_rank * sensitivity_rank * attention_rank) ** (1.0 / 3.0)
        top_dimensions = row["dimension_errors"]["after_rmsnorm"][
            "top_by_input_error"
        ][:5]
        damaging.append(
            {
                "token_index": row["token_index"],
                "decoded_token": row["decoded_token"],
                "damage_score": float(score),
                "post_rmsnorm_cosine": row["post_rmsnorm_cosine"],
                "relative_l2_error": row["relative_l2_error"],
                "mean_k_cosine_without_bias": row[
                    "mean_k_cosine_without_bias"
                ],
                "mean_v_cosine": row["mean_v_cosine"],
                "native_attention_probability": row["attention"][
                    "native_probability_mean_across_blocks"
                ],
                "native_attention_rank": 1
                + sorted(attention_values, reverse=True).index(
                    row["attention"]["native_probability_mean_across_blocks"]
                ),
                "top_bad_dimensions": [
                    dimension["dimension"] for dimension in top_dimensions
                ],
                "mean_propagated_k_error": row["top_dimension_summary"][
                    "mean_propagated_k_error"
                ],
                "mean_propagated_v_error": row["top_dimension_summary"][
                    "mean_propagated_v_error"
                ],
                "strongest_attention_change": row["attention"][
                    "strongest_attention_change"
                ],
                "likely_mechanism": (
                    "mapping error -> sensitive Wk/Wv coordinates -> changed fixed-Q "
                    "attention and/or value contribution"
                ),
            }
        )
    damaging.sort(key=lambda row: row["damage_score"], reverse=True)
    severe_half = sorted(
        tokens, key=lambda row: row["relative_l2_error"], reverse=True
    )[: max(1, math.ceil(len(tokens) / 2))]
    counterexamples = sorted(
        severe_half,
        key=lambda row: row["attention"][
            "native_probability_mean_across_blocks"
        ],
    )[: max(1, min(worst_token_count, len(severe_half)))]
    counterexamples = [
        {
            "token_index": row["token_index"],
            "decoded_token": row["decoded_token"],
            "relative_l2_error": row["relative_l2_error"],
            "post_rmsnorm_cosine": row["post_rmsnorm_cosine"],
            "native_attention_probability": row["attention"][
                "native_probability_mean_across_blocks"
            ],
            "native_argmax_frequency": row["attention"][
                "native_argmax_frequency_across_blocks"
            ],
        }
        for row in counterexamples
    ]
    return {
        "skipped": False,
        "ranking_basis": "post-RMSNorm cosine, ascending (worst first)",
        "attention_definition": (
            "Fixed native Q; statistics aggregate batch, attention heads, and image queries."
        ),
        "tokens_ranked_worst_to_best": ranked,
        "cohort_comparison": cohorts,
        "repeated_token_comparison": repeated,
        "most_damaging_token_errors": damaging[: max(1, worst_token_count)],
        "badly_mapped_low_attention_counterexamples": counterexamples,
    }


def _iter_scalar_leaves(value: Any, path: str = ""):
    """Yield JSON-compatible scalar leaves for a comparison-friendly CSV."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_scalar_leaves(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_scalar_leaves(child, f"{path}[{index}]")
    elif value is None or isinstance(value, (str, int, float, bool)):
        yield path, value


def _save_structured_report(report: dict[str, Any], output_path: str | Path) -> tuple[Path, Path]:
    json_path = Path(output_path)
    csv_path = json_path.with_suffix(".csv")
    token_csv_path = json_path.with_name(f"{json_path.stem}_tokens.csv")
    dimension_csv_path = json_path.with_name(f"{json_path.stem}_dimensions.csv")
    attention_csv_path = json_path.with_name(f"{json_path.stem}_attention.csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = report.get("token_importance_diagnostics", {})
    has_token_diagnostics = diagnostics and not diagnostics.get("skipped", False)
    report["artifacts"] = {
        "json": str(json_path),
        # Backward-compatible alias retained for earlier report consumers.
        "csv": str(csv_path),
        "scalar_csv": str(csv_path),
    }
    if has_token_diagnostics:
        report["artifacts"].update(
            {
                "token_csv": str(token_csv_path),
                "dimension_csv": str(dimension_csv_path),
                "attention_csv": str(attention_csv_path),
            }
        )

    json_temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(json_temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    json_temporary.replace(json_path)

    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(csv_temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "value"))
        writer.writeheader()
        for path, value in _iter_scalar_leaves(report):
            writer.writerow({"path": path, "value": value})
    csv_temporary.replace(csv_path)

    if has_token_diagnostics:
        token_rows = diagnostics["tokens_ranked_worst_to_best"]
        token_fields = (
            "rank_worst_to_best",
            "token_index",
            "decoded_token",
            "raw_cosine",
            "after_fc1_cosine",
            "post_gelu_cosine",
            "after_fc2_cosine",
            "post_rmsnorm_cosine",
            "mean_k_cosine_without_bias",
            "minimum_k_cosine_without_bias",
            "mean_v_cosine",
            "minimum_v_cosine",
            "relative_l2_error",
            "native_attention_probability",
            "native_attention_maximum",
            "native_argmax_frequency",
            "native_top_k_frequency",
            "mean_absolute_probability_change",
            "contribution_cosine",
            "contribution_relative_l2_error",
        )
        temporary = token_csv_path.with_suffix(token_csv_path.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=token_fields)
            writer.writeheader()
            for row in token_rows:
                attention = row["attention"]
                writer.writerow(
                    {
                        "rank_worst_to_best": row["rank_worst_to_best"],
                        "token_index": row["token_index"],
                        "decoded_token": row["decoded_token"],
                        "raw_cosine": row["raw_cosine"],
                        "after_fc1_cosine": row["after_fc1_cosine"],
                        "post_gelu_cosine": row["post_gelu_cosine"],
                        "after_fc2_cosine": row["after_fc2_cosine"],
                        "post_rmsnorm_cosine": row["post_rmsnorm_cosine"],
                        "mean_k_cosine_without_bias": row[
                            "mean_k_cosine_without_bias"
                        ],
                        "minimum_k_cosine_without_bias": row[
                            "minimum_k_cosine_without_bias"
                        ],
                        "mean_v_cosine": row["mean_v_cosine"],
                        "minimum_v_cosine": row["minimum_v_cosine"],
                        "relative_l2_error": row["relative_l2_error"],
                        "native_attention_probability": attention[
                            "native_probability_mean_across_blocks"
                        ],
                        "native_attention_maximum": attention[
                            "native_probability_maximum_across_blocks"
                        ],
                        "native_argmax_frequency": attention[
                            "native_argmax_frequency_across_blocks"
                        ],
                        "native_top_k_frequency": attention[
                            "native_top_k_frequency_across_blocks"
                        ],
                        "mean_absolute_probability_change": attention[
                            "mean_absolute_probability_change_across_blocks"
                        ],
                        "contribution_cosine": attention[
                            "contribution_cosine_across_blocks"
                        ],
                        "contribution_relative_l2_error": attention[
                            "contribution_relative_l2_error_across_blocks"
                        ],
                    }
                )
        temporary.replace(token_csv_path)

        dimension_fields = (
            "token_index",
            "decoded_token",
            "stage",
            "ranking",
            "dimension",
            "native_value",
            "mapped_value",
            "input_error",
            "relative_error",
            "sign_match",
            "mean_wk_column_l2",
            "mean_wv_column_l2",
            "mean_propagated_k_error",
            "mean_propagated_v_error",
        )
        temporary = dimension_csv_path.with_suffix(
            dimension_csv_path.suffix + ".tmp"
        )
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=dimension_fields)
            writer.writeheader()
            for token in token_rows:
                dimensions = token["dimension_errors"]
                for stage in ("before_gelu", "after_gelu"):
                    for row in dimensions[stage]["top_by_absolute_error"]:
                        writer.writerow(
                            {
                                "token_index": token["token_index"],
                                "decoded_token": token["decoded_token"],
                                "stage": stage,
                                "ranking": "absolute_input_error",
                                "dimension": row["dimension"],
                                "native_value": row["native_value"],
                                "mapped_value": row["mapped_value"],
                                "input_error": row["absolute_error"],
                                "relative_error": row["relative_error"],
                                "sign_match": row["sign_match"],
                            }
                        )
                rms = dimensions["after_rmsnorm"]
                for ranking in (
                    "top_by_input_error",
                    "top_by_propagated_k_error",
                    "top_by_propagated_v_error",
                ):
                    for row in rms[ranking]:
                        writer.writerow(
                            {
                                "token_index": token["token_index"],
                                "decoded_token": token["decoded_token"],
                                "stage": "after_rmsnorm",
                                "ranking": ranking,
                                "dimension": row["dimension"],
                                "native_value": row["native_value"],
                                "mapped_value": row["mapped_value"],
                                "input_error": row["input_error"],
                                "relative_error": row["relative_error"],
                                "sign_match": row["sign_match"],
                                "mean_wk_column_l2": row[
                                    "mean_wk_column_l2"
                                ],
                                "mean_wv_column_l2": row[
                                    "mean_wv_column_l2"
                                ],
                                "mean_propagated_k_error": row[
                                    "mean_propagated_k_error"
                                ],
                                "mean_propagated_v_error": row[
                                    "mean_propagated_v_error"
                                ],
                            }
                        )
        temporary.replace(dimension_csv_path)

        attention_fields = (
            "token_index",
            "decoded_token",
            "block",
            "native_logit_mean",
            "mapped_logit_mean",
            "mean_logit_difference",
            "mean_absolute_logit_difference",
            "native_attention_probability_mean",
            "mapped_attention_probability_mean",
            "mean_attention_probability_difference",
            "mean_absolute_attention_probability_difference",
            "native_argmax_frequency",
            "mapped_argmax_frequency",
            "native_top_k_frequency",
            "mapped_top_k_frequency",
            "value_cosine",
            "value_relative_l2_error",
            "native_contribution_norm",
            "mapped_contribution_norm",
            "contribution_cosine",
            "contribution_relative_l2_error",
        )
        temporary = attention_csv_path.with_suffix(
            attention_csv_path.suffix + ".tmp"
        )
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=attention_fields)
            writer.writeheader()
            for token in token_rows:
                for row in token["attention"]["per_block"]:
                    writer.writerow(
                        {
                            "token_index": token["token_index"],
                            "decoded_token": token["decoded_token"],
                            **{field: row[field] for field in attention_fields[2:]},
                        }
                    )
        temporary.replace(attention_csv_path)
    return json_path, csv_path


def _capture_caption_projection(model, caption: torch.Tensor) -> dict[str, torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}
    projection = model.y_embedder.y_proj
    modules = {
        "caption_projection_fc1": projection.fc1,
        "caption_projection_activation": projection.act,
        "caption_projection_fc2": projection.fc2,
    }
    handles = []
    for name, module in modules.items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, stage=name: captures.__setitem__(stage, output.detach())
            )
        )
    try:
        captures["caption_projection_output"] = model.y_embedder(
            caption, model.training
        ).detach()
    finally:
        for handle in handles:
            handle.remove()
    return captures


class _CrossAttentionCapture:
    """Capture functionally relevant tensors without changing attention."""

    def __init__(self, model):
        self.model = model
        self.records: list[dict[str, torch.Tensor]] = [dict() for _ in model.blocks]
        self.handles = []

    def __enter__(self):
        for block_index, block in enumerate(self.model.blocks):
            attention = block.cross_attn
            self.handles.append(
                attention.register_forward_pre_hook(
                    lambda module, inputs, index=block_index: self._pre_hook(index, module, inputs)
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    lambda _module, _inputs, output, index=block_index: self.records[index].__setitem__(
                        "cross_attention_output", output.detach().cpu()
                    )
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def _pre_hook(self, index: int, module, inputs) -> None:
        x, condition = inputs[:2]
        mask = inputs[2] if len(inputs) > 2 else None
        batch, image_tokens, channels = x.shape
        first_dim = 1 if getattr(module, "use_xformers", False) else batch

        query = module.q_norm(module.q_linear(x)).view(
            first_dim, -1, module.num_heads, module.head_dim
        )
        key_value = module.kv_linear(condition).view(first_dim, -1, 2, channels)
        key, value = key_value.unbind(2)
        key_without_bias = F.linear(
            condition, module.kv_linear.weight[:channels], None
        )
        key = module.k_norm(key).view(first_dim, -1, module.num_heads, module.head_dim)
        key_without_bias = module.k_norm(key_without_bias).view(
            first_dim, -1, module.num_heads, module.head_dim
        )
        value = value.view(first_dim, -1, module.num_heads, module.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        key_without_bias = key_without_bias.transpose(1, 2)
        value = value.transpose(1, 2)
        raw_logits = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(
            module.head_dim
        )
        logits_for_softmax = raw_logits
        reported_logits = raw_logits
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            additive_mask = (1 - mask.to(raw_logits.dtype)) * -10000.0
            logits_for_softmax = raw_logits + additive_mask[:, None, None, :]
            if mask.shape[0] == 1:
                reported_logits = raw_logits[..., mask[0].bool()]

        self.records[index]["query"] = query.detach().cpu()
        self.records[index]["key"] = key.detach().cpu()
        self.records[index]["key_without_bias"] = key_without_bias.detach().cpu()
        self.records[index]["value"] = value.detach().cpu()
        self.records[index]["attention_logits"] = reported_logits.detach().cpu()
        attention_weights = torch.softmax(logits_for_softmax, dim=-1)
        if isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.shape[0] == 1:
            attention_weights = attention_weights[..., mask[0].bool()]
        self.records[index]["attention_weights"] = attention_weights.detach().cpu()


class SanaConditioningDebugger:
    """Compare two text representations along SANA's real conditioning path."""

    def __init__(
        self,
        model,
        *,
        include_per_token: bool = False,
        top_n_dimensions: int = 20,
        worst_token_count: int = 5,
    ):
        self.model = model
        self.include_per_token = include_per_token
        self.top_n_dimensions = max(int(top_n_dimensions), 0)
        self.worst_token_count = max(int(worst_token_count), 0)
        self._validate_model()

    def _validate_model(self) -> None:
        required = ("y_embedder", "blocks", "forward_with_dpmsolver")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise TypeError(f"Model is not a supported SANA model; missing {missing}")
        if not hasattr(self.model.y_embedder, "y_proj"):
            raise TypeError("SANA caption embedder has no y_proj module")
        for index, block in enumerate(self.model.blocks):
            attention = getattr(block, "cross_attn", None)
            if attention is None or not hasattr(attention, "kv_linear"):
                raise TypeError(f"SANA block {index} has no supported cross-attention kv_linear")

    @torch.no_grad()
    def run(
        self,
        native_embedding: torch.Tensor,
        mapped_embedding: torch.Tensor,
        *,
        native_mask: torch.Tensor,
        mapped_mask: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        data_info: dict[str, Any],
        prompt: str | None = None,
        prompt_index: int | None = None,
        token_labels: list[list[str]] | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        native_tokens = _as_token_tensor(native_embedding)
        mapped_tokens = _as_token_tensor(mapped_embedding)
        if native_tokens.shape != mapped_tokens.shape:
            raise ValueError(
                f"Native and mapped conditioning shapes differ: {tuple(native_tokens.shape)} "
                f"vs {tuple(mapped_tokens.shape)}"
            )
        batch, token_count, _channels = native_tokens.shape
        if token_labels is not None and (
            len(token_labels) != batch
            or any(len(labels) != token_count for labels in token_labels)
        ):
            raise ValueError(
                f"Expected token_labels shaped [{batch}][{token_count}]"
            )
        native_mask = _as_mask(native_mask, batch, token_count).to(native_tokens.device)
        mapped_mask = _as_mask(mapped_mask, batch, token_count).to(native_tokens.device)
        common_mask = native_mask & mapped_mask

        report: dict[str, Any] = {
            "title": "SANA CONDITIONING DEBUG",
            "architecture": {
                "model_class": self.model.__class__.__name__,
                "block_count": len(self.model.blocks),
                "caption_input_dimension": int(native_tokens.shape[-1]),
                "conditioning_dimension": int(self.model.hidden_size),
                "path": [
                    "text_encoder_last_hidden_state",
                    "model_dtype_cast",
                    "CaptionEmbedder.y_proj.fc1",
                    "CaptionEmbedder.y_proj.activation",
                    "CaptionEmbedder.y_proj.fc2",
                    "attention_y_norm",
                    "attention_mask_or_xformers_packing",
                    "per_block_cross_attn.kv_linear",
                    "per_block_cross_attention",
                ],
            },
            "prompt": prompt,
            "prompt_index": prompt_index,
            "token_labels": token_labels,
            "mask_alignment": {
                "native_active_tokens": int(native_mask.sum()),
                "mapped_active_tokens": int(mapped_mask.sum()),
                "common_active_tokens": int(common_mask.sum()),
                "native_active_positions": torch.nonzero(
                    native_mask, as_tuple=False
                ).cpu().tolist(),
                "mapped_active_positions": torch.nonzero(
                    mapped_mask, as_tuple=False
                ).cpu().tolist(),
                "different_mask_positions": torch.nonzero(
                    native_mask != mapped_mask, as_tuple=False
                ).cpu().tolist(),
            },
            "stages": {},
            "blocks": [],
        }

        if not common_mask.any():
            report["fatal_alignment_error"] = (
                "Native and mapped attention masks have no active positions in common. "
                "The projected sequence layout is incompatible with SANA's injection layout."
            )
            print("\n==============================")
            print("SANA CONDITIONING DEBUG")
            print("==============================")
            print("MASK ALIGNMENT FAILURE")
            print(report["fatal_alignment_error"])
            print(
                "Native active positions:",
                [position[1] for position in report["mask_alignment"]["native_active_positions"]],
            )
            print(
                "Mapped active positions:",
                [position[1] for position in report["mask_alignment"]["mapped_active_positions"]],
            )
            if output_path is not None:
                json_path, csv_path = _save_structured_report(report, output_path)
                print(f"Alignment failure reports: {json_path}, {csv_path}")
            return report

        report["stages"]["raw_full_sequence"] = _compare_tensors(
            native_tokens, mapped_tokens, include_per_token=self.include_per_token
        )
        report["stages"]["raw_common_active_tokens"] = _compare_tensors(
            native_tokens,
            mapped_tokens,
            mask=common_mask,
            include_per_token=self.include_per_token,
        )

        native_cast = native_tokens.to(dtype=self.model.dtype)
        mapped_cast = mapped_tokens.to(dtype=self.model.dtype)
        report["stages"]["model_dtype_cast"] = _compare_tensors(
            native_cast,
            mapped_cast,
            mask=common_mask,
            include_per_token=self.include_per_token,
        )

        native_projected = _capture_caption_projection(self.model, native_cast[:, None])
        mapped_projected = _capture_caption_projection(self.model, mapped_cast[:, None])
        for stage_name in native_projected:
            native_stage = _as_token_tensor(native_projected[stage_name])
            mapped_stage = _as_token_tensor(mapped_projected[stage_name])
            report["stages"][stage_name] = _compare_tensors(
                native_stage,
                mapped_stage,
                mask=common_mask,
                include_per_token=self.include_per_token,
            )

        native_pre_gelu = _as_token_tensor(
            native_projected["caption_projection_fc1"]
        )
        mapped_pre_gelu = _as_token_tensor(
            mapped_projected["caption_projection_fc1"]
        )
        native_post_gelu = _as_token_tensor(
            native_projected["caption_projection_activation"]
        )
        mapped_post_gelu = _as_token_tensor(
            mapped_projected["caption_projection_activation"]
        )
        report["pre_gelu_signs"] = _sign_metrics(
            native_pre_gelu, mapped_pre_gelu, mask=common_mask
        )
        report["gelu_analysis"] = _gelu_analysis(
            native_pre_gelu,
            mapped_pre_gelu,
            native_post_gelu,
            mapped_post_gelu,
            mask=common_mask,
            top_n=self.top_n_dimensions,
        )

        native_conditioning = native_projected["caption_projection_output"]
        mapped_conditioning = mapped_projected["caption_projection_output"]
        if getattr(self.model, "y_norm", False):
            native_conditioning = self.model.attention_y_norm(native_conditioning)
            mapped_conditioning = self.model.attention_y_norm(mapped_conditioning)
        native_conditioning_tokens = _as_token_tensor(native_conditioning)
        mapped_conditioning_tokens = _as_token_tensor(mapped_conditioning)
        report["stages"]["final_conditioning_after_attention_y_norm"] = _compare_tensors(
            native_conditioning_tokens,
            mapped_conditioning_tokens,
            mask=common_mask,
            include_per_token=self.include_per_token,
        )

        native_values_by_block: list[torch.Tensor] = []
        mapped_values_by_block: list[torch.Tensor] = []
        native_keys_without_bias_by_block: list[torch.Tensor] = []
        mapped_keys_without_bias_by_block: list[torch.Tensor] = []
        projection_input_metric = _compare_tensors(
            native_conditioning_tokens,
            mapped_conditioning_tokens,
            mask=common_mask,
            include_per_token=self.include_per_token,
        )
        for block_index, block in enumerate(self.model.blocks):
            attention = block.cross_attn
            channels = native_conditioning_tokens.shape[-1]
            native_kv = attention.kv_linear(native_conditioning_tokens).view(
                batch, token_count, 2, channels
            )
            mapped_kv = attention.kv_linear(mapped_conditioning_tokens).view(
                batch, token_count, 2, channels
            )
            native_key_raw, native_value = native_kv.unbind(2)
            mapped_key_raw, mapped_value = mapped_kv.unbind(2)
            native_key = attention.k_norm(native_key_raw)
            mapped_key = attention.k_norm(mapped_key_raw)
            kv_weight = attention.kv_linear.weight
            kv_bias = attention.kv_linear.bias
            key_weight = kv_weight[:channels]
            value_weight = kv_weight[channels:]
            native_key_without_bias_raw = F.linear(
                native_conditioning_tokens, key_weight, None
            )
            mapped_key_without_bias_raw = F.linear(
                mapped_conditioning_tokens, key_weight, None
            )
            native_value_without_bias = F.linear(
                native_conditioning_tokens, value_weight, None
            )
            mapped_value_without_bias = F.linear(
                mapped_conditioning_tokens, value_weight, None
            )
            native_key_without_bias = attention.k_norm(native_key_without_bias_raw)
            mapped_key_without_bias = attention.k_norm(mapped_key_without_bias_raw)
            key_raw_metric = _compare_tensors(
                native_key_raw,
                mapped_key_raw,
                mask=common_mask,
                include_per_token=self.include_per_token,
            )
            key_metric = _compare_tensors(
                native_key,
                mapped_key,
                mask=common_mask,
                include_per_token=self.include_per_token,
            )
            value_metric = _compare_tensors(
                native_value,
                mapped_value,
                mask=common_mask,
                include_per_token=self.include_per_token,
            )
            key_without_bias_metric = _compare_tensors(
                native_key_without_bias,
                mapped_key_without_bias,
                mask=common_mask,
                include_per_token=self.include_per_token,
            )
            value_without_bias_metric = _compare_tensors(
                native_value_without_bias,
                mapped_value_without_bias,
                mask=common_mask,
                include_per_token=self.include_per_token,
            )
            centered_native_key = _center_active_tokens(native_key, common_mask)
            centered_mapped_key = _center_active_tokens(mapped_key, common_mask)
            centered_key_metric = _compare_tensors(
                centered_native_key,
                centered_mapped_key,
                mask=common_mask,
                include_per_token=self.include_per_token,
            )
            if kv_bias is None:
                key_bias_norm = 0.0
                value_bias_norm = 0.0
            else:
                key_bias_norm = float(kv_bias[:channels].detach().float().norm())
                value_bias_norm = float(kv_bias[channels:].detach().float().norm())
            native_values_by_block.append(native_value)
            mapped_values_by_block.append(mapped_value)
            native_keys_without_bias_by_block.append(native_key_without_bias)
            mapped_keys_without_bias_by_block.append(mapped_key_without_bias)
            report["blocks"].append(
                {
                    "block": block_index,
                    "key_projection": {
                        "input": dict(projection_input_metric),
                        "output_weight_only_before_key_norm": _compare_tensors(
                            native_key_without_bias_raw,
                            mapped_key_without_bias_raw,
                            mask=common_mask,
                            include_per_token=self.include_per_token,
                        ),
                        "output_weight_only": key_without_bias_metric,
                        "output_before_key_norm": key_raw_metric,
                        "output": key_metric,
                        "output_centered_across_active_tokens": centered_key_metric,
                        "bias_norm": key_bias_norm,
                        "bias_to_native_output_norm_ratio": float(
                            key_bias_norm / max(key_metric["native_norm"], 1e-12)
                        ),
                    },
                    "value_projection": {
                        "input": dict(projection_input_metric),
                        "output_weight_only": value_without_bias_metric,
                        "output": value_metric,
                        "bias_norm": value_bias_norm,
                        "bias_to_native_output_norm_ratio": float(
                            value_bias_norm / max(value_metric["native_norm"], 1e-12)
                        ),
                    },
                    # Keep these aliases for compatibility with earlier reports.
                    "key": key_metric,
                    "value": value_metric,
                    "value_error_dimensions": _projection_dimension_analysis(
                        native_conditioning_tokens,
                        mapped_conditioning_tokens,
                        mask=common_mask,
                        kv_linear=attention.kv_linear,
                        top_n=self.top_n_dimensions,
                    ),
                }
            )

        report["per_token_statistics"] = _per_token_statistics(
            native_post_gelu=native_post_gelu,
            mapped_post_gelu=mapped_post_gelu,
            native_post_norm=native_conditioning_tokens,
            mapped_post_norm=mapped_conditioning_tokens,
            native_pre_gelu=native_pre_gelu,
            mapped_pre_gelu=mapped_pre_gelu,
            native_values=native_values_by_block,
            mapped_values=mapped_values_by_block,
            mask=common_mask,
            token_labels=token_labels,
            worst_count=self.worst_token_count,
        )

        native_input = native_tokens[:, None]
        mapped_input = mapped_tokens[:, None]
        native_mask_for_model = native_mask.to(dtype=torch.uint8)
        common_mask_for_model = common_mask.to(dtype=torch.uint8)

        normal_native_output = self.model.forward_with_dpmsolver(
            latent,
            timestep,
            native_input,
            data_info=data_info,
            mask=native_mask_for_model,
        )
        reinjected_native_output = self.model.forward_with_dpmsolver(
            latent.clone(),
            timestep.clone(),
            native_input.clone(),
            data_info=data_info,
            mask=native_mask_for_model.clone(),
        )
        native_with_mapped_mask_output = self.model.forward_with_dpmsolver(
            latent.clone(),
            timestep.clone(),
            native_input.clone(),
            data_info=data_info,
            mask=mapped_mask.to(dtype=torch.uint8),
        )
        sanity_difference = (normal_native_output.float() - reinjected_native_output.float()).abs()
        report["native_reinjection_sanity"] = {
            "passed": bool(
                torch.allclose(
                    normal_native_output,
                    reinjected_native_output,
                    rtol=1e-4,
                    atol=1e-5,
                )
            ),
            "maximum_absolute_difference": float(sanity_difference.max()),
            "mean_absolute_difference": float(sanity_difference.mean()),
            "output_comparison": _compare_tensors(
                normal_native_output, reinjected_native_output
            ),
        }
        mapped_mask_difference = (
            normal_native_output.float() - native_with_mapped_mask_output.float()
        ).abs()
        report["native_embedding_with_mapped_mask"] = {
            "matches_normal_native": bool(
                torch.allclose(
                    normal_native_output,
                    native_with_mapped_mask_output,
                    rtol=1e-4,
                    atol=1e-5,
                )
            ),
            "maximum_absolute_difference": float(mapped_mask_difference.max()),
            "mean_absolute_difference": float(mapped_mask_difference.mean()),
            "output_comparison": _compare_tensors(
                normal_native_output, native_with_mapped_mask_output
            ),
        }

        with _CrossAttentionCapture(self.model) as native_capture:
            native_common_output = self.model.forward_with_dpmsolver(
                latent,
                timestep,
                native_input,
                data_info=data_info,
                mask=common_mask_for_model,
            )
        with _CrossAttentionCapture(self.model) as mapped_capture:
            mapped_common_output = self.model.forward_with_dpmsolver(
                latent,
                timestep,
                mapped_input,
                data_info=data_info,
                mask=common_mask_for_model,
            )

        for block_index, block_report in enumerate(report["blocks"]):
            native_record = native_capture.records[block_index]
            mapped_record = mapped_capture.records[block_index]
            native_key_without_bias = native_record["key_without_bias"]
            mapped_key_without_bias = mapped_record["key_without_bias"]
            block_report["fixed_native_query_attention"] = (
                _fixed_query_attention_metrics(
                    native_record["query"],
                    native_record["key"],
                    mapped_record["key"],
                )
            )
            block_report["fixed_native_query_attention_without_key_bias"] = (
                _fixed_query_attention_metrics(
                    native_record["query"],
                    native_key_without_bias,
                    mapped_key_without_bias,
                )
            )
            block_report["key_bias_attention_effect"] = {
                "native_actual_vs_weight_only": _fixed_query_attention_metrics(
                    native_record["query"],
                    native_record["key"],
                    native_key_without_bias,
                ),
                "mapped_actual_vs_weight_only": _fixed_query_attention_metrics(
                    native_record["query"],
                    mapped_record["key"],
                    mapped_key_without_bias,
                ),
            }
            for stage_name in (
                "query",
                "attention_logits",
                "attention_weights",
                "cross_attention_output",
            ):
                block_report[stage_name] = _compare_tensors(
                    native_record[stage_name], mapped_record[stage_name]
                )

        report["token_importance_diagnostics"] = (
            _build_token_importance_diagnostics(
                native_stages={
                    "raw_embedding": native_tokens,
                    "after_fc1": native_pre_gelu,
                    "after_gelu": native_post_gelu,
                    "after_fc2": _as_token_tensor(
                        native_projected["caption_projection_fc2"]
                    ),
                    "after_rmsnorm": native_conditioning_tokens,
                },
                mapped_stages={
                    "raw_embedding": mapped_tokens,
                    "after_fc1": mapped_pre_gelu,
                    "after_gelu": mapped_post_gelu,
                    "after_fc2": _as_token_tensor(
                        mapped_projected["caption_projection_fc2"]
                    ),
                    "after_rmsnorm": mapped_conditioning_tokens,
                },
                native_keys_without_bias=native_keys_without_bias_by_block,
                mapped_keys_without_bias=mapped_keys_without_bias_by_block,
                native_values=native_values_by_block,
                mapped_values=mapped_values_by_block,
                native_records=native_capture.records,
                mapped_records=mapped_capture.records,
                common_mask=common_mask,
                token_labels=token_labels,
                blocks=self.model.blocks,
                top_n_dimensions=self.top_n_dimensions,
                worst_token_count=self.worst_token_count,
            )
        )

        report["stages"]["denoiser_output_at_debug_timestep"] = _compare_tensors(
            native_common_output, mapped_common_output
        )
        report["debug_timestep"] = timestep.detach().float().cpu().tolist()

        self.print_report(report)
        if output_path is not None:
            json_path, csv_path = _save_structured_report(report, output_path)
            print(f"Full conditioning debug reports: {json_path}, {csv_path}")
            artifacts = report.get("artifacts", {})
            if "token_csv" in artifacts:
                print(
                    "Token diagnostics CSVs: "
                    f"{artifacts['token_csv']}, {artifacts['dimension_csv']}, "
                    f"{artifacts['attention_csv']}"
                )
        return report

    @staticmethod
    def _print_metric(name: str, metric: dict[str, Any], indent: str = "") -> None:
        print(
            f"{indent}{name}: shape={metric['tensor_shape']} "
            f"cos={metric['cosine_similarity']:.6f} "
            f"token_cos[min/max]={metric['minimum_vector_cosine']:.6f}/"
            f"{metric['maximum_vector_cosine']:.6f} "
            f"L2={metric['mean_l2_distance']:.6f} "
            f"relative_L2={metric['relative_l2_error']:.6f} "
            f"norm[native/mapped]={metric['native_norm']:.6f}/"
            f"{metric['mapped_norm']:.6f} "
            f"norm_ratio={metric['norm_ratio']:.6f} "
            f"MAE/max={metric['mean_absolute_error']:.6f}/"
            f"{metric['maximum_absolute_error']:.6f}"
        )

    @classmethod
    def print_report(cls, report: dict[str, Any]) -> None:
        print("\n==============================")
        print("SANA CONDITIONING DEBUG")
        print("==============================")
        sanity = report["native_reinjection_sanity"]
        status = "PASS" if sanity["passed"] else "FAIL"
        print("Native reinjection sanity check:", status)
        print(
            "  max_abs_diff="
            f"{sanity['maximum_absolute_difference']:.8g} "
            f"mean_abs_diff={sanity['mean_absolute_difference']:.8g}"
        )
        if not sanity["passed"]:
            print("  WARNING: injection-path sanity failed; interpret later stages only after fixing it.")
        mapped_mask_sanity = report["native_embedding_with_mapped_mask"]
        mask_status = "MATCH" if mapped_mask_sanity["matches_normal_native"] else "DIFFERS"
        print("Native embedding with mapped attention mask:", mask_status)
        print(
            "  max_abs_diff="
            f"{mapped_mask_sanity['maximum_absolute_difference']:.8g} "
            f"mean_abs_diff={mapped_mask_sanity['mean_absolute_difference']:.8g}"
        )
        masks = report["mask_alignment"]
        print(
            "Masks: "
            f"native={masks['native_active_tokens']} mapped={masks['mapped_active_tokens']} "
            f"common={masks['common_active_tokens']} "
            f"different_positions={len(masks['different_mask_positions'])}"
        )
        print("\nConditioning stages")
        for stage_name, metric in report["stages"].items():
            cls._print_metric(stage_name, metric, indent="  ")

        signs = report["pre_gelu_signs"]
        gelu = report["gelu_analysis"]
        print("\nPRE-GELU SIGN DIAGNOSTICS")
        print(
            "  agreement="
            f"{100 * signs['sign_agreement_fraction']:.2f}% "
            f"disagreement={100 * signs['sign_disagreement_fraction']:.2f}% "
            "native+/mapped-="
            f"{100 * signs['native_positive_mapped_negative_fraction']:.2f}% "
            "native-/mapped+="
            f"{100 * signs['native_negative_mapped_positive_fraction']:.2f}%"
        )
        print(
            "  magnitude_weighted_sign_error="
            f"{100 * signs['magnitude_weighted_sign_error']:.2f}%"
        )
        for threshold, values in signs["threshold_sign_agreement"].items():
            agreement = values["agreement_fraction"]
            rendered = "n/a" if agreement is None else f"{100 * agreement:.2f}%"
            print(
                f"  |native| > {threshold}: agreement={rendered} "
                f"n={values['selected_coordinates']}"
            )

        print("\nGELU EFFECT")
        print(
            f"  cosine before/after={gelu['before']['cosine_similarity']:.6f}/"
            f"{gelu['after']['cosine_similarity']:.6f} "
            f"relative_L2 before/after={gelu['before']['relative_l2_error']:.6f}/"
            f"{gelu['after']['relative_l2_error']:.6f} "
            "coordinates amplified="
            f"{100 * gelu['mismatch_amplified_fraction']:.2f}%"
        )
        print("  Largest post-GELU coordinate errors (full top-N in JSON/CSV):")
        for row in gelu["top_post_gelu_error_coordinates"][:5]:
            print(
                "    token="
                f"{row['token_index']} dim={row['dimension']} "
                f"before={row['native_before']:.4g}/{row['mapped_before']:.4g} "
                f"after={row['native_after']:.4g}/{row['mapped_after']:.4g} "
                f"sign_match={row['sign_match_before']}"
            )

        print("\nPer-block cross-attention")
        for block in report["blocks"]:
            print(f"  Block {block['block']:02d}")
            cls._print_metric(
                "K input", block["key_projection"]["input"], indent="    "
            )
            cls._print_metric(
                "K output (weight only, before bias)",
                block["key_projection"]["output_weight_only"],
                indent="    ",
            )
            cls._print_metric(
                "K output (actual, with bias)",
                block["key_projection"]["output"],
                indent="    ",
            )
            cls._print_metric(
                "K output (token-centered)",
                block["key_projection"]["output_centered_across_active_tokens"],
                indent="    ",
            )
            cls._print_metric(
                "V output", block["value_projection"]["output"], indent="    "
            )
            print(
                "    K/V bias norm="
                f"{block['key_projection']['bias_norm']:.6f}/"
                f"{block['value_projection']['bias_norm']:.6f}"
            )
            fixed = block["fixed_native_query_attention"]
            fixed_without_bias = block[
                "fixed_native_query_attention_without_key_bias"
            ]
            bias_effect = block["key_bias_attention_effect"]
            print(
                "    fixed-Q attention: "
                f"cos={fixed['attention']['cosine_similarity']:.6f} "
                f"MAE={fixed['attention']['mean_absolute_error']:.6f} "
                f"JS={fixed['jensen_shannon_divergence']:.6f} "
                f"argmax_agreement={100 * fixed['argmax_token_agreement_fraction']:.2f}% "
                f"top3_overlap={100 * fixed['top3_token_overlap_fraction']:.2f}% "
                f"entropy[native/mapped]={fixed['native_attention_entropy']:.6f}/"
                f"{fixed['mapped_attention_entropy']:.6f}"
            )
            print(
                "    fixed-Q without K bias: "
                f"cos={fixed_without_bias['attention']['cosine_similarity']:.6f}; "
                "bias effect MAE[native/mapped]="
                f"{bias_effect['native_actual_vs_weight_only']['attention']['mean_absolute_error']:.8f}/"
                f"{bias_effect['mapped_actual_vs_weight_only']['attention']['mean_absolute_error']:.8f}"
            )
            print(
                "    full-path attention: "
                f"cos={block['attention_weights']['cosine_similarity']:.6f} "
                "(Q and K both allowed to diverge)"
            )
            print("    Largest estimated V-error dimensions (first 5; full top-N saved):")
            for row in block["value_error_dimensions"][
                "top_by_estimated_v_contribution"
            ][:5]:
                print(
                    f"      dim={row['dimension']:4d} "
                    f"input_rms={row['input_rms_error']:.5g} "
                    f"Wk={row['wk_column_l2']:.5g} "
                    f"Wv={row['wv_column_l2']:.5g} "
                    f"V_contrib={row['estimated_v_contribution_l2']:.5g}"
                )

        print("\nWorst active tokens")
        for metric, rows in report["per_token_statistics"]["worst_tokens"].items():
            rendered = ", ".join(
                f"{row['token_index']}:{row['token_label']!r}={row['value']:.4f}"
                for row in rows
            )
            print(f"  {metric}: {rendered}")

        diagnostics = report.get("token_importance_diagnostics")
        if not diagnostics:
            return
        if diagnostics.get("skipped", False):
            print("\nTOKEN IMPORTANCE DIAGNOSTICS SKIPPED")
            print(f"  {diagnostics['reason']}")
            return

        print("\n==============================")
        print("TOKEN IMPORTANCE DIAGNOSTICS")
        print("==============================")
        print(
            "rank token/position       raw     GELU      RMS   K(no b)      V   relL2  native_attn"
        )
        ranked = diagnostics["tokens_ranked_worst_to_best"]
        for row in ranked:
            label = repr(row["decoded_token"])
            token_position = f"{label}@{row['token_index']}"
            print(
                f"{row['rank_worst_to_best']:>4} {token_position:<18.18} "
                f"{row['raw_cosine']:>7.3f} "
                f"{row['post_gelu_cosine']:>8.3f} "
                f"{row['post_rmsnorm_cosine']:>8.3f} "
                f"{row['mean_k_cosine_without_bias']:>8.3f} "
                f"{row['mean_v_cosine']:>6.3f} "
                f"{row['relative_l2_error']:>8.3f} "
                f"{row['attention']['native_probability_mean_across_blocks']:>11.3f}"
            )

        print("\nWorst/middle/best cohorts")
        print("cohort                 n   RMS cos   V cos   prop K   prop V   native_attn")
        for cohort in diagnostics["cohort_comparison"]:
            def render(value: float | None) -> str:
                return "    n/a" if value is None else f"{value:>7.3f}"

            print(
                f"{cohort['cohort']:<22} {cohort['token_count']:>2} "
                f"{render(cohort['mean_post_rmsnorm_cosine'])} "
                f"{render(cohort['mean_v_cosine'])} "
                f"{render(cohort['mean_propagated_k_error'])} "
                f"{render(cohort['mean_propagated_v_error'])} "
                f"{render(cohort['mean_native_attention_probability'])}"
            )

        print("\nMOST DAMAGING TOKEN ERRORS")
        for row in diagnostics["most_damaging_token_errors"]:
            change = row["strongest_attention_change"]
            print(
                f"  {row['decoded_token']!r} position={row['token_index']} "
                f"score={row['damage_score']:.3f} RMS_cos={row['post_rmsnorm_cosine']:.3f} "
                f"attention_rank={row['native_attention_rank']}/{len(ranked)}"
            )
            print(
                f"    dims={row['top_bad_dimensions']} "
                f"propagated K/V={row['mean_propagated_k_error']:.4g}/"
                f"{row['mean_propagated_v_error']:.4g}"
            )
            print(
                f"    strongest block={change['block']} fixed-Q probability "
                f"native={change['native_attention_probability_mean']:.4f} -> "
                f"mapped={change['mapped_attention_probability_mean']:.4f}; "
                f"contribution cos={change['contribution_cosine']:.4f} "
                f"relL2={change['contribution_relative_l2_error']:.4f}"
            )

        repeated = diagnostics["repeated_token_comparison"]
        if repeated:
            print("\nRepeated lexical tokens")
            for group in repeated:
                rendered = ", ".join(
                    f"pos={row['position']} RMS={row['post_rmsnorm_cosine']:.3f} "
                    f"K={row['mean_k_cosine_without_bias']:.3f} "
                    f"V={row['mean_v_cosine']:.3f} "
                    f"attn={row['native_attention_probability']:.3f}"
                    for row in group["occurrences"]
                )
                print(f"  {group['decoded_token']!r}: {rendered}")


def debug_conditioning(
    native_embedding: torch.Tensor,
    mapped_embedding: torch.Tensor,
    *,
    model,
    native_mask: torch.Tensor,
    mapped_mask: torch.Tensor,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    data_info: dict[str, Any],
    prompt: str | None = None,
    prompt_index: int | None = None,
    token_labels: list[list[str]] | None = None,
    output_path: str | Path | None = None,
    include_per_token: bool = False,
    top_n_dimensions: int = 20,
    worst_token_count: int = 5,
) -> dict[str, Any]:
    """Convenience wrapper for :class:`SanaConditioningDebugger`."""
    return SanaConditioningDebugger(
        model,
        include_per_token=include_per_token,
        top_n_dimensions=top_n_dimensions,
        worst_token_count=worst_token_count,
    ).run(
        native_embedding,
        mapped_embedding,
        native_mask=native_mask,
        mapped_mask=mapped_mask,
        latent=latent,
        timestep=timestep,
        data_info=data_info,
        prompt=prompt,
        prompt_index=prompt_index,
        token_labels=token_labels,
        output_path=output_path,
    )
