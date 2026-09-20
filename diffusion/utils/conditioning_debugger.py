"""Read-only diagnostics for SANA text conditioning.

The debugger calls the model's existing caption projection, normalization,
cross-attention K/V projections, and denoiser forward methods.  It does not
replace modules or alter inference outputs.
"""

from __future__ import annotations

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

    native_norms = native_vectors.norm(dim=-1)
    mapped_norms = mapped_vectors.norm(dim=-1)
    cosines = F.cosine_similarity(native_vectors, mapped_vectors, dim=-1, eps=1e-12)
    l2 = (native_vectors - mapped_vectors).norm(dim=-1)
    native_norm = native_norms.mean()
    mapped_norm = mapped_norms.mean()
    result: dict[str, Any] = {
        "tensor_shape": original_shape,
        "compared_vectors": int(native_vectors.shape[0]),
        "cosine_similarity": float(cosines.mean()),
        "minimum_vector_cosine": float(cosines.min()),
        "maximum_vector_cosine": float(cosines.max()),
        "mean_l2_distance": float(l2.mean()),
        "native_norm": float(native_norm),
        "mapped_norm": float(mapped_norm),
        "relative_norm_difference": float(
            (mapped_norm - native_norm).abs() / native_norm.clamp_min(1e-12)
        ),
    }
    if include_per_token:
        result["per_vector_cosine"] = cosines.cpu().tolist()
    return result


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
        key, _value = key_value.unbind(2)
        key = module.k_norm(key).view(first_dim, -1, module.num_heads, module.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
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
        self.records[index]["attention_logits"] = reported_logits.detach().cpu()
        attention_weights = torch.softmax(logits_for_softmax, dim=-1)
        if isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.shape[0] == 1:
            attention_weights = attention_weights[..., mask[0].bool()]
        self.records[index]["attention_weights"] = attention_weights.detach().cpu()


class SanaConditioningDebugger:
    """Compare two text representations along SANA's real conditioning path."""

    def __init__(self, model, *, include_per_token: bool = False):
        self.model = model
        self.include_per_token = include_per_token
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
        native_mask = _as_mask(native_mask, batch, token_count).to(native_tokens.device)
        mapped_mask = _as_mask(mapped_mask, batch, token_count).to(native_tokens.device)
        common_mask = native_mask & mapped_mask
        if not common_mask.any():
            raise ValueError("Native and mapped masks have no active token positions in common")

        report: dict[str, Any] = {
            "title": "TEXT CONDITIONING DEBUG",
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
            "mask_alignment": {
                "native_active_tokens": int(native_mask.sum()),
                "mapped_active_tokens": int(mapped_mask.sum()),
                "common_active_tokens": int(common_mask.sum()),
                "different_mask_positions": torch.nonzero(
                    native_mask != mapped_mask, as_tuple=False
                ).cpu().tolist(),
            },
            "stages": {},
            "blocks": [],
        }

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

        for block_index, block in enumerate(self.model.blocks):
            attention = block.cross_attn
            channels = native_conditioning_tokens.shape[-1]
            native_kv = attention.kv_linear(native_conditioning_tokens).view(
                batch, token_count, 2, channels
            )
            mapped_kv = attention.kv_linear(mapped_conditioning_tokens).view(
                batch, token_count, 2, channels
            )
            native_key, native_value = native_kv.unbind(2)
            mapped_key, mapped_value = mapped_kv.unbind(2)
            native_key = attention.k_norm(native_key)
            mapped_key = attention.k_norm(mapped_key)
            report["blocks"].append(
                {
                    "block": block_index,
                    "key": _compare_tensors(
                        native_key,
                        mapped_key,
                        mask=common_mask,
                        include_per_token=self.include_per_token,
                    ),
                    "value": _compare_tensors(
                        native_value,
                        mapped_value,
                        mask=common_mask,
                        include_per_token=self.include_per_token,
                    ),
                }
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
            for stage_name in (
                "query",
                "attention_logits",
                "attention_weights",
                "cross_attention_output",
            ):
                block_report[stage_name] = _compare_tensors(
                    native_record[stage_name], mapped_record[stage_name]
                )

        report["stages"]["denoiser_output_at_debug_timestep"] = _compare_tensors(
            native_common_output, mapped_common_output
        )
        report["debug_timestep"] = timestep.detach().float().cpu().tolist()

        self.print_report(report)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            temporary_path.replace(output_path)
            print(f"Full conditioning debug report: {output_path}")
        return report

    @staticmethod
    def _print_metric(name: str, metric: dict[str, Any], indent: str = "") -> None:
        print(
            f"{indent}{name}: shape={metric['tensor_shape']} "
            f"cos={metric['cosine_similarity']:.6f} "
            f"token_cos[min/max]={metric['minimum_vector_cosine']:.6f}/"
            f"{metric['maximum_vector_cosine']:.6f} "
            f"L2={metric['mean_l2_distance']:.6f} "
            f"norm[native/mapped]={metric['native_norm']:.6f}/"
            f"{metric['mapped_norm']:.6f} "
            f"relative_norm_diff={metric['relative_norm_difference']:.6f}"
        )

    @classmethod
    def print_report(cls, report: dict[str, Any]) -> None:
        print("\n==============================")
        print("TEXT CONDITIONING DEBUG")
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
        print("\nPer-block cross-attention")
        for block in report["blocks"]:
            print(f"  Block {block['block']:02d}")
            for name in (
                "key",
                "value",
                "query",
                "attention_logits",
                "attention_weights",
                "cross_attention_output",
            ):
                cls._print_metric(name, block[name], indent="    ")


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
    output_path: str | Path | None = None,
    include_per_token: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper for :class:`SanaConditioningDebugger`."""
    return SanaConditioningDebugger(
        model, include_per_token=include_per_token
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
        output_path=output_path,
    )
