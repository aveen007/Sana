import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "diffusion"
    / "utils"
    / "conditioning_debugger.py"
)
_SPEC = importlib.util.spec_from_file_location("conditioning_debugger_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DEBUGGER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEBUGGER)

SanaConditioningDebugger = _DEBUGGER.SanaConditioningDebugger
_fixed_query_attention_metrics = _DEBUGGER._fixed_query_attention_metrics
_projection_dimension_analysis = _DEBUGGER._projection_dimension_analysis
_sign_metrics = _DEBUGGER._sign_metrics


class _CaptionProjection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, tensor):
        return self.fc2(self.act(self.fc1(tensor)))


class _CaptionEmbedder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.y_proj = _CaptionProjection(channels)

    def forward(self, caption, train):
        return self.y_proj(caption)


class _CrossAttention(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.head_dim = channels // heads
        self.use_xformers = False
        self.q_linear = nn.Linear(channels, channels)
        self.kv_linear = nn.Linear(channels, channels * 2)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.proj = nn.Linear(channels, channels)

    def forward(self, image, condition, mask=None):
        batch, image_tokens, channels = image.shape
        query = self.q_linear(image).view(
            batch, image_tokens, self.num_heads, self.head_dim
        )
        key_value = self.kv_linear(condition).view(batch, -1, 2, channels)
        key, value = key_value.unbind(2)
        key = key.view(batch, -1, self.num_heads, self.head_dim)
        value = value.view(batch, -1, self.num_heads, self.head_dim)
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        return self.proj(output.transpose(1, 2).reshape(batch, image_tokens, channels))


class _Block(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.cross_attn = _CrossAttention(channels, heads)

    def forward(self, image, condition, mask=None):
        return image + self.cross_attn(image, condition, mask)


class _Model(nn.Module):
    def __init__(self, channels: int = 4):
        super().__init__()
        self.hidden_size = channels
        self.dtype = torch.float32
        self.y_norm = True
        self.y_embedder = _CaptionEmbedder(channels)
        self.attention_y_norm = nn.RMSNorm(channels)
        self.blocks = nn.ModuleList([_Block(channels, 2), _Block(channels, 2)])

    def forward_with_dpmsolver(self, latent, timestep, caption, data_info=None, mask=None):
        condition = self.attention_y_norm(self.y_embedder(caption, self.training))
        condition = condition.squeeze(1).masked_select(mask.bool().unsqueeze(-1)).view(
            1, -1, self.hidden_size
        )
        image = latent
        lengths = mask.sum(dim=1).tolist()
        for block in self.blocks:
            image = block(image, condition, lengths)
        return image


class ConditioningDebuggerTests(unittest.TestCase):
    def test_sign_metrics_weight_serious_disagreements_more(self):
        native = torch.tensor([2.0, -2.0, 0.001, -0.001])
        mapped = torch.tensor([-2.0, -2.0, -0.001, -0.001])
        result = _sign_metrics(native, mapped)

        self.assertEqual(result["sign_agreement_fraction"], 0.5)
        self.assertGreater(result["magnitude_weighted_sign_error"], 0.49)
        self.assertEqual(
            result["threshold_sign_agreement"]["0.01"]["agreement_fraction"],
            0.5,
        )

    def test_value_dimension_analysis_finds_strong_value_only_dimension(self):
        layer = nn.Linear(3, 6, bias=False)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[3:, 1] = 4.0
        native = torch.tensor([[[0.0, 2.0, 0.0]]])
        mapped = torch.zeros_like(native)
        result = _projection_dimension_analysis(
            native,
            mapped,
            mask=torch.ones(1, 1, dtype=torch.bool),
            kv_linear=layer,
            top_n=1,
        )

        top = result["top_by_estimated_v_contribution"][0]
        self.assertEqual(top["dimension"], 1)
        self.assertGreater(top["wv_column_l2"], 0)
        self.assertEqual(top["wk_column_l2"], 0)
        self.assertEqual(result["value_error_reconstruction_max_abs_residual"], 0)

    def test_fixed_query_attention_is_identical_when_keys_match(self):
        query = torch.randn(1, 2, 3, 4)
        key = torch.randn(1, 2, 5, 4)
        result = _fixed_query_attention_metrics(query, key, key.clone())

        self.assertGreater(result["attention"]["cosine_similarity"], 0.999999)
        self.assertEqual(result["jensen_shannon_divergence"], 0)
        self.assertEqual(result["argmax_token_agreement_fraction"], 1)
        self.assertEqual(result["top3_token_overlap_fraction"], 1)

    def test_full_debugger_writes_json_and_csv(self):
        torch.manual_seed(7)
        model = _Model().eval()
        native = torch.randn(1, 1, 3, 4)
        mapped = native + 0.2 * torch.randn_like(native)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.uint8)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "conditioning.json"
            report = SanaConditioningDebugger(
                model, top_n_dimensions=2, worst_token_count=2
            ).run(
                native,
                mapped,
                native_mask=mask,
                mapped_mask=mask,
                latent=torch.randn(1, 3, 4),
                timestep=torch.tensor([500.0]),
                data_info={},
                prompt="test prompt",
                prompt_index=0,
                token_labels=[["a", "b", "<pad>"]],
                output_path=output,
            )

            self.assertTrue(report["native_reinjection_sanity"]["passed"])
            self.assertIn("gelu_analysis", report)
            self.assertIn("fixed_native_query_attention", report["blocks"][0])
            self.assertEqual(len(report["per_token_statistics"]["tokens"]), 2)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".csv").is_file())
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(saved["artifacts"]["csv"].endswith("conditioning.csv"))


if __name__ == "__main__":
    unittest.main()
