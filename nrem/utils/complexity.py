"""Deterministic parameter and FLOP profiling for NREM paper models.

The profiler intentionally uses a small, dependency-free counting convention
instead of relying on version-sensitive third-party packages. A multiply-add
is counted as two floating-point operations. Linear, convolution, transposed
convolution, and attention matrix multiplications are counted from the shapes
observed during a real inference forward pass.

For ``torch.nn.TransformerEncoderLayer`` we additionally count 16 operations
per output element for normalization, residual, activation, and other
pointwise work. This convention is shared by Transformer-AE and TranAD and is
recorded in every generated result row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from nrem.config import load_config
from nrem.methods.factory import build_method


FLOP_CONVENTION = (
    "multiply_add=2; linear/conv from runtime tensor shapes; "
    "attention includes QK^T and AV; transformer pointwise overhead=16/output_element"
)


@dataclass(frozen=True)
class ComplexityResult:
    method: str
    config: str
    input_shape: str
    parameters: int
    trainable_parameters: int
    flops_per_sample: int
    parameters_millions: float
    flops_giga_per_sample: float
    flop_convention: str = FLOP_CONVENTION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return int(total), int(trainable)


class _FlopCounter:
    def __init__(self) -> None:
        self.flops = 0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _first_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
        raise TypeError(f"Expected a tensor output, got {type(value)!r}")

    def _linear(self, module: nn.Linear, _inputs, output) -> None:
        y = self._first_tensor(output)
        self.flops += 2 * y.numel() * module.in_features

    def _conv1d(self, module: nn.Conv1d, _inputs, output) -> None:
        y = self._first_tensor(output)
        kernel_macs = (module.in_channels // module.groups) * module.kernel_size[0]
        self.flops += 2 * y.numel() * kernel_macs

    def _conv_transpose1d(self, module: nn.ConvTranspose1d, inputs, _output) -> None:
        x = self._first_tensor(inputs)
        kernel_macs = (module.out_channels // module.groups) * module.kernel_size[0]
        self.flops += 2 * x.numel() * kernel_macs

    def _multihead_attention(self, module: nn.MultiheadAttention, inputs, _output) -> None:
        query = self._first_tensor(inputs)
        if module.batch_first:
            batch, tokens, embed_dim = query.shape
        else:
            tokens, batch, embed_dim = query.shape
        head_dim = embed_dim // module.num_heads
        # Q, K, V, and output projections.
        self.flops += 2 * batch * tokens * embed_dim * (4 * embed_dim)
        # QK^T and attention-value products.
        self.flops += 4 * batch * module.num_heads * tokens * tokens * head_dim

    def _transformer_layer(self, _module: nn.TransformerEncoderLayer, _inputs, output) -> None:
        y = self._first_tensor(output)
        self.flops += 16 * y.numel()

    def _association_layer(self, module: nn.Module, inputs, _output) -> None:
        # The local Anomaly Transformer adapter implements attention with
        # explicit torch.matmul calls rather than nn.MultiheadAttention.
        x = self._first_tensor(inputs)
        batch, tokens, embed_dim = x.shape
        heads = int(getattr(module, "heads"))
        head_dim = embed_dim // heads
        self.flops += 4 * batch * heads * tokens * tokens * head_dim

    def register(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                hook = self._linear
            elif isinstance(module, nn.Conv1d):
                hook = self._conv1d
            elif isinstance(module, nn.ConvTranspose1d):
                hook = self._conv_transpose1d
            elif isinstance(module, nn.MultiheadAttention):
                hook = self._multihead_attention
            elif isinstance(module, nn.TransformerEncoderLayer):
                hook = self._transformer_layer
            elif module.__class__.__name__ == "_AssociationLayer":
                hook = self._association_layer
            else:
                continue
            self._handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def profile_config(
    config_path: str | Path,
    *,
    method_label: str | None = None,
    device: str = "cpu",
    max_len: int | None = None,
) -> ComplexityResult:
    """Profile one configuration through its actual inference forward pass."""

    path = Path(config_path)
    cfg = load_config(path)
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    channels = int(data_cfg.get("channels", model_cfg.get("channels", 6)))
    sample_length = int(max_len or data_cfg.get("max_len", 48000))
    label = method_label or str((cfg.get("experiment") or {}).get("name", path.stem))

    model = build_method(cfg).to(device).eval()
    parameters, trainable = count_parameters(model)
    sample = torch.zeros(1, channels, sample_length, device=device)
    counter = _FlopCounter()
    counter.register(model)
    try:
        with torch.inference_mode():
            model.forward({"x": sample}, training=False)
    finally:
        counter.close()

    return ComplexityResult(
        method=label,
        config=path.as_posix(),
        input_shape=f"1x{channels}x{sample_length}",
        parameters=parameters,
        trainable_parameters=trainable,
        flops_per_sample=int(counter.flops),
        parameters_millions=parameters / 1e6,
        flops_giga_per_sample=counter.flops / 1e9,
    )


def profile_many(
    specs: Iterable[tuple[str, str | Path]],
    *,
    device: str = "cpu",
    max_len: int | None = None,
) -> list[ComplexityResult]:
    return [
        profile_config(path, method_label=label, device=device, max_len=max_len)
        for label, path in specs
    ]
