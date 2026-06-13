"""Reference quantization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


QuantDType = Literal["int8", "int4", "fp8"]


@dataclass(frozen=True)
class QuantizedTensor:
    values: Any
    scale: Any
    zero_point: Any | None
    dtype: QuantDType
    axis: int | None
    original_shape: tuple[int, ...]
    original_nbytes: int

    @property
    def quantized_nbytes(self) -> int:
        element_count = 1
        for dim in self.original_shape:
            element_count *= dim
        if self.dtype == "int4":
            value_bytes = (element_count + 1) // 2
        else:
            value_bytes = element_count
        scale_bytes = int(self.scale.numel() * self.scale.element_size())
        zero_bytes = (
            0
            if self.zero_point is None
            else int(self.zero_point.numel() * self.zero_point.element_size())
        )
        return value_bytes + scale_bytes + zero_bytes

    @property
    def memory_saving_ratio(self) -> float:
        return 1.0 - (self.quantized_nbytes / self.original_nbytes)

    def to_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "axis": self.axis,
            "original_shape": list(self.original_shape),
            "original_nbytes": self.original_nbytes,
            "quantized_nbytes": self.quantized_nbytes,
            "memory_saving_ratio": self.memory_saving_ratio,
        }


class WeightQuantizer:
    def __init__(self, *, dtype: QuantDType = "int8", axis: int = 1) -> None:
        if dtype not in {"int8", "int4"}:
            raise ValueError(f"unsupported quant dtype: {dtype}")
        if axis not in {0, 1}:
            raise ValueError("axis must be 0 or 1")
        self.dtype = dtype
        self.axis = axis

    def quantize(self, tensor: Any) -> QuantizedTensor:
        import torch

        values = torch.as_tensor(tensor).float()
        qmax = 127 if self.dtype == "int8" else 7
        qmin = -128 if self.dtype == "int8" else -8
        reduce_dim = self.axis
        max_abs = values.abs().amax(dim=reduce_dim, keepdim=True).clamp_min(1e-8)
        scale = max_abs / qmax
        quantized = torch.round(values / scale).clamp(qmin, qmax).to(torch.int8)
        return QuantizedTensor(
            values=quantized,
            scale=scale,
            zero_point=None,
            dtype=self.dtype,
            axis=self.axis,
            original_shape=tuple(values.shape),
            original_nbytes=int(values.numel() * values.element_size()),
        )

    def dequantize(self, tensor: QuantizedTensor) -> Any:
        if tensor.dtype != self.dtype:
            raise ValueError(f"expected {self.dtype}, got {tensor.dtype}")
        return tensor.values.float() * tensor.scale


class KVQuantizer:
    def __init__(self, *, dtype: Literal["int8", "fp8"] = "int8") -> None:
        if dtype not in {"int8", "fp8"}:
            raise ValueError(f"unsupported KV quant dtype: {dtype}")
        self.dtype = dtype

    def quantize(self, tensor: Any) -> QuantizedTensor:
        import torch

        values = torch.as_tensor(tensor).float()
        if self.dtype == "fp8":
            max_abs = values.abs().max().clamp_min(1e-8)
            scale = max_abs / 127.0
            quantized = torch.round(values / scale).clamp(-127, 127).to(torch.int8)
            return QuantizedTensor(
                values=quantized,
                scale=scale.reshape(()),
                zero_point=None,
                dtype="fp8",
                axis=None,
                original_shape=tuple(values.shape),
                original_nbytes=int(values.numel() * values.element_size()),
            )

        min_value = values.min()
        max_value = values.max()
        scale = ((max_value - min_value) / 255.0).clamp_min(1e-8)
        zero_point = torch.round(-min_value / scale).clamp(0, 255).to(torch.uint8)
        quantized = torch.round(values / scale + zero_point.float()).clamp(0, 255)
        return QuantizedTensor(
            values=quantized.to(torch.uint8),
            scale=scale.reshape(()),
            zero_point=zero_point.reshape(()),
            dtype="int8",
            axis=None,
            original_shape=tuple(values.shape),
            original_nbytes=int(values.numel() * values.element_size()),
        )

    def dequantize(self, tensor: QuantizedTensor) -> Any:
        if tensor.dtype == "fp8":
            return tensor.values.float() * tensor.scale
        if tensor.zero_point is None:
            raise ValueError("KV quantized tensor requires a zero point")
        return (tensor.values.float() - tensor.zero_point.float()) * tensor.scale
