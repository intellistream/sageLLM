# Task 3: accel/ 模型压缩与加速

**状态**: 🔲 待开始  
**预计时间**: 4h  
**课题对应**: 4.3 模型压缩与推理加速方法  
**可并行**: ✅ 是（与 Task 1-2, 4-5 并行）

---

## 背景

课题 4.3 要求：
- "FP8/INT4/INT8 量化"
- "结构化稀疏与剪枝"
- "混合精度推理"
- "成本模型指导的加速策略选择"

本任务创建 `accel/` 模块，提供统一的模型压缩和加速能力。

---

## 工作目录

```
/home/shuhao/SAGE/packages/sage-common/src/sage/common/components/sage_llm/sageLLM/accel/
├── __init__.py
├── quantize/                # 量化
│   ├── __init__.py
│   ├── fp8.py              # FP8 量化
│   ├── int4.py             # INT4 量化
│   └── mixed_precision.py  # 混合精度
├── sparsity/               # 稀疏
│   ├── __init__.py
│   └── structured.py       # 结构化稀疏
└── cost_model/             # 成本模型
    ├── __init__.py
    └── estimator.py        # 成本估算
```

---

## 参考资料

- vLLM Quantization: https://docs.vllm.ai/en/latest/quantization/supported_hardware.html
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- SparseGPT: https://arxiv.org/abs/2301.00774
- FP8 Training: https://arxiv.org/abs/2209.05433
- llm.c: https://github.com/karpathy/llm.c (混合精度参考)

---

## 任务清单

### 1. 量化协议定义 (`quantize/__init__.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import torch


class QuantizationType(Enum):
    """量化类型"""
    NONE = auto()       # 无量化 (FP16/BF16)
    FP8_E4M3 = auto()   # FP8 (E4M3 格式)
    FP8_E5M2 = auto()   # FP8 (E5M2 格式)
    INT8 = auto()       # INT8 对称量化
    INT4 = auto()       # INT4 分组量化
    NF4 = auto()        # NormalFloat 4-bit (QLoRA)


class QuantizationGranularity(Enum):
    """量化粒度"""
    PER_TENSOR = auto()     # 整个张量共享一个 scale
    PER_CHANNEL = auto()    # 每个通道一个 scale
    PER_GROUP = auto()      # 分组量化（如每 128 个元素）
    PER_TOKEN = auto()      # 每个 token 一个 scale（用于激活）


@dataclass
class QuantizationConfig:
    """量化配置"""
    quant_type: QuantizationType
    granularity: QuantizationGranularity = QuantizationGranularity.PER_TENSOR
    
    # 分组量化参数
    group_size: int = 128
    
    # 校准参数
    calibration_samples: int = 128
    
    # 混合精度参数
    sensitive_layers: List[str] = None  # 保持高精度的层
    
    # 算法参数
    use_symmetric: bool = True          # 对称量化
    clip_ratio: float = 1.0             # 裁剪比例
    
    def __post_init__(self):
        if self.sensitive_layers is None:
            self.sensitive_layers = []


@dataclass
class QuantizationOutput:
    """量化输出"""
    quantized_weight: torch.Tensor
    scales: torch.Tensor
    zeros: Optional[torch.Tensor] = None  # 非对称量化的零点
    group_size: int = 128
    quant_type: QuantizationType = QuantizationType.INT8


class Quantizer(ABC):
    """量化器基类"""
    
    @property
    @abstractmethod
    def quant_type(self) -> QuantizationType:
        """返回量化类型"""
        ...
    
    @abstractmethod
    def quantize(
        self,
        weight: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        """量化权重
        
        Args:
            weight: 原始权重 [out_features, in_features]
            config: 量化配置
            
        Returns:
            量化输出
        """
        ...
    
    @abstractmethod
    def dequantize(
        self,
        output: QuantizationOutput,
    ) -> torch.Tensor:
        """反量化权重
        
        Args:
            output: 量化输出
            
        Returns:
            反量化后的权重
        """
        ...


class QuantizerRegistry:
    """量化器注册表"""
    
    _quantizers: Dict[QuantizationType, type] = {}
    
    @classmethod
    def register(cls, quant_type: QuantizationType):
        """装饰器：注册量化器"""
        def decorator(quantizer_cls):
            cls._quantizers[quant_type] = quantizer_cls
            return quantizer_cls
        return decorator
    
    @classmethod
    def get(cls, quant_type: QuantizationType) -> Quantizer:
        """获取量化器实例"""
        if quant_type not in cls._quantizers:
            raise ValueError(f"Unknown quantization type: {quant_type}")
        return cls._quantizers[quant_type]()
    
    @classmethod
    def list_available(cls) -> List[QuantizationType]:
        """列出可用的量化类型"""
        return list(cls._quantizers.keys())
```

### 2. FP8 量化 (`quantize/fp8.py`)

```python
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from . import (
    Quantizer, QuantizerRegistry, QuantizationType, 
    QuantizationConfig, QuantizationOutput, QuantizationGranularity
)


@dataclass
class FP8Format:
    """FP8 格式定义"""
    name: str
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    max_value: float
    min_value: float  # 最小正值


# E4M3: 4 位指数, 3 位尾数
FP8_E4M3 = FP8Format(
    name="E4M3",
    exponent_bits=4,
    mantissa_bits=3,
    exponent_bias=7,
    max_value=448.0,      # 2^8 * (1 + 7/8)
    min_value=2**-9,      # 最小非零正值
)

# E5M2: 5 位指数, 2 位尾数
FP8_E5M2 = FP8Format(
    name="E5M2",
    exponent_bits=5,
    mantissa_bits=2,
    exponent_bias=15,
    max_value=57344.0,    # 2^15 * (1 + 3/4)
    min_value=2**-16,     # 最小非零正值
)


@QuantizerRegistry.register(QuantizationType.FP8_E4M3)
class FP8E4M3Quantizer(Quantizer):
    """FP8 E4M3 量化器
    
    E4M3 更适合权重量化：
    - 更大的动态范围
    - 更高的精度（对于 [-1, 1] 范围内的值）
    """
    
    def __init__(self):
        self.format = FP8_E4M3
    
    @property
    def quant_type(self) -> QuantizationType:
        return QuantizationType.FP8_E4M3
    
    def quantize(
        self,
        weight: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        """FP8 E4M3 量化
        
        实现步骤：
        1. 计算 scale 使得 weight/scale 在 FP8 范围内
        2. 将 weight 转换为 FP8 表示
        """
        # 计算缩放因子
        if config.granularity == QuantizationGranularity.PER_TENSOR:
            scales = self._compute_scale_per_tensor(weight)
        elif config.granularity == QuantizationGranularity.PER_CHANNEL:
            scales = self._compute_scale_per_channel(weight)
        elif config.granularity == QuantizationGranularity.PER_GROUP:
            scales = self._compute_scale_per_group(weight, config.group_size)
        else:
            raise ValueError(f"Unsupported granularity: {config.granularity}")
        
        # 缩放
        scaled_weight = weight / scales.view(-1, 1)
        
        # 裁剪到 FP8 范围
        clipped = torch.clamp(
            scaled_weight, 
            -self.format.max_value * config.clip_ratio,
            self.format.max_value * config.clip_ratio,
        )
        
        # 模拟 FP8 量化（实际硬件支持时直接转换）
        # 这里使用 round-to-nearest 模拟
        quantized = self._simulate_fp8_rounding(clipped)
        
        return QuantizationOutput(
            quantized_weight=quantized.to(torch.float16),  # 存储为 FP16（硬件不支持 FP8 时）
            scales=scales,
            quant_type=self.quant_type,
        )
    
    def _compute_scale_per_tensor(self, weight: torch.Tensor) -> torch.Tensor:
        """计算张量级 scale"""
        abs_max = weight.abs().max()
        scale = abs_max / self.format.max_value
        return scale.clamp(min=1e-8)
    
    def _compute_scale_per_channel(self, weight: torch.Tensor) -> torch.Tensor:
        """计算通道级 scale"""
        abs_max = weight.abs().amax(dim=1)
        scale = abs_max / self.format.max_value
        return scale.clamp(min=1e-8)
    
    def _compute_scale_per_group(
        self, 
        weight: torch.Tensor, 
        group_size: int,
    ) -> torch.Tensor:
        """计算分组 scale"""
        out_features, in_features = weight.shape
        num_groups = (in_features + group_size - 1) // group_size
        
        # Pad if needed
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size))
        
        # Reshape to [out_features, num_groups, group_size]
        grouped = weight.view(out_features, num_groups, group_size)
        
        # Compute scale per group
        abs_max = grouped.abs().amax(dim=-1)
        scale = abs_max / self.format.max_value
        return scale.clamp(min=1e-8)
    
    def _simulate_fp8_rounding(self, x: torch.Tensor) -> torch.Tensor:
        """模拟 FP8 舍入
        
        在没有原生 FP8 支持的硬件上，使用 FP16 模拟 FP8 精度。
        """
        # 对于 E4M3，尾数有 3 位，精度约为 1/8
        precision = 2 ** (-self.format.mantissa_bits)
        
        # Round to nearest
        rounded = torch.round(x / precision) * precision
        return rounded
    
    def dequantize(self, output: QuantizationOutput) -> torch.Tensor:
        """反量化"""
        return output.quantized_weight * output.scales.view(-1, 1)


@QuantizerRegistry.register(QuantizationType.FP8_E5M2)
class FP8E5M2Quantizer(Quantizer):
    """FP8 E5M2 量化器
    
    E5M2 更适合激活量化：
    - 更大的动态范围
    - 与 FP16 更兼容（相同的指数位数）
    """
    
    def __init__(self):
        self.format = FP8_E5M2
    
    @property
    def quant_type(self) -> QuantizationType:
        return QuantizationType.FP8_E5M2
    
    def quantize(
        self,
        weight: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        # 与 E4M3 类似，但使用 E5M2 格式
        scales = self._compute_scale_per_tensor(weight)
        scaled_weight = weight / scales
        clipped = torch.clamp(scaled_weight, -self.format.max_value, self.format.max_value)
        quantized = self._simulate_fp8_rounding(clipped)
        
        return QuantizationOutput(
            quantized_weight=quantized.to(torch.float16),
            scales=scales,
            quant_type=self.quant_type,
        )
    
    def _compute_scale_per_tensor(self, weight: torch.Tensor) -> torch.Tensor:
        abs_max = weight.abs().max()
        scale = abs_max / self.format.max_value
        return scale.clamp(min=1e-8)
    
    def _simulate_fp8_rounding(self, x: torch.Tensor) -> torch.Tensor:
        precision = 2 ** (-self.format.mantissa_bits)
        rounded = torch.round(x / precision) * precision
        return rounded
    
    def dequantize(self, output: QuantizationOutput) -> torch.Tensor:
        return output.quantized_weight * output.scales
```

### 3. INT4 量化 (`quantize/int4.py`)

```python
import torch
from typing import Optional, Tuple

from . import (
    Quantizer, QuantizerRegistry, QuantizationType,
    QuantizationConfig, QuantizationOutput, QuantizationGranularity
)


@QuantizerRegistry.register(QuantizationType.INT4)
class INT4Quantizer(Quantizer):
    """INT4 分组量化器
    
    实现 GPTQ/AWQ 风格的 INT4 量化：
    - 分组量化（默认 group_size=128）
    - 支持对称和非对称量化
    - 支持 zero-point（非对称）
    """
    
    INT4_MIN = -8
    INT4_MAX = 7
    
    @property
    def quant_type(self) -> QuantizationType:
        return QuantizationType.INT4
    
    def quantize(
        self,
        weight: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizationOutput:
        """INT4 分组量化
        
        Args:
            weight: 权重张量 [out_features, in_features]
            config: 量化配置
            
        Returns:
            量化输出，包含打包后的 INT4 权重
        """
        out_features, in_features = weight.shape
        group_size = config.group_size
        
        # 确保可以整除
        assert in_features % group_size == 0, \
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        
        num_groups = in_features // group_size
        
        # Reshape to [out_features, num_groups, group_size]
        grouped = weight.view(out_features, num_groups, group_size)
        
        if config.use_symmetric:
            # 对称量化
            scales, zeros = self._compute_symmetric_params(grouped)
            quantized = self._quantize_symmetric(grouped, scales)
        else:
            # 非对称量化
            scales, zeros = self._compute_asymmetric_params(grouped)
            quantized = self._quantize_asymmetric(grouped, scales, zeros)
        
        # 打包 INT4（2 个 INT4 打包到 1 个 INT8）
        packed = self._pack_int4(quantized)
        
        return QuantizationOutput(
            quantized_weight=packed,
            scales=scales,
            zeros=zeros,
            group_size=group_size,
            quant_type=self.quant_type,
        )
    
    def _compute_symmetric_params(
        self,
        grouped: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """计算对称量化参数
        
        Args:
            grouped: [out_features, num_groups, group_size]
            
        Returns:
            scales: [out_features, num_groups]
            zeros: None (对称量化不需要)
        """
        abs_max = grouped.abs().amax(dim=-1)
        scales = abs_max / self.INT4_MAX
        scales = scales.clamp(min=1e-8)
        return scales, None
    
    def _compute_asymmetric_params(
        self,
        grouped: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算非对称量化参数
        
        Args:
            grouped: [out_features, num_groups, group_size]
            
        Returns:
            scales: [out_features, num_groups]
            zeros: [out_features, num_groups] (零点，INT4 范围内)
        """
        min_val = grouped.amin(dim=-1)
        max_val = grouped.amax(dim=-1)
        
        # scale = (max - min) / 15 (INT4 有 16 个值)
        scales = (max_val - min_val) / 15.0
        scales = scales.clamp(min=1e-8)
        
        # zero point: round((-min) / scale)
        zeros = torch.round(-min_val / scales).clamp(0, 15).to(torch.int8)
        
        return scales, zeros
    
    def _quantize_symmetric(
        self,
        grouped: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """对称量化"""
        # [out, groups, group_size] / [out, groups, 1]
        scaled = grouped / scales.unsqueeze(-1)
        quantized = torch.round(scaled).clamp(self.INT4_MIN, self.INT4_MAX)
        return quantized.to(torch.int8)
    
    def _quantize_asymmetric(
        self,
        grouped: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
    ) -> torch.Tensor:
        """非对称量化"""
        scaled = grouped / scales.unsqueeze(-1)
        quantized = torch.round(scaled + zeros.unsqueeze(-1).float())
        quantized = quantized.clamp(0, 15)
        return quantized.to(torch.int8)
    
    def _pack_int4(self, quantized: torch.Tensor) -> torch.Tensor:
        """打包 INT4 到 INT8
        
        2 个 INT4 值打包到 1 个 INT8：
        - 低 4 位：第一个 INT4
        - 高 4 位：第二个 INT4
        """
        out_features, num_groups, group_size = quantized.shape
        assert group_size % 2 == 0
        
        # Reshape to pair up elements
        paired = quantized.view(out_features, num_groups, group_size // 2, 2)
        
        # Pack: low | (high << 4)
        # 先转换到 0-15 范围（对于对称量化需要加 8）
        low = (paired[..., 0] + 8) & 0xF
        high = (paired[..., 1] + 8) & 0xF
        
        packed = low | (high << 4)
        return packed.to(torch.uint8)
    
    def dequantize(self, output: QuantizationOutput) -> torch.Tensor:
        """反量化 INT4"""
        # 解包
        unpacked = self._unpack_int4(output.quantized_weight)
        
        # 获取形状
        scales = output.scales
        out_features, num_groups = scales.shape
        group_size = output.group_size
        
        # Reshape
        unpacked = unpacked.view(out_features, num_groups, group_size).float()
        
        # 反量化
        if output.zeros is None:
            # 对称量化
            dequantized = unpacked * scales.unsqueeze(-1)
        else:
            # 非对称量化
            dequantized = (unpacked - output.zeros.unsqueeze(-1).float()) * scales.unsqueeze(-1)
        
        # Reshape back
        return dequantized.view(out_features, -1)
    
    def _unpack_int4(self, packed: torch.Tensor) -> torch.Tensor:
        """解包 INT4"""
        # Extract low and high nibbles
        low = (packed & 0xF).to(torch.int8) - 8
        high = ((packed >> 4) & 0xF).to(torch.int8) - 8
        
        # Interleave
        unpacked = torch.stack([low, high], dim=-1).flatten(start_dim=-2)
        return unpacked
```

### 4. 混合精度推理 (`quantize/mixed_precision.py`)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import torch
import torch.nn as nn

from . import QuantizationType, QuantizationConfig


@dataclass
class LayerPrecision:
    """层精度配置"""
    layer_name: str
    weight_precision: QuantizationType
    activation_precision: QuantizationType
    
    # 是否保持高精度（用于敏感层）
    keep_high_precision: bool = False


class MixedPrecisionConfig:
    """混合精度配置
    
    支持不同层使用不同精度：
    - 敏感层（如 embedding, lm_head）使用高精度
    - 中间层使用低精度
    """
    
    # 默认敏感层模式
    DEFAULT_SENSITIVE_PATTERNS = [
        "embed",
        "lm_head",
        "norm",
        "layernorm",
    ]
    
    def __init__(
        self,
        default_weight_precision: QuantizationType = QuantizationType.INT4,
        default_activation_precision: QuantizationType = QuantizationType.FP8_E5M2,
        sensitive_patterns: Optional[List[str]] = None,
        layer_configs: Optional[Dict[str, LayerPrecision]] = None,
    ):
        self.default_weight_precision = default_weight_precision
        self.default_activation_precision = default_activation_precision
        self.sensitive_patterns = sensitive_patterns or self.DEFAULT_SENSITIVE_PATTERNS
        self.layer_configs = layer_configs or {}
    
    def get_layer_precision(self, layer_name: str) -> LayerPrecision:
        """获取层的精度配置"""
        # 检查是否有显式配置
        if layer_name in self.layer_configs:
            return self.layer_configs[layer_name]
        
        # 检查是否匹配敏感层模式
        layer_name_lower = layer_name.lower()
        for pattern in self.sensitive_patterns:
            if pattern in layer_name_lower:
                return LayerPrecision(
                    layer_name=layer_name,
                    weight_precision=QuantizationType.NONE,
                    activation_precision=QuantizationType.NONE,
                    keep_high_precision=True,
                )
        
        # 返回默认精度
        return LayerPrecision(
            layer_name=layer_name,
            weight_precision=self.default_weight_precision,
            activation_precision=self.default_activation_precision,
            keep_high_precision=False,
        )
    
    def set_layer_precision(
        self,
        layer_name: str,
        weight_precision: QuantizationType,
        activation_precision: Optional[QuantizationType] = None,
    ) -> None:
        """设置特定层的精度"""
        self.layer_configs[layer_name] = LayerPrecision(
            layer_name=layer_name,
            weight_precision=weight_precision,
            activation_precision=activation_precision or self.default_activation_precision,
        )


class MixedPrecisionQuantizer:
    """混合精度量化器
    
    对整个模型应用混合精度量化。
    """
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self._quantized_layers: Set[str] = set()
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """量化整个模型
        
        Args:
            model: 原始模型
            calibration_data: 校准数据（用于某些量化方法）
            
        Returns:
            量化后的模型
        """
        from . import QuantizerRegistry
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                precision = self.config.get_layer_precision(name)
                
                if precision.keep_high_precision:
                    continue
                
                # 获取量化器
                if precision.weight_precision != QuantizationType.NONE:
                    quantizer = QuantizerRegistry.get(precision.weight_precision)
                    
                    # 量化权重
                    quant_config = QuantizationConfig(
                        quant_type=precision.weight_precision,
                    )
                    quant_output = quantizer.quantize(module.weight.data, quant_config)
                    
                    # 替换为量化后的权重
                    # 注意：实际实现需要包装为支持量化推理的 module
                    module.weight.data = quantizer.dequantize(quant_output)
                    
                    self._quantized_layers.add(name)
        
        return model
    
    def get_quantization_summary(self) -> Dict:
        """获取量化摘要"""
        return {
            "total_quantized_layers": len(self._quantized_layers),
            "quantized_layers": list(self._quantized_layers),
            "default_weight_precision": self.config.default_weight_precision.name,
            "default_activation_precision": self.config.default_activation_precision.name,
        }
```

### 5. 结构化稀疏 (`sparsity/structured.py`)

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class SparsityPattern(Enum):
    """稀疏模式"""
    UNSTRUCTURED = auto()     # 非结构化（任意位置）
    N_M = auto()              # N:M 稀疏（如 2:4）
    BLOCK = auto()            # 块稀疏
    CHANNEL = auto()          # 通道稀疏（剪枝整个通道）
    HEAD = auto()             # 注意力头稀疏


@dataclass
class SparsityConfig:
    """稀疏配置"""
    pattern: SparsityPattern
    target_sparsity: float = 0.5     # 目标稀疏度
    
    # N:M 稀疏参数
    n: int = 2  # 每 M 个中保留 N 个
    m: int = 4
    
    # 块稀疏参数
    block_size: Tuple[int, int] = (32, 32)
    
    # 剪枝参数
    importance_metric: str = "magnitude"  # magnitude, gradient, taylor


@dataclass
class SparsityOutput:
    """稀疏输出"""
    sparse_weight: torch.Tensor
    mask: torch.Tensor
    actual_sparsity: float
    pattern: SparsityPattern


class StructuredSparseTransform:
    """结构化稀疏变换
    
    支持多种稀疏模式：
    - N:M 稀疏：NVIDIA Ampere+ 支持的 2:4 稀疏
    - 块稀疏：整块剪枝
    - 通道稀疏：剪枝整个输出通道
    """
    
    def __init__(self, config: SparsityConfig):
        self.config = config
    
    def apply(self, weight: torch.Tensor) -> SparsityOutput:
        """应用稀疏变换
        
        Args:
            weight: 权重张量 [out_features, in_features]
            
        Returns:
            稀疏输出
        """
        if self.config.pattern == SparsityPattern.N_M:
            return self._apply_n_m_sparsity(weight)
        elif self.config.pattern == SparsityPattern.BLOCK:
            return self._apply_block_sparsity(weight)
        elif self.config.pattern == SparsityPattern.CHANNEL:
            return self._apply_channel_sparsity(weight)
        else:
            return self._apply_unstructured_sparsity(weight)
    
    def _apply_n_m_sparsity(self, weight: torch.Tensor) -> SparsityOutput:
        """应用 N:M 稀疏
        
        保留每 M 个元素中绝对值最大的 N 个。
        """
        n, m = self.config.n, self.config.m
        out_features, in_features = weight.shape
        
        # 确保 in_features 可被 m 整除
        assert in_features % m == 0, f"in_features ({in_features}) must be divisible by m ({m})"
        
        # Reshape to [out_features, num_groups, m]
        grouped = weight.view(out_features, -1, m)
        
        # 获取每组中绝对值最大的 n 个位置
        _, indices = torch.topk(grouped.abs(), k=n, dim=-1)
        
        # 创建 mask
        mask = torch.zeros_like(grouped)
        mask.scatter_(-1, indices, 1.0)
        mask = mask.view(out_features, in_features)
        
        # 应用 mask
        sparse_weight = weight * mask
        
        actual_sparsity = 1.0 - (mask.sum() / mask.numel()).item()
        
        return SparsityOutput(
            sparse_weight=sparse_weight,
            mask=mask,
            actual_sparsity=actual_sparsity,
            pattern=SparsityPattern.N_M,
        )
    
    def _apply_block_sparsity(self, weight: torch.Tensor) -> SparsityOutput:
        """应用块稀疏
        
        以固定大小的块为单位进行剪枝。
        """
        block_h, block_w = self.config.block_size
        out_features, in_features = weight.shape
        
        # 计算块数量
        num_blocks_h = out_features // block_h
        num_blocks_w = in_features // block_w
        total_blocks = num_blocks_h * num_blocks_w
        
        # 计算每个块的重要性（使用 L2 范数）
        block_importance = torch.zeros(num_blocks_h, num_blocks_w)
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = weight[
                    i*block_h:(i+1)*block_h,
                    j*block_w:(j+1)*block_w
                ]
                block_importance[i, j] = block.norm()
        
        # 确定要保留的块数量
        num_keep = int(total_blocks * (1 - self.config.target_sparsity))
        
        # 获取最重要的块
        flat_importance = block_importance.view(-1)
        _, top_indices = torch.topk(flat_importance, k=num_keep)
        
        # 创建 mask
        mask = torch.zeros(out_features, in_features)
        for idx in top_indices:
            i = idx // num_blocks_w
            j = idx % num_blocks_w
            mask[
                i*block_h:(i+1)*block_h,
                j*block_w:(j+1)*block_w
            ] = 1.0
        
        sparse_weight = weight * mask
        actual_sparsity = 1.0 - (mask.sum() / mask.numel()).item()
        
        return SparsityOutput(
            sparse_weight=sparse_weight,
            mask=mask,
            actual_sparsity=actual_sparsity,
            pattern=SparsityPattern.BLOCK,
        )
    
    def _apply_channel_sparsity(self, weight: torch.Tensor) -> SparsityOutput:
        """应用通道稀疏
        
        剪枝整个输出通道。
        """
        out_features, in_features = weight.shape
        
        # 计算每个输出通道的重要性
        channel_importance = weight.abs().sum(dim=1)
        
        # 确定要保留的通道数量
        num_keep = int(out_features * (1 - self.config.target_sparsity))
        
        # 获取最重要的通道
        _, top_indices = torch.topk(channel_importance, k=num_keep)
        
        # 创建 mask
        mask = torch.zeros(out_features, 1)
        mask[top_indices] = 1.0
        mask = mask.expand(-1, in_features)
        
        sparse_weight = weight * mask
        actual_sparsity = 1.0 - (mask.sum() / mask.numel()).item()
        
        return SparsityOutput(
            sparse_weight=sparse_weight,
            mask=mask,
            actual_sparsity=actual_sparsity,
            pattern=SparsityPattern.CHANNEL,
        )
    
    def _apply_unstructured_sparsity(self, weight: torch.Tensor) -> SparsityOutput:
        """应用非结构化稀疏"""
        # 计算阈值
        flat = weight.abs().view(-1)
        k = int(flat.numel() * self.config.target_sparsity)
        threshold = torch.kthvalue(flat, k).values
        
        # 创建 mask
        mask = (weight.abs() >= threshold).float()
        
        sparse_weight = weight * mask
        actual_sparsity = 1.0 - (mask.sum() / mask.numel()).item()
        
        return SparsityOutput(
            sparse_weight=sparse_weight,
            mask=mask,
            actual_sparsity=actual_sparsity,
            pattern=SparsityPattern.UNSTRUCTURED,
        )
```

### 6. 成本模型 (`cost_model/estimator.py`)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum, auto

from ..quantize import QuantizationType
from ..sparsity.structured import SparsityPattern


class AcceleratorType(Enum):
    """加速器类型"""
    NVIDIA_A100 = auto()
    NVIDIA_H100 = auto()
    HUAWEI_ASCEND_910B = auto()
    CAMBRICON_MLU590 = auto()
    HYGON_DCU = auto()


@dataclass
class AcceleratorSpec:
    """加速器规格"""
    name: str
    type: AcceleratorType
    
    # 计算能力
    fp32_tflops: float
    fp16_tflops: float
    bf16_tflops: float
    int8_tops: float
    fp8_tflops: Optional[float] = None
    
    # 内存
    hbm_gb: float
    hbm_bandwidth_gbps: float
    
    # 稀疏支持
    supports_2_4_sparsity: bool = False
    sparse_speedup: float = 1.0  # 2:4 稀疏加速比
    
    # 量化支持
    supported_quant_types: List[QuantizationType] = None
    
    def __post_init__(self):
        if self.supported_quant_types is None:
            self.supported_quant_types = [
                QuantizationType.NONE,
                QuantizationType.INT8,
            ]


# 预定义加速器规格
ACCELERATOR_SPECS = {
    AcceleratorType.NVIDIA_A100: AcceleratorSpec(
        name="NVIDIA A100 80GB",
        type=AcceleratorType.NVIDIA_A100,
        fp32_tflops=19.5,
        fp16_tflops=312,
        bf16_tflops=312,
        int8_tops=624,
        fp8_tflops=None,
        hbm_gb=80,
        hbm_bandwidth_gbps=2039,
        supports_2_4_sparsity=True,
        sparse_speedup=2.0,
        supported_quant_types=[
            QuantizationType.NONE,
            QuantizationType.INT8,
            QuantizationType.INT4,
        ],
    ),
    AcceleratorType.NVIDIA_H100: AcceleratorSpec(
        name="NVIDIA H100 80GB",
        type=AcceleratorType.NVIDIA_H100,
        fp32_tflops=67,
        fp16_tflops=990,
        bf16_tflops=990,
        int8_tops=1980,
        fp8_tflops=1980,
        hbm_gb=80,
        hbm_bandwidth_gbps=3350,
        supports_2_4_sparsity=True,
        sparse_speedup=2.0,
        supported_quant_types=[
            QuantizationType.NONE,
            QuantizationType.FP8_E4M3,
            QuantizationType.FP8_E5M2,
            QuantizationType.INT8,
            QuantizationType.INT4,
        ],
    ),
    AcceleratorType.HUAWEI_ASCEND_910B: AcceleratorSpec(
        name="Huawei Ascend 910B",
        type=AcceleratorType.HUAWEI_ASCEND_910B,
        fp32_tflops=8,  # 估计值
        fp16_tflops=320,
        bf16_tflops=320,
        int8_tops=640,
        hbm_gb=64,
        hbm_bandwidth_gbps=1200,
        supports_2_4_sparsity=False,
        supported_quant_types=[
            QuantizationType.NONE,
            QuantizationType.INT8,
        ],
    ),
}


@dataclass
class InferenceCost:
    """推理成本估算"""
    # 时间成本
    compute_time_ms: float
    memory_time_ms: float
    total_time_ms: float
    
    # 资源利用
    compute_utilization: float  # 计算利用率 (0-1)
    memory_bandwidth_utilization: float  # 带宽利用率 (0-1)
    
    # 内存占用
    weight_memory_mb: float
    activation_memory_mb: float
    kv_cache_memory_mb: float
    total_memory_mb: float
    
    # 吞吐量
    tokens_per_second: float


class CostEstimator:
    """成本估算器
    
    估算不同配置下的推理成本，用于指导优化策略选择。
    """
    
    def __init__(self, accelerator: AcceleratorSpec):
        self.accelerator = accelerator
    
    def estimate_linear_layer(
        self,
        in_features: int,
        out_features: int,
        batch_size: int,
        seq_len: int,
        quant_type: QuantizationType = QuantizationType.NONE,
        sparsity: float = 0.0,
        sparsity_pattern: Optional[SparsityPattern] = None,
    ) -> InferenceCost:
        """估算线性层推理成本
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            batch_size: 批次大小
            seq_len: 序列长度
            quant_type: 量化类型
            sparsity: 稀疏度 (0-1)
            sparsity_pattern: 稀疏模式
            
        Returns:
            推理成本估算
        """
        # 计算 FLOPs
        total_tokens = batch_size * seq_len
        flops = 2 * total_tokens * in_features * out_features
        
        # 应用稀疏加速
        effective_sparsity = sparsity
        if sparsity_pattern == SparsityPattern.N_M and self.accelerator.supports_2_4_sparsity:
            flops = flops * (1 - sparsity) / self.accelerator.sparse_speedup
        else:
            flops = flops * (1 - sparsity)
        
        # 获取计算吞吐量
        compute_tflops = self._get_compute_tflops(quant_type)
        
        # 计算时间
        compute_time_ms = (flops / (compute_tflops * 1e12)) * 1000
        
        # 内存占用
        weight_bytes = self._get_weight_bytes(in_features, out_features, quant_type)
        weight_memory_mb = weight_bytes / (1024 ** 2)
        
        # 激活内存
        activation_bytes = total_tokens * (in_features + out_features) * 2  # FP16
        activation_memory_mb = activation_bytes / (1024 ** 2)
        
        # 内存访问时间
        total_memory_bytes = weight_bytes + activation_bytes
        memory_time_ms = (total_memory_bytes / (self.accelerator.hbm_bandwidth_gbps * 1e9)) * 1000
        
        # 总时间（取计算和内存的最大值，因为可能重叠）
        total_time_ms = max(compute_time_ms, memory_time_ms)
        
        # 利用率
        compute_utilization = compute_time_ms / total_time_ms if total_time_ms > 0 else 0
        memory_bandwidth_utilization = memory_time_ms / total_time_ms if total_time_ms > 0 else 0
        
        return InferenceCost(
            compute_time_ms=compute_time_ms,
            memory_time_ms=memory_time_ms,
            total_time_ms=total_time_ms,
            compute_utilization=compute_utilization,
            memory_bandwidth_utilization=memory_bandwidth_utilization,
            weight_memory_mb=weight_memory_mb,
            activation_memory_mb=activation_memory_mb,
            kv_cache_memory_mb=0,  # 线性层不涉及 KV cache
            total_memory_mb=weight_memory_mb + activation_memory_mb,
            tokens_per_second=(total_tokens / total_time_ms * 1000) if total_time_ms > 0 else 0,
        )
    
    def _get_compute_tflops(self, quant_type: QuantizationType) -> float:
        """获取指定量化类型的计算吞吐量"""
        if quant_type == QuantizationType.NONE:
            return self.accelerator.fp16_tflops
        elif quant_type in (QuantizationType.FP8_E4M3, QuantizationType.FP8_E5M2):
            return self.accelerator.fp8_tflops or self.accelerator.fp16_tflops
        elif quant_type in (QuantizationType.INT8, QuantizationType.INT4):
            return self.accelerator.int8_tops
        else:
            return self.accelerator.fp16_tflops
    
    def _get_weight_bytes(
        self,
        in_features: int,
        out_features: int,
        quant_type: QuantizationType,
    ) -> int:
        """计算权重内存占用"""
        num_elements = in_features * out_features
        
        bytes_per_element = {
            QuantizationType.NONE: 2,      # FP16
            QuantizationType.FP8_E4M3: 1,
            QuantizationType.FP8_E5M2: 1,
            QuantizationType.INT8: 1,
            QuantizationType.INT4: 0.5,
            QuantizationType.NF4: 0.5,
        }.get(quant_type, 2)
        
        return int(num_elements * bytes_per_element)
    
    def recommend_configuration(
        self,
        model_size_params: int,
        max_batch_size: int,
        max_seq_len: int,
        target_latency_ms: Optional[float] = None,
        target_throughput_tps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """推荐最佳配置
        
        根据目标延迟/吞吐量推荐量化和稀疏配置。
        """
        recommendations = {
            "accelerator": self.accelerator.name,
            "supported_quant_types": [q.name for q in self.accelerator.supported_quant_types],
            "supports_2_4_sparsity": self.accelerator.supports_2_4_sparsity,
        }
        
        # 检查是否支持 FP8
        if QuantizationType.FP8_E4M3 in self.accelerator.supported_quant_types:
            recommendations["recommended_weight_quant"] = "FP8_E4M3"
            recommendations["recommended_activation_quant"] = "FP8_E5M2"
        elif QuantizationType.INT4 in self.accelerator.supported_quant_types:
            recommendations["recommended_weight_quant"] = "INT4"
            recommendations["recommended_activation_quant"] = "INT8"
        else:
            recommendations["recommended_weight_quant"] = "INT8"
            recommendations["recommended_activation_quant"] = "NONE"
        
        # 检查稀疏
        if self.accelerator.supports_2_4_sparsity:
            recommendations["recommended_sparsity"] = "N_M (2:4)"
            recommendations["expected_speedup"] = "~2x"
        else:
            recommendations["recommended_sparsity"] = "NONE"
        
        return recommendations
```

---

## 单元测试要求

创建 `tests/unit/test_accel.py`：

```python
import pytest
import torch
from sageLLM.accel.quantize import (
    QuantizerRegistry, QuantizationType, QuantizationConfig,
    QuantizationGranularity
)
from sageLLM.accel.quantize.fp8 import FP8E4M3Quantizer
from sageLLM.accel.quantize.int4 import INT4Quantizer
from sageLLM.accel.sparsity.structured import (
    StructuredSparseTransform, SparsityConfig, SparsityPattern
)
from sageLLM.accel.cost_model.estimator import (
    CostEstimator, ACCELERATOR_SPECS, AcceleratorType
)


class TestFP8Quantizer:
    """FP8 量化测试"""
    
    def test_fp8_e4m3_quantize_dequantize(self):
        """测试 FP8 E4M3 量化/反量化"""
        quantizer = FP8E4M3Quantizer()
        weight = torch.randn(256, 512)
        
        config = QuantizationConfig(
            quant_type=QuantizationType.FP8_E4M3,
            granularity=QuantizationGranularity.PER_TENSOR,
        )
        
        output = quantizer.quantize(weight, config)
        reconstructed = quantizer.dequantize(output)
        
        # 检查重建误差
        error = (weight - reconstructed).abs().mean()
        assert error < 0.1  # 允许一定误差


class TestINT4Quantizer:
    """INT4 量化测试"""
    
    def test_int4_symmetric_quantize(self):
        """测试 INT4 对称量化"""
        quantizer = INT4Quantizer()
        weight = torch.randn(256, 512)
        
        config = QuantizationConfig(
            quant_type=QuantizationType.INT4,
            group_size=128,
            use_symmetric=True,
        )
        
        output = quantizer.quantize(weight, config)
        
        assert output.quantized_weight.dtype == torch.uint8
        assert output.zeros is None  # 对称量化无 zero point
    
    def test_int4_asymmetric_quantize(self):
        """测试 INT4 非对称量化"""
        quantizer = INT4Quantizer()
        weight = torch.randn(256, 512) + 1.0  # 非对称分布
        
        config = QuantizationConfig(
            quant_type=QuantizationType.INT4,
            group_size=128,
            use_symmetric=False,
        )
        
        output = quantizer.quantize(weight, config)
        
        assert output.zeros is not None


class TestStructuredSparsity:
    """结构化稀疏测试"""
    
    def test_2_4_sparsity(self):
        """测试 2:4 稀疏"""
        config = SparsityConfig(
            pattern=SparsityPattern.N_M,
            n=2,
            m=4,
        )
        transform = StructuredSparseTransform(config)
        
        weight = torch.randn(256, 512)
        output = transform.apply(weight)
        
        # 检查稀疏度约为 50%
        assert abs(output.actual_sparsity - 0.5) < 0.01
    
    def test_block_sparsity(self):
        """测试块稀疏"""
        config = SparsityConfig(
            pattern=SparsityPattern.BLOCK,
            target_sparsity=0.75,
            block_size=(32, 32),
        )
        transform = StructuredSparseTransform(config)
        
        weight = torch.randn(256, 512)
        output = transform.apply(weight)
        
        # 检查稀疏度接近目标
        assert abs(output.actual_sparsity - 0.75) < 0.1


class TestCostEstimator:
    """成本估算测试"""
    
    def test_h100_linear_cost(self):
        """测试 H100 线性层成本估算"""
        spec = ACCELERATOR_SPECS[AcceleratorType.NVIDIA_H100]
        estimator = CostEstimator(spec)
        
        cost = estimator.estimate_linear_layer(
            in_features=4096,
            out_features=4096,
            batch_size=32,
            seq_len=1024,
            quant_type=QuantizationType.NONE,
        )
        
        assert cost.total_time_ms > 0
        assert cost.tokens_per_second > 0
    
    def test_recommendation(self):
        """测试配置推荐"""
        spec = ACCELERATOR_SPECS[AcceleratorType.NVIDIA_H100]
        estimator = CostEstimator(spec)
        
        rec = estimator.recommend_configuration(
            model_size_params=7_000_000_000,
            max_batch_size=32,
            max_seq_len=4096,
        )
        
        assert "FP8" in rec["recommended_weight_quant"]
```

---

## 接口约定

### 输入接口

| 接口 | 来源 | 说明 |
|------|------|------|
| `torch.Tensor` | 模型权重 | 原始 FP16/BF16 权重 |
| `AcceleratorSpec` | backends | 硬件规格 |

### 输出接口

| 接口 | 目标 | 说明 |
|------|------|------|
| `QuantizationOutput` | runtime | 量化后的权重 |
| `SparsityOutput` | runtime | 稀疏权重+mask |
| `InferenceCost` | scheduler | 成本估算结果 |

---

## 验收标准

- [ ] FP8 量化：E4M3/E5M2 格式正确实现
- [ ] INT4 量化：对称/非对称量化误差 < 5%（余弦相似度 > 0.95）
- [ ] 2:4 稀疏：精确实现 50% 稀疏度
- [ ] 成本模型：估算误差 < 20%（与实际测量比较）
- [ ] 单元测试覆盖率 > 80%
- [ ] 代码通过 `ruff check` 和 `mypy`

---

## 输出物清单

```
accel/
├── __init__.py
├── quantize/
│   ├── __init__.py           # ✅ 协议定义
│   ├── fp8.py                # ✅ 完整实现
│   ├── int4.py               # ✅ 完整实现
│   └── mixed_precision.py    # ✅ 完整实现
├── sparsity/
│   ├── __init__.py
│   └── structured.py         # ✅ 完整实现
└── cost_model/
    ├── __init__.py
    └── estimator.py          # ✅ 完整实现

tests/unit/
└── test_accel.py             # ✅ 测试文件
```
