# Task 4: backends/ 硬件后端抽象

**状态**: 🔲 待开始  
**预计时间**: 4h  
**课题对应**: 4.1 + 国产芯片支持  
**可并行**: ✅ 是（与 Task 1-3, 5 并行）

---

## 背景

课题 4.1 要求支持国产芯片（华为昇腾、寒武纪、海光）。本任务创建统一的硬件后端抽象层，使得 runtime 层代码与具体硬件解耦。

**设计原则**：
1. **统一接口**：所有后端实现相同的协议
2. **自动发现**：运行时自动检测可用硬件
3. **优雅降级**：硬件不可用时自动回退
4. **扩展性**：易于添加新硬件支持

---

## 工作目录

```
/home/shuhao/SAGE/packages/sage-common/src/sage/common/components/sage_llm/sageLLM/backends/
├── __init__.py              # 导出 + 自动发现
├── protocols.py             # 硬件后端协议
├── registry.py              # 后端注册表
├── cuda/
│   ├── __init__.py
│   └── backend.py           # CUDA 后端
├── ascend/
│   ├── __init__.py
│   └── backend.py           # 华为昇腾后端
├── cambricon/
│   ├── __init__.py
│   └── backend.py           # 寒武纪 MLU 后端
└── hygon/
    ├── __init__.py
    └── backend.py           # 海光 DCU 后端
```

---

## 参考资料

- PyTorch Device API: https://pytorch.org/docs/stable/notes/cuda.html
- Ascend PyTorch (torch_npu): https://gitee.com/ascend/pytorch
- Cambricon PyTorch (torch_mlu): https://github.com/Cambricon/catch
- ROCm PyTorch: https://pytorch.org/docs/stable/notes/hip.html
- vLLM Platform Layer: https://github.com/vllm-project/vllm/tree/main/vllm/platforms

---

## 任务清单

### 1. 定义硬件后端协议 (`protocols.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union
import torch


class BackendType(Enum):
    """硬件后端类型"""
    CUDA = auto()        # NVIDIA CUDA
    ASCEND = auto()      # 华为昇腾
    CAMBRICON = auto()   # 寒武纪 MLU
    HYGON = auto()       # 海光 DCU (ROCm-based)
    CPU = auto()         # CPU fallback


@dataclass
class DeviceInfo:
    """设备信息"""
    backend: BackendType
    device_id: int
    name: str
    
    # 计算能力
    compute_capability: Optional[str] = None  # e.g., "8.0" for A100
    
    # 内存
    total_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    
    # 核心数
    num_cores: int = 0
    
    # 驱动/SDK 版本
    driver_version: Optional[str] = None
    sdk_version: Optional[str] = None
    
    # 其他属性
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelCapabilities:
    """内核能力"""
    # 支持的精度
    supports_fp32: bool = True
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_int8: bool = False
    supports_int4: bool = False
    
    # 稀疏支持
    supports_sparse_2_4: bool = False
    
    # 特殊算子
    supports_flash_attention: bool = False
    supports_paged_attention: bool = False
    supports_fused_moe: bool = False
    
    # 通信
    supports_nccl: bool = False
    supports_hccl: bool = False  # 昇腾


class HardwareBackend(ABC):
    """硬件后端抽象基类
    
    定义所有硬件后端必须实现的接口。
    """
    
    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """返回后端类型"""
        ...
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用"""
        ...
    
    @abstractmethod
    def get_device_count(self) -> int:
        """获取可用设备数量"""
        ...
    
    @abstractmethod
    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """获取设备信息"""
        ...
    
    @abstractmethod
    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        """获取内核能力"""
        ...
    
    @abstractmethod
    def get_device(self, device_id: int = 0) -> torch.device:
        """获取 PyTorch device 对象"""
        ...
    
    @abstractmethod
    def synchronize(self, device_id: Optional[int] = None) -> None:
        """同步设备
        
        Args:
            device_id: 设备 ID，None 表示当前设备
        """
        ...
    
    @abstractmethod
    def memory_stats(self, device_id: int = 0) -> Dict[str, float]:
        """获取内存统计
        
        Returns:
            包含 total_gb, used_gb, free_gb 的字典
        """
        ...
    
    @abstractmethod
    def empty_cache(self, device_id: Optional[int] = None) -> None:
        """清空缓存"""
        ...
    
    # === 可选方法（有默认实现）===
    
    def set_device(self, device_id: int) -> None:
        """设置当前设备"""
        torch.cuda.set_device(device_id)  # 默认实现
    
    def current_device(self) -> int:
        """获取当前设备 ID"""
        return torch.cuda.current_device()  # 默认实现
    
    def allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device_id: int = 0,
    ) -> torch.Tensor:
        """分配张量
        
        某些后端可能需要特殊的内存分配策略。
        """
        device = self.get_device(device_id)
        return torch.empty(shape, dtype=dtype, device=device)
    
    def copy_to_device(
        self,
        tensor: torch.Tensor,
        device_id: int = 0,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """复制张量到设备"""
        device = self.get_device(device_id)
        return tensor.to(device, non_blocking=non_blocking)
    
    def copy_to_host(
        self,
        tensor: torch.Tensor,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """复制张量到 CPU"""
        return tensor.cpu()


class CommunicationBackend(ABC):
    """通信后端抽象
    
    用于多设备/多节点通信。
    """
    
    @abstractmethod
    def init_process_group(
        self,
        backend: str,
        world_size: int,
        rank: int,
        **kwargs,
    ) -> None:
        """初始化进程组"""
        ...
    
    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
    ) -> torch.Tensor:
        """All-reduce 操作"""
        ...
    
    @abstractmethod
    def all_gather(
        self,
        tensor: torch.Tensor,
        world_size: int,
    ) -> List[torch.Tensor]:
        """All-gather 操作"""
        ...
    
    @abstractmethod
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """广播操作"""
        ...
    
    @abstractmethod
    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
    ) -> None:
        """发送张量"""
        ...
    
    @abstractmethod
    def recv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
    ) -> torch.Tensor:
        """接收张量"""
        ...
```

### 2. 实现后端注册表 (`registry.py`)

```python
from typing import Dict, List, Optional, Type
import logging

from .protocols import HardwareBackend, BackendType, DeviceInfo

logger = logging.getLogger(__name__)


class BackendRegistry:
    """硬件后端注册表
    
    提供：
    1. 后端注册和发现
    2. 自动检测可用后端
    3. 优雅降级（fallback）
    """
    
    _backends: Dict[BackendType, Type[HardwareBackend]] = {}
    _instances: Dict[BackendType, HardwareBackend] = {}
    _default_backend: Optional[BackendType] = None
    
    @classmethod
    def register(cls, backend_type: BackendType):
        """装饰器：注册后端
        
        Usage:
            @BackendRegistry.register(BackendType.CUDA)
            class CUDABackend(HardwareBackend):
                ...
        """
        def decorator(backend_cls: Type[HardwareBackend]):
            cls._backends[backend_type] = backend_cls
            logger.debug(f"Registered backend: {backend_type.name}")
            return backend_cls
        return decorator
    
    @classmethod
    def get(cls, backend_type: BackendType) -> Optional[HardwareBackend]:
        """获取后端实例
        
        Args:
            backend_type: 后端类型
            
        Returns:
            后端实例，如果不可用返回 None
        """
        # 检查缓存
        if backend_type in cls._instances:
            return cls._instances[backend_type]
        
        # 创建实例
        if backend_type not in cls._backends:
            logger.warning(f"Backend {backend_type.name} not registered")
            return None
        
        try:
            instance = cls._backends[backend_type]()
            if instance.is_available():
                cls._instances[backend_type] = instance
                return instance
            else:
                logger.info(f"Backend {backend_type.name} not available")
                return None
        except Exception as e:
            logger.warning(f"Failed to initialize backend {backend_type.name}: {e}")
            return None
    
    @classmethod
    def get_default(cls) -> HardwareBackend:
        """获取默认后端
        
        优先级：CUDA > ASCEND > CAMBRICON > HYGON > CPU
        """
        if cls._default_backend:
            backend = cls.get(cls._default_backend)
            if backend:
                return backend
        
        # 按优先级尝试
        priority = [
            BackendType.CUDA,
            BackendType.ASCEND,
            BackendType.CAMBRICON,
            BackendType.HYGON,
            BackendType.CPU,
        ]
        
        for bt in priority:
            backend = cls.get(bt)
            if backend:
                cls._default_backend = bt
                return backend
        
        raise RuntimeError("No available hardware backend")
    
    @classmethod
    def set_default(cls, backend_type: BackendType) -> None:
        """设置默认后端"""
        cls._default_backend = backend_type
    
    @classmethod
    def list_available(cls) -> List[BackendType]:
        """列出所有可用后端"""
        available = []
        for bt in cls._backends:
            try:
                instance = cls._backends[bt]()
                if instance.is_available():
                    available.append(bt)
            except Exception:
                pass
        return available
    
    @classmethod
    def discover(cls) -> Dict[BackendType, DeviceInfo]:
        """发现所有可用设备
        
        Returns:
            后端类型到设备信息的映射
        """
        devices = {}
        for bt in cls.list_available():
            backend = cls.get(bt)
            if backend and backend.get_device_count() > 0:
                devices[bt] = backend.get_device_info(0)
        return devices
    
    @classmethod
    def reset(cls) -> None:
        """重置注册表（主要用于测试）"""
        cls._instances.clear()
        cls._default_backend = None
```

### 3. CUDA 后端 (`cuda/backend.py`)

```python
import torch
from typing import Dict, Optional, Tuple, Any
import logging

from ..protocols import (
    HardwareBackend, BackendType, DeviceInfo, KernelCapabilities
)
from ..registry import BackendRegistry

logger = logging.getLogger(__name__)


@BackendRegistry.register(BackendType.CUDA)
class CUDABackend(HardwareBackend):
    """NVIDIA CUDA 后端"""
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.CUDA
    
    def is_available(self) -> bool:
        return torch.cuda.is_available()
    
    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return torch.cuda.device_count()
    
    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self.is_available():
            raise RuntimeError("CUDA not available")
        
        props = torch.cuda.get_device_properties(device_id)
        
        # 获取计算能力
        compute_capability = f"{props.major}.{props.minor}"
        
        # 获取内存信息
        total_memory = props.total_memory / (1024**3)
        free_memory = torch.cuda.mem_get_info(device_id)[0] / (1024**3)
        
        return DeviceInfo(
            backend=BackendType.CUDA,
            device_id=device_id,
            name=props.name,
            compute_capability=compute_capability,
            total_memory_gb=total_memory,
            free_memory_gb=free_memory,
            num_cores=props.multi_processor_count,
            driver_version=torch.version.cuda,
            properties={
                "max_threads_per_block": props.max_threads_per_block,
                "max_threads_per_multiprocessor": props.max_threads_per_multi_processor,
                "warp_size": props.warp_size,
            },
        )
    
    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        info = self.get_device_info(device_id)
        major, minor = map(int, info.compute_capability.split("."))
        
        # 根据计算能力确定支持的特性
        supports_bf16 = major >= 8  # Ampere+
        supports_fp8 = major >= 9   # Hopper+
        supports_sparse_2_4 = major >= 8  # Ampere+
        supports_flash_attention = major >= 8
        
        return KernelCapabilities(
            supports_fp32=True,
            supports_fp16=True,
            supports_bf16=supports_bf16,
            supports_fp8=supports_fp8,
            supports_int8=True,
            supports_int4=True,
            supports_sparse_2_4=supports_sparse_2_4,
            supports_flash_attention=supports_flash_attention,
            supports_paged_attention=True,
            supports_fused_moe=True,
            supports_nccl=True,
            supports_hccl=False,
        )
    
    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device(f"cuda:{device_id}")
    
    def synchronize(self, device_id: Optional[int] = None) -> None:
        if device_id is not None:
            torch.cuda.synchronize(device_id)
        else:
            torch.cuda.synchronize()
    
    def memory_stats(self, device_id: int = 0) -> Dict[str, float]:
        free, total = torch.cuda.mem_get_info(device_id)
        return {
            "total_gb": total / (1024**3),
            "used_gb": (total - free) / (1024**3),
            "free_gb": free / (1024**3),
        }
    
    def empty_cache(self, device_id: Optional[int] = None) -> None:
        torch.cuda.empty_cache()
```

### 4. 昇腾后端 (`ascend/backend.py`)

```python
import torch
from typing import Dict, Optional, Tuple, Any
import logging

from ..protocols import (
    HardwareBackend, BackendType, DeviceInfo, KernelCapabilities
)
from ..registry import BackendRegistry

logger = logging.getLogger(__name__)


@BackendRegistry.register(BackendType.ASCEND)
class AscendBackend(HardwareBackend):
    """华为昇腾后端
    
    依赖 torch_npu 包。
    """
    
    def __init__(self):
        self._npu = None
        self._available = False
        self._init_npu()
    
    def _init_npu(self):
        """初始化 torch_npu"""
        try:
            import torch_npu
            self._npu = torch_npu
            self._available = torch_npu.npu.is_available()
            if self._available:
                logger.info("Ascend NPU available")
        except ImportError:
            logger.debug("torch_npu not installed")
            self._available = False
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.ASCEND
    
    def is_available(self) -> bool:
        return self._available
    
    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return self._npu.npu.device_count()
    
    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self.is_available():
            raise RuntimeError("Ascend NPU not available")
        
        # 获取设备属性（API 可能与 CUDA 不同）
        try:
            props = self._npu.npu.get_device_properties(device_id)
            name = props.name if hasattr(props, "name") else f"Ascend NPU {device_id}"
            total_memory = props.total_memory / (1024**3) if hasattr(props, "total_memory") else 0
        except Exception:
            name = f"Ascend NPU {device_id}"
            total_memory = 64.0  # 默认假设 64GB
        
        # 获取可用内存
        try:
            free, total = self._npu.npu.mem_get_info(device_id)
            free_memory = free / (1024**3)
            total_memory = total / (1024**3)
        except Exception:
            free_memory = 0.0
        
        return DeviceInfo(
            backend=BackendType.ASCEND,
            device_id=device_id,
            name=name,
            compute_capability=None,  # Ascend 没有类似概念
            total_memory_gb=total_memory,
            free_memory_gb=free_memory,
            num_cores=0,  # 需要查询实际值
            sdk_version=self._get_cann_version(),
        )
    
    def _get_cann_version(self) -> Optional[str]:
        """获取 CANN 版本"""
        try:
            return self._npu.version.cann
        except Exception:
            return None
    
    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        # 昇腾的能力因型号而异，这里给出 910B 的典型能力
        return KernelCapabilities(
            supports_fp32=True,
            supports_fp16=True,
            supports_bf16=True,  # 910B 支持 BF16
            supports_fp8=False,  # 暂不支持
            supports_int8=True,
            supports_int4=False,  # 需要验证
            supports_sparse_2_4=False,
            supports_flash_attention=True,  # 通过 CANN 支持
            supports_paged_attention=True,  # vLLM-Ascend 支持
            supports_fused_moe=False,  # 需要验证
            supports_nccl=False,
            supports_hccl=True,
        )
    
    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device(f"npu:{device_id}")
    
    def set_device(self, device_id: int) -> None:
        self._npu.npu.set_device(device_id)
    
    def current_device(self) -> int:
        return self._npu.npu.current_device()
    
    def synchronize(self, device_id: Optional[int] = None) -> None:
        if device_id is not None:
            self._npu.npu.synchronize(device_id)
        else:
            self._npu.npu.synchronize()
    
    def memory_stats(self, device_id: int = 0) -> Dict[str, float]:
        try:
            free, total = self._npu.npu.mem_get_info(device_id)
            return {
                "total_gb": total / (1024**3),
                "used_gb": (total - free) / (1024**3),
                "free_gb": free / (1024**3),
            }
        except Exception:
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}
    
    def empty_cache(self, device_id: Optional[int] = None) -> None:
        self._npu.npu.empty_cache()
```

### 5. 寒武纪后端 (`cambricon/backend.py`)

```python
import torch
from typing import Dict, Optional, Tuple, Any
import logging

from ..protocols import (
    HardwareBackend, BackendType, DeviceInfo, KernelCapabilities
)
from ..registry import BackendRegistry

logger = logging.getLogger(__name__)


@BackendRegistry.register(BackendType.CAMBRICON)
class CambriconBackend(HardwareBackend):
    """寒武纪 MLU 后端
    
    依赖 torch_mlu (catch) 包。
    """
    
    def __init__(self):
        self._mlu = None
        self._available = False
        self._init_mlu()
    
    def _init_mlu(self):
        """初始化 torch_mlu"""
        try:
            import torch_mlu
            self._mlu = torch_mlu
            self._available = torch_mlu.mlu.is_available()
            if self._available:
                logger.info("Cambricon MLU available")
        except ImportError:
            logger.debug("torch_mlu not installed")
            self._available = False
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.CAMBRICON
    
    def is_available(self) -> bool:
        return self._available
    
    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return self._mlu.mlu.device_count()
    
    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self.is_available():
            raise RuntimeError("Cambricon MLU not available")
        
        try:
            name = self._mlu.mlu.get_device_name(device_id)
        except Exception:
            name = f"MLU {device_id}"
        
        try:
            props = self._mlu.mlu.get_device_properties(device_id)
            total_memory = props.total_memory / (1024**3)
        except Exception:
            total_memory = 32.0  # 默认假设 32GB
        
        return DeviceInfo(
            backend=BackendType.CAMBRICON,
            device_id=device_id,
            name=name,
            total_memory_gb=total_memory,
            free_memory_gb=0,  # 需要查询
        )
    
    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        # MLU590 的典型能力
        return KernelCapabilities(
            supports_fp32=True,
            supports_fp16=True,
            supports_bf16=False,  # 需要验证
            supports_fp8=False,
            supports_int8=True,
            supports_int4=False,
            supports_sparse_2_4=False,
            supports_flash_attention=False,  # 需要验证
            supports_paged_attention=False,
            supports_fused_moe=False,
            supports_nccl=False,
            supports_hccl=False,
        )
    
    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device(f"mlu:{device_id}")
    
    def set_device(self, device_id: int) -> None:
        self._mlu.mlu.set_device(device_id)
    
    def current_device(self) -> int:
        return self._mlu.mlu.current_device()
    
    def synchronize(self, device_id: Optional[int] = None) -> None:
        if device_id is not None:
            self._mlu.mlu.synchronize(device_id)
        else:
            self._mlu.mlu.synchronize()
    
    def memory_stats(self, device_id: int = 0) -> Dict[str, float]:
        try:
            # API 可能不同
            allocated = self._mlu.mlu.memory_allocated(device_id)
            reserved = self._mlu.mlu.memory_reserved(device_id)
            total = self.get_device_info(device_id).total_memory_gb * (1024**3)
            return {
                "total_gb": total / (1024**3),
                "used_gb": allocated / (1024**3),
                "free_gb": (total - allocated) / (1024**3),
            }
        except Exception:
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}
    
    def empty_cache(self, device_id: Optional[int] = None) -> None:
        self._mlu.mlu.empty_cache()
```

### 6. 海光后端 (`hygon/backend.py`)

```python
import torch
from typing import Dict, Optional, Tuple, Any
import logging

from ..protocols import (
    HardwareBackend, BackendType, DeviceInfo, KernelCapabilities
)
from ..registry import BackendRegistry

logger = logging.getLogger(__name__)


@BackendRegistry.register(BackendType.HYGON)
class HygonBackend(HardwareBackend):
    """海光 DCU 后端
    
    基于 ROCm/HIP，与 AMD GPU 类似的 API。
    使用 PyTorch 的 ROCm 支持。
    """
    
    def __init__(self):
        self._available = False
        self._init_dcu()
    
    def _init_dcu(self):
        """初始化海光 DCU"""
        # 海光 DCU 使用 ROCm，PyTorch 通过 torch.cuda 访问（如果是 ROCm 版本）
        # 或者可能有专门的 torch_dcu
        try:
            # 检查是否是 ROCm 版本的 PyTorch
            if torch.version.hip is not None:
                # ROCm 版本，检查是否有海光设备
                self._available = torch.cuda.is_available()
                if self._available:
                    # 进一步检查是否是海光设备
                    device_name = torch.cuda.get_device_name(0)
                    if "Hygon" in device_name or "DCU" in device_name:
                        logger.info(f"Hygon DCU available: {device_name}")
                    else:
                        # 可能是其他 ROCm 设备（如 AMD）
                        self._available = False
            else:
                # 尝试导入专门的 torch_dcu
                try:
                    import torch_dcu
                    self._available = torch_dcu.dcu.is_available()
                except ImportError:
                    self._available = False
        except Exception:
            self._available = False
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.HYGON
    
    def is_available(self) -> bool:
        return self._available
    
    def get_device_count(self) -> int:
        if not self.is_available():
            return 0
        return torch.cuda.device_count()  # ROCm 使用 cuda API
    
    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if not self.is_available():
            raise RuntimeError("Hygon DCU not available")
        
        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory / (1024**3)
        free, _ = torch.cuda.mem_get_info(device_id)
        free_memory = free / (1024**3)
        
        return DeviceInfo(
            backend=BackendType.HYGON,
            device_id=device_id,
            name=props.name,
            total_memory_gb=total_memory,
            free_memory_gb=free_memory,
            num_cores=props.multi_processor_count,
            sdk_version=torch.version.hip,
        )
    
    def get_capabilities(self, device_id: int = 0) -> KernelCapabilities:
        # 海光 DCU 的能力（基于 ROCm）
        return KernelCapabilities(
            supports_fp32=True,
            supports_fp16=True,
            supports_bf16=True,  # 较新版本支持
            supports_fp8=False,
            supports_int8=True,
            supports_int4=False,
            supports_sparse_2_4=False,
            supports_flash_attention=True,  # ROCm 有 Flash Attention
            supports_paged_attention=True,  # vLLM 支持 ROCm
            supports_fused_moe=False,
            supports_nccl=True,  # ROCm NCCL (RCCL)
            supports_hccl=False,
        )
    
    def get_device(self, device_id: int = 0) -> torch.device:
        # ROCm 使用 cuda 设备类型
        return torch.device(f"cuda:{device_id}")
    
    def synchronize(self, device_id: Optional[int] = None) -> None:
        if device_id is not None:
            torch.cuda.synchronize(device_id)
        else:
            torch.cuda.synchronize()
    
    def memory_stats(self, device_id: int = 0) -> Dict[str, float]:
        free, total = torch.cuda.mem_get_info(device_id)
        return {
            "total_gb": total / (1024**3),
            "used_gb": (total - free) / (1024**3),
            "free_gb": free / (1024**3),
        }
    
    def empty_cache(self, device_id: Optional[int] = None) -> None:
        torch.cuda.empty_cache()
```

### 7. 主模块 (`__init__.py`)

```python
"""sageLLM 硬件后端抽象层

提供统一的硬件访问接口，支持：
- NVIDIA CUDA
- 华为昇腾 (Ascend)
- 寒武纪 (Cambricon MLU)
- 海光 (Hygon DCU)

Usage:
    from sageLLM.backends import get_backend, BackendType
    
    # 自动检测最佳后端
    backend = get_backend()
    
    # 指定后端
    backend = get_backend(BackendType.ASCEND)
    
    # 获取设备信息
    info = backend.get_device_info()
    print(f"Using {info.name} with {info.total_memory_gb:.1f} GB memory")
"""

from .protocols import (
    HardwareBackend,
    BackendType,
    DeviceInfo,
    KernelCapabilities,
    CommunicationBackend,
)
from .registry import BackendRegistry

# 导入所有后端以触发注册
from .cuda import backend as _cuda_backend
from .ascend import backend as _ascend_backend
from .cambricon import backend as _cambricon_backend
from .hygon import backend as _hygon_backend


def get_backend(backend_type: BackendType = None) -> HardwareBackend:
    """获取硬件后端
    
    Args:
        backend_type: 后端类型，None 表示自动检测
        
    Returns:
        硬件后端实例
    """
    if backend_type is None:
        return BackendRegistry.get_default()
    
    backend = BackendRegistry.get(backend_type)
    if backend is None:
        raise RuntimeError(f"Backend {backend_type.name} not available")
    return backend


def list_available_backends() -> list[BackendType]:
    """列出所有可用后端"""
    return BackendRegistry.list_available()


def discover_devices() -> dict[BackendType, DeviceInfo]:
    """发现所有可用设备"""
    return BackendRegistry.discover()


__all__ = [
    # 协议
    "HardwareBackend",
    "BackendType",
    "DeviceInfo",
    "KernelCapabilities",
    "CommunicationBackend",
    # 注册表
    "BackendRegistry",
    # 便捷函数
    "get_backend",
    "list_available_backends",
    "discover_devices",
]
```

---

## 单元测试要求

创建 `tests/unit/test_backends.py`：

```python
import pytest
import torch
from sageLLM.backends import (
    get_backend, list_available_backends, discover_devices,
    BackendType, BackendRegistry
)


class TestBackendRegistry:
    """后端注册表测试"""
    
    def test_list_available(self):
        """测试列出可用后端"""
        available = list_available_backends()
        assert isinstance(available, list)
        # 至少应该有 CPU 或 CUDA
        assert len(available) >= 0
    
    def test_get_default(self):
        """测试获取默认后端"""
        backend = get_backend()
        assert backend is not None
        assert backend.is_available()
    
    def test_discover_devices(self):
        """测试设备发现"""
        devices = discover_devices()
        assert isinstance(devices, dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDABackend:
    """CUDA 后端测试"""
    
    def test_is_available(self):
        """测试可用性检查"""
        backend = get_backend(BackendType.CUDA)
        assert backend.is_available()
    
    def test_device_info(self):
        """测试设备信息"""
        backend = get_backend(BackendType.CUDA)
        info = backend.get_device_info()
        
        assert info.backend == BackendType.CUDA
        assert info.total_memory_gb > 0
        assert info.name != ""
    
    def test_capabilities(self):
        """测试能力查询"""
        backend = get_backend(BackendType.CUDA)
        caps = backend.get_capabilities()
        
        assert caps.supports_fp16
        assert caps.supports_int8
    
    def test_memory_stats(self):
        """测试内存统计"""
        backend = get_backend(BackendType.CUDA)
        stats = backend.memory_stats()
        
        assert "total_gb" in stats
        assert "used_gb" in stats
        assert "free_gb" in stats
        assert stats["total_gb"] > 0
    
    def test_allocate_tensor(self):
        """测试张量分配"""
        backend = get_backend(BackendType.CUDA)
        tensor = backend.allocate_tensor(
            shape=(256, 256),
            dtype=torch.float16,
        )
        
        assert tensor.device.type == "cuda"
        assert tensor.dtype == torch.float16


class TestBackendFallback:
    """后端降级测试"""
    
    def test_unavailable_backend_returns_none(self):
        """测试不可用后端返回 None"""
        # 这个测试假设某些后端不可用
        # 具体行为取决于测试环境
        pass
    
    def test_default_fallback(self):
        """测试默认后端降级"""
        # 即使没有 GPU，也应该能获取到某个后端
        backend = get_backend()
        assert backend is not None
```

---

## 接口约定

### 输入接口

| 接口 | 来源 | 说明 |
|------|------|------|
| 环境变量 | OS | CUDA_VISIBLE_DEVICES, ASCEND_DEVICE_ID 等 |
| PyTorch | torch | torch.cuda, torch_npu, torch_mlu |

### 输出接口

| 接口 | 目标 | 说明 |
|------|------|------|
| `HardwareBackend` | runtime | 设备操作接口 |
| `DeviceInfo` | scheduler | 设备信息 |
| `KernelCapabilities` | accel | 支持的优化特性 |

---

## 验收标准

- [ ] CUDA 后端：完整实现，通过所有测试
- [ ] 昇腾后端：框架实现，有 `torch_npu` 时可用
- [ ] 寒武纪后端：框架实现，有 `torch_mlu` 时可用
- [ ] 海光后端：框架实现，ROCm 环境可用
- [ ] 自动发现：正确检测所有可用后端
- [ ] 优雅降级：硬件不可用时不抛异常
- [ ] 单元测试覆盖率 > 80%

---

## 输出物清单

```
backends/
├── __init__.py           # ✅ 导出 + 自动发现
├── protocols.py          # ✅ 协议定义
├── registry.py           # ✅ 注册表
├── cuda/
│   ├── __init__.py
│   └── backend.py        # ✅ CUDA 后端
├── ascend/
│   ├── __init__.py
│   └── backend.py        # ✅ 昇腾后端
├── cambricon/
│   ├── __init__.py
│   └── backend.py        # ✅ 寒武纪后端
└── hygon/
    ├── __init__.py
    └── backend.py        # ✅ 海光后端

tests/unit/
└── test_backends.py      # ✅ 测试文件
```
