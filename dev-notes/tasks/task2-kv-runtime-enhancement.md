# Task 2: kv_runtime 多粒度 KV 管理增强

**状态**: 🔲 待开始  
**预计时间**: 4h  
**课题对应**: 4.2 面向国产芯片的 KV 池化与上下文缓存优化  
**可并行**: ✅ 是（与 Task 1, 3-5 并行）

---

## 背景

课题 4.2 要求：
- "按 token 段、注意力头等粒度的块级资源池"
- "HBM/主存/NVMe 三级存储"
- "冷热 KV 识别模型和分层迁移策略"
- "跨请求/批次的 KV 复用"

本任务在现有 `kv_runtime` 基础上进行增强。

---

## 工作目录

```
/home/shuhao/SAGE/packages/sage-common/src/sage/common/components/sage_llm/sageLLM/kv_runtime/
├── __init__.py
├── protocols.py         # 现有，可能需要扩展
├── blocks/              # 🆕 多粒度块管理
│   ├── __init__.py
│   └── multi_granular.py
├── hierarchy/           # 🆕 三级存储层次
│   ├── __init__.py
│   └── tiered_storage.py
├── migration/           # 🆕 冷热迁移
│   ├── __init__.py
│   └── hot_cold.py
└── reuse/               # 🆕 跨请求复用
    ├── __init__.py
    └── cross_request.py
```

---

## 参考资料

- vLLM BlockManager: https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager_v2.py
- Infinite-LLM: https://arxiv.org/abs/2401.02669 (DistKV 分层存储)
- vLLM Prefix Caching: https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html
- PagedAttention: https://arxiv.org/abs/2309.06180

---

## 任务清单

### 1. 设计多粒度 KV 块 (`blocks/multi_granular.py`)

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import time


class KVGranularity(Enum):
    """KV 块粒度
    
    传统方案只有 BLOCK 粒度（如 16 tokens），
    我们支持更细粒度的管理以提高复用率和内存效率。
    """
    BLOCK = auto()      # 块级（传统，如 16 tokens）
    TOKEN = auto()      # Token 级（最细粒度）
    HEAD = auto()       # 注意力头级
    LAYER = auto()      # 层级（最粗粒度）


class StorageTier(Enum):
    """存储层级"""
    HBM = auto()    # GPU 高带宽内存（最快，最贵）
    DDR = auto()    # CPU 主存（中等）
    NVME = auto()   # NVMe SSD（最慢，最便宜）


@dataclass
class KVBlockDescriptor:
    """KV 块描述符
    
    描述一个 KV Cache 块的元数据，不包含实际数据。
    """
    block_id: int
    granularity: KVGranularity
    
    # 位置信息
    layer_ids: List[int]          # 包含的层 ID
    head_ids: List[int]           # 包含的头 ID
    token_range: Tuple[int, int]  # Token 范围 [start, end)
    
    # 所属信息
    sequence_id: int
    request_id: str
    
    # 存储位置
    tier: StorageTier = StorageTier.HBM
    device_id: int = 0
    offset: int = 0               # 在存储中的偏移
    size_bytes: int = 0
    
    # 状态信息
    ref_count: int = 1
    is_shared: bool = False       # 是否被多个请求共享
    
    # 访问统计（用于冷热识别）
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    access_frequency: float = 0.0  # 访问频率（次/秒）
    
    # 元数据
    token_hash: Optional[str] = None  # 用于前缀匹配
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self) -> None:
        """更新访问统计"""
        now = time.time()
        elapsed = now - self.last_access_time
        if elapsed > 0:
            self.access_frequency = self.access_count / elapsed
        self.last_access_time = now
        self.access_count += 1


@dataclass
class KVPoolConfig:
    """KV 池配置"""
    # 容量配置
    hbm_capacity_bytes: int = 16 * 1024**3    # 16 GB
    ddr_capacity_bytes: int = 64 * 1024**3    # 64 GB
    nvme_capacity_bytes: int = 256 * 1024**3  # 256 GB
    
    # 块配置
    block_size: int = 16          # tokens per block
    default_granularity: KVGranularity = KVGranularity.BLOCK
    
    # 行为配置
    enable_sharing: bool = True   # 允许跨请求共享
    enable_tiering: bool = True   # 启用分层存储


class MultiGranularKVPool:
    """多粒度 KV 池
    
    支持不同粒度的 KV Cache 管理：
    - BLOCK: 传统块级，适合批量操作
    - TOKEN: Token 级，适合细粒度复用
    - HEAD: 头级，适合 MQA/GQA 优化
    - LAYER: 层级，适合 early exit
    """
    
    def __init__(self, config: KVPoolConfig):
        self.config = config
        
        # 块索引
        self._blocks: Dict[int, KVBlockDescriptor] = {}
        self._next_block_id = 0
        
        # 按序列索引
        self._sequence_blocks: Dict[int, List[int]] = {}
        
        # 按层级索引（用于快速查找）
        self._tier_blocks: Dict[StorageTier, List[int]] = {
            tier: [] for tier in StorageTier
        }
        
        # 空闲列表
        self._free_blocks: Dict[StorageTier, List[int]] = {
            tier: [] for tier in StorageTier
        }
        
        # 统计
        self._stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "total_migrations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
    
    def allocate(
        self,
        sequence_id: int,
        request_id: str,
        num_tokens: int,
        layer_ids: List[int],
        head_ids: Optional[List[int]] = None,
        granularity: Optional[KVGranularity] = None,
        preferred_tier: StorageTier = StorageTier.HBM,
    ) -> List[KVBlockDescriptor]:
        """分配 KV 块
        
        Args:
            sequence_id: 序列 ID
            request_id: 请求 ID
            num_tokens: 需要的 token 数
            layer_ids: 层 ID 列表
            head_ids: 头 ID 列表（可选，用于细粒度分配）
            granularity: 粒度（默认使用配置）
            preferred_tier: 首选存储层
            
        Returns:
            分配的 KV 块描述符列表
        """
        granularity = granularity or self.config.default_granularity
        
        # 计算需要的块数
        if granularity == KVGranularity.BLOCK:
            num_blocks = (num_tokens + self.config.block_size - 1) // self.config.block_size
        elif granularity == KVGranularity.TOKEN:
            num_blocks = num_tokens
        else:
            num_blocks = 1  # HEAD/LAYER 粒度
        
        # 分配块
        allocated = []
        for i in range(num_blocks):
            block = self._allocate_single_block(
                sequence_id=sequence_id,
                request_id=request_id,
                granularity=granularity,
                layer_ids=layer_ids,
                head_ids=head_ids or [],
                token_start=i * self.config.block_size,
                token_end=min((i + 1) * self.config.block_size, num_tokens),
                tier=preferred_tier,
            )
            allocated.append(block)
        
        self._stats["total_allocations"] += len(allocated)
        return allocated
    
    def _allocate_single_block(
        self,
        sequence_id: int,
        request_id: str,
        granularity: KVGranularity,
        layer_ids: List[int],
        head_ids: List[int],
        token_start: int,
        token_end: int,
        tier: StorageTier,
    ) -> KVBlockDescriptor:
        """分配单个块"""
        block_id = self._next_block_id
        self._next_block_id += 1
        
        block = KVBlockDescriptor(
            block_id=block_id,
            granularity=granularity,
            layer_ids=layer_ids,
            head_ids=head_ids,
            token_range=(token_start, token_end),
            sequence_id=sequence_id,
            request_id=request_id,
            tier=tier,
        )
        
        # 注册块
        self._blocks[block_id] = block
        self._tier_blocks[tier].append(block_id)
        
        if sequence_id not in self._sequence_blocks:
            self._sequence_blocks[sequence_id] = []
        self._sequence_blocks[sequence_id].append(block_id)
        
        return block
    
    def deallocate(self, blocks: List[KVBlockDescriptor]) -> None:
        """释放 KV 块"""
        for block in blocks:
            if block.ref_count > 1:
                block.ref_count -= 1
            else:
                self._free_block(block)
        
        self._stats["total_deallocations"] += len(blocks)
    
    def _free_block(self, block: KVBlockDescriptor) -> None:
        """释放单个块"""
        block_id = block.block_id
        
        # 从索引移除
        if block_id in self._blocks:
            del self._blocks[block_id]
        
        if block.sequence_id in self._sequence_blocks:
            if block_id in self._sequence_blocks[block.sequence_id]:
                self._sequence_blocks[block.sequence_id].remove(block_id)
        
        if block_id in self._tier_blocks[block.tier]:
            self._tier_blocks[block.tier].remove(block_id)
        
        # 加入空闲列表
        self._free_blocks[block.tier].append(block_id)
    
    def get_blocks_by_sequence(self, sequence_id: int) -> List[KVBlockDescriptor]:
        """获取序列的所有块"""
        block_ids = self._sequence_blocks.get(sequence_id, [])
        return [self._blocks[bid] for bid in block_ids if bid in self._blocks]
    
    def query_by_prefix(
        self,
        token_ids: List[int],
        min_match_length: int = 1,
    ) -> Optional[List[KVBlockDescriptor]]:
        """根据前缀查询可复用的 KV 块
        
        Args:
            token_ids: Token 序列
            min_match_length: 最小匹配长度
            
        Returns:
            匹配的 KV 块列表，如果没有匹配返回 None
        """
        # 这里需要与 prefix_reuse 模块集成
        # 简化实现：遍历所有块找前缀匹配
        ...
        return None
    
    def get_tier_usage(self, tier: StorageTier) -> Dict[str, Any]:
        """获取存储层使用情况"""
        blocks = self._tier_blocks[tier]
        total_bytes = sum(
            self._blocks[bid].size_bytes 
            for bid in blocks 
            if bid in self._blocks
        )
        
        capacity = {
            StorageTier.HBM: self.config.hbm_capacity_bytes,
            StorageTier.DDR: self.config.ddr_capacity_bytes,
            StorageTier.NVME: self.config.nvme_capacity_bytes,
        }[tier]
        
        return {
            "tier": tier.name,
            "num_blocks": len(blocks),
            "used_bytes": total_bytes,
            "capacity_bytes": capacity,
            "utilization": total_bytes / capacity if capacity > 0 else 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取池统计"""
        return {
            **self._stats,
            "total_blocks": len(self._blocks),
            "hbm_usage": self.get_tier_usage(StorageTier.HBM),
            "ddr_usage": self.get_tier_usage(StorageTier.DDR),
            "nvme_usage": self.get_tier_usage(StorageTier.NVME),
        }
```

### 2. 实现三级存储层次 (`hierarchy/tiered_storage.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch

from ..blocks.multi_granular import StorageTier, KVBlockDescriptor


@dataclass
class TierConfig:
    """存储层配置"""
    tier: StorageTier
    capacity_bytes: int
    bandwidth_gbps: float     # 带宽（GB/s）
    latency_us: float         # 延迟（微秒）
    
    # 可选：设备特定配置
    device_id: Optional[int] = None
    path: Optional[str] = None  # NVMe 路径


@dataclass
class TierUsage:
    """存储层使用情况"""
    tier: StorageTier
    used_bytes: int
    free_bytes: int
    capacity_bytes: int
    num_blocks: int
    
    @property
    def utilization(self) -> float:
        if self.capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes


class StorageBackend(ABC):
    """存储后端抽象"""
    
    @abstractmethod
    def read(self, offset: int, size: int) -> torch.Tensor:
        """读取数据"""
        ...
    
    @abstractmethod
    def write(self, offset: int, data: torch.Tensor) -> None:
        """写入数据"""
        ...
    
    @abstractmethod
    def get_free_space(self) -> int:
        """获取空闲空间"""
        ...


class HBMBackend(StorageBackend):
    """HBM（GPU 显存）后端"""
    
    def __init__(self, device_id: int, capacity_bytes: int):
        self.device_id = device_id
        self.capacity_bytes = capacity_bytes
        self.device = torch.device(f"cuda:{device_id}")
        
        # 预分配显存池
        self._pool: Optional[torch.Tensor] = None
        self._allocated = 0
    
    def initialize(self) -> None:
        """初始化显存池"""
        self._pool = torch.empty(
            self.capacity_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
    
    def read(self, offset: int, size: int) -> torch.Tensor:
        if self._pool is None:
            raise RuntimeError("HBM backend not initialized")
        return self._pool[offset:offset + size].clone()
    
    def write(self, offset: int, data: torch.Tensor) -> None:
        if self._pool is None:
            raise RuntimeError("HBM backend not initialized")
        self._pool[offset:offset + len(data)] = data.to(self.device).view(-1)
    
    def get_free_space(self) -> int:
        return self.capacity_bytes - self._allocated


class DDRBackend(StorageBackend):
    """DDR（CPU 主存）后端"""
    
    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self._pool: Optional[torch.Tensor] = None
        self._allocated = 0
    
    def initialize(self) -> None:
        """初始化内存池"""
        self._pool = torch.empty(
            self.capacity_bytes,
            dtype=torch.uint8,
            pin_memory=True,  # 锁页内存，加速 GPU 传输
        )
    
    def read(self, offset: int, size: int) -> torch.Tensor:
        if self._pool is None:
            raise RuntimeError("DDR backend not initialized")
        return self._pool[offset:offset + size].clone()
    
    def write(self, offset: int, data: torch.Tensor) -> None:
        if self._pool is None:
            raise RuntimeError("DDR backend not initialized")
        self._pool[offset:offset + len(data)] = data.cpu().view(-1)
    
    def get_free_space(self) -> int:
        return self.capacity_bytes - self._allocated


class NVMeBackend(StorageBackend):
    """NVMe SSD 后端"""
    
    def __init__(self, path: str, capacity_bytes: int):
        self.path = path
        self.capacity_bytes = capacity_bytes
        self._file = None
        self._allocated = 0
    
    def initialize(self) -> None:
        """初始化文件"""
        import os
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = open(self.path, "wb+")
        # 预分配文件
        self._file.truncate(self.capacity_bytes)
    
    def read(self, offset: int, size: int) -> torch.Tensor:
        if self._file is None:
            raise RuntimeError("NVMe backend not initialized")
        self._file.seek(offset)
        data = self._file.read(size)
        return torch.frombuffer(data, dtype=torch.uint8).clone()
    
    def write(self, offset: int, data: torch.Tensor) -> None:
        if self._file is None:
            raise RuntimeError("NVMe backend not initialized")
        self._file.seek(offset)
        self._file.write(data.cpu().numpy().tobytes())
    
    def get_free_space(self) -> int:
        return self.capacity_bytes - self._allocated
    
    def close(self) -> None:
        if self._file:
            self._file.close()


class TieredKVStorage:
    """三级 KV 存储管理器
    
    管理 HBM -> DDR -> NVMe 三级存储：
    - HBM: 热数据，高速访问
    - DDR: 温数据，CPU 锁页内存
    - NVMe: 冷数据，持久化存储
    """
    
    def __init__(
        self,
        hbm_config: TierConfig,
        ddr_config: TierConfig,
        nvme_config: Optional[TierConfig] = None,
    ):
        self.configs = {
            StorageTier.HBM: hbm_config,
            StorageTier.DDR: ddr_config,
        }
        if nvme_config:
            self.configs[StorageTier.NVME] = nvme_config
        
        # 初始化后端
        self.backends: Dict[StorageTier, StorageBackend] = {}
        self._init_backends()
        
        # 块位置映射
        self._block_locations: Dict[int, tuple] = {}  # block_id -> (tier, offset)
    
    def _init_backends(self) -> None:
        """初始化存储后端"""
        hbm_cfg = self.configs[StorageTier.HBM]
        self.backends[StorageTier.HBM] = HBMBackend(
            device_id=hbm_cfg.device_id or 0,
            capacity_bytes=hbm_cfg.capacity_bytes,
        )
        
        ddr_cfg = self.configs[StorageTier.DDR]
        self.backends[StorageTier.DDR] = DDRBackend(
            capacity_bytes=ddr_cfg.capacity_bytes,
        )
        
        if StorageTier.NVME in self.configs:
            nvme_cfg = self.configs[StorageTier.NVME]
            self.backends[StorageTier.NVME] = NVMeBackend(
                path=nvme_cfg.path or "/tmp/sagellm_kv_cache.bin",
                capacity_bytes=nvme_cfg.capacity_bytes,
            )
        
        # 初始化所有后端
        for backend in self.backends.values():
            backend.initialize()
    
    def get_tier_usage(self, tier: StorageTier) -> TierUsage:
        """获取存储层使用情况"""
        if tier not in self.backends:
            raise ValueError(f"Tier {tier} not configured")
        
        backend = self.backends[tier]
        config = self.configs[tier]
        free = backend.get_free_space()
        
        return TierUsage(
            tier=tier,
            used_bytes=config.capacity_bytes - free,
            free_bytes=free,
            capacity_bytes=config.capacity_bytes,
            num_blocks=sum(
                1 for loc in self._block_locations.values() 
                if loc[0] == tier
            ),
        )
    
    def read_blocks(
        self,
        blocks: List[KVBlockDescriptor],
        target_tier: StorageTier = StorageTier.HBM,
    ) -> torch.Tensor:
        """读取 KV 块到目标层
        
        如果块不在目标层，会自动迁移。
        
        Args:
            blocks: 要读取的块
            target_tier: 目标存储层
            
        Returns:
            拼接后的 KV 数据张量
        """
        data_list = []
        
        for block in blocks:
            if block.block_id not in self._block_locations:
                raise ValueError(f"Block {block.block_id} not found in storage")
            
            current_tier, offset = self._block_locations[block.block_id]
            
            # 读取数据
            data = self.backends[current_tier].read(offset, block.size_bytes)
            
            # 如果需要迁移到其他层
            if current_tier != target_tier:
                # 迁移（简化实现，实际应该异步）
                self._migrate_block(block, current_tier, target_tier)
            
            # 转移到目标设备
            if target_tier == StorageTier.HBM:
                device_id = self.configs[StorageTier.HBM].device_id or 0
                data = data.to(f"cuda:{device_id}")
            
            data_list.append(data)
            block.update_access()
        
        return torch.cat(data_list) if data_list else torch.tensor([])
    
    def write_blocks(
        self,
        data: torch.Tensor,
        blocks: List[KVBlockDescriptor],
    ) -> None:
        """写入 KV 数据
        
        Args:
            data: KV 数据张量
            blocks: 块描述符列表
        """
        offset = 0
        for block in blocks:
            block_data = data[offset:offset + block.size_bytes]
            tier = block.tier
            
            # 分配存储位置
            backend = self.backends[tier]
            storage_offset = self._allocate_space(tier, block.size_bytes)
            
            # 写入
            backend.write(storage_offset, block_data)
            
            # 记录位置
            self._block_locations[block.block_id] = (tier, storage_offset)
            
            offset += block.size_bytes
    
    def _allocate_space(self, tier: StorageTier, size: int) -> int:
        """在指定层分配空间（简化实现）"""
        # 实际实现需要更复杂的空间管理
        backend = self.backends[tier]
        free = backend.get_free_space()
        if size > free:
            raise MemoryError(f"Not enough space in {tier.name}")
        
        # 简化：顺序分配
        offset = self.configs[tier].capacity_bytes - free
        return offset
    
    def _migrate_block(
        self,
        block: KVBlockDescriptor,
        from_tier: StorageTier,
        to_tier: StorageTier,
    ) -> None:
        """迁移块到另一层"""
        if from_tier == to_tier:
            return
        
        # 读取
        _, offset = self._block_locations[block.block_id]
        data = self.backends[from_tier].read(offset, block.size_bytes)
        
        # 写入新位置
        new_offset = self._allocate_space(to_tier, block.size_bytes)
        self.backends[to_tier].write(new_offset, data)
        
        # 更新位置记录
        self._block_locations[block.block_id] = (to_tier, new_offset)
        block.tier = to_tier
    
    def get_estimated_latency(
        self,
        tier: StorageTier,
        size_bytes: int,
    ) -> float:
        """估算访问延迟（微秒）"""
        config = self.configs[tier]
        # 延迟 + 传输时间
        transfer_time_us = (size_bytes / (config.bandwidth_gbps * 1e9)) * 1e6
        return config.latency_us + transfer_time_us
```

### 3. 实现冷热识别与迁移 (`migration/hot_cold.py`)

```python
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional
import time

from ..blocks.multi_granular import KVBlockDescriptor, StorageTier
from ..hierarchy.tiered_storage import TieredKVStorage


@dataclass
class MigrationPlan:
    """迁移计划"""
    block_id: int
    from_tier: StorageTier
    to_tier: StorageTier
    priority: int = 0
    deadline_ms: Optional[float] = None


@dataclass
class MigrationResult:
    """迁移结果"""
    success: bool
    block_id: int
    from_tier: StorageTier
    to_tier: StorageTier
    duration_ms: float
    size_bytes: int


class HotColdClassifier:
    """KV 块冷热分类器
    
    基于访问频率和最近访问时间判断块的冷热程度：
    - hot: 频繁访问，应保留在 HBM
    - warm: 中等访问，可以在 DDR
    - cold: 很少访问，可以迁移到 NVMe
    """
    
    def __init__(
        self,
        hot_frequency_threshold: float = 1.0,    # 访问频率 > 1次/秒为 hot
        cold_timeout_s: float = 60.0,            # 60秒未访问为 cold
        warm_frequency_threshold: float = 0.1,   # 访问频率 < 0.1次/秒为 cold
    ):
        self.hot_frequency_threshold = hot_frequency_threshold
        self.cold_timeout_s = cold_timeout_s
        self.warm_frequency_threshold = warm_frequency_threshold
    
    def classify(
        self,
        block: KVBlockDescriptor,
    ) -> Literal["hot", "warm", "cold"]:
        """分类 KV 块
        
        Args:
            block: KV 块描述符
            
        Returns:
            "hot", "warm", 或 "cold"
        """
        now = time.time()
        time_since_access = now - block.last_access_time
        
        # 根据访问频率判断
        if block.access_frequency >= self.hot_frequency_threshold:
            return "hot"
        
        # 根据最近访问时间判断
        if time_since_access > self.cold_timeout_s:
            return "cold"
        
        # 根据低频率判断
        if block.access_frequency < self.warm_frequency_threshold:
            return "cold"
        
        return "warm"
    
    def predict_lifetime(self, block: KVBlockDescriptor) -> float:
        """预测 KV 块剩余生命周期（秒）
        
        基于访问模式预测块还会被使用多久。
        用于决定是否值得迁移。
        """
        # 简化实现：基于访问频率估算
        if block.access_frequency > 0:
            # 假设访问会持续，预测为当前频率的倒数的 10 倍
            return min(10.0 / block.access_frequency, 3600.0)
        else:
            # 无访问历史，假设短期内不会再访问
            return 0.0
    
    def get_priority_score(self, block: KVBlockDescriptor) -> float:
        """计算迁移优先级分数
        
        分数越高，越应该被迁移到更低层级。
        """
        classification = self.classify(block)
        base_score = {"hot": 0.0, "warm": 0.5, "cold": 1.0}[classification]
        
        # 调整因素
        time_since_access = time.time() - block.last_access_time
        time_factor = min(time_since_access / self.cold_timeout_s, 1.0)
        
        frequency_factor = 1.0 - min(block.access_frequency / self.hot_frequency_threshold, 1.0)
        
        return base_score * 0.4 + time_factor * 0.3 + frequency_factor * 0.3


class KVMigrator:
    """KV 块迁移器
    
    负责在存储层之间迁移 KV 块：
    - 根据冷热分类自动迁移
    - 支持批量迁移
    - 支持与计算重叠
    """
    
    def __init__(
        self,
        storage: TieredKVStorage,
        classifier: HotColdClassifier,
    ):
        self.storage = storage
        self.classifier = classifier
        
        # 统计
        self._stats = {
            "total_migrations": 0,
            "hbm_to_ddr": 0,
            "ddr_to_nvme": 0,
            "ddr_to_hbm": 0,
            "nvme_to_ddr": 0,
            "total_bytes_migrated": 0,
        }
    
    def plan_migration(
        self,
        blocks: List[KVBlockDescriptor],
        pressure: Dict[StorageTier, float],
    ) -> List[MigrationPlan]:
        """规划迁移
        
        Args:
            blocks: 所有 KV 块
            pressure: 各层压力 (0.0-1.0)，高压力层需要腾出空间
            
        Returns:
            迁移计划列表
        """
        plans = []
        
        # 1. 处理高压力层：向下迁移 cold 块
        for tier, p in pressure.items():
            if p > 0.9:  # 90% 以上需要迁移
                tier_blocks = [b for b in blocks if b.tier == tier]
                
                # 按优先级排序（高分 = 更应该迁移）
                tier_blocks.sort(
                    key=lambda b: self.classifier.get_priority_score(b),
                    reverse=True,
                )
                
                # 选择要迁移的块
                target_tier = self._get_lower_tier(tier)
                if target_tier is None:
                    continue
                
                # 迁移足够的块降到 80%
                bytes_to_free = int((p - 0.8) * self.storage.configs[tier].capacity_bytes)
                bytes_planned = 0
                
                for block in tier_blocks:
                    if bytes_planned >= bytes_to_free:
                        break
                    
                    classification = self.classifier.classify(block)
                    if classification in ("cold", "warm"):
                        plans.append(MigrationPlan(
                            block_id=block.block_id,
                            from_tier=tier,
                            to_tier=target_tier,
                            priority=int(self.classifier.get_priority_score(block) * 100),
                        ))
                        bytes_planned += block.size_bytes
        
        # 2. 处理低压力层：向上迁移 hot 块
        for tier, p in pressure.items():
            if p < 0.5:  # 50% 以下有空间
                higher_tier = self._get_higher_tier(tier)
                if higher_tier is None:
                    continue
                
                higher_blocks = [b for b in blocks if b.tier == higher_tier]
                for block in higher_blocks:
                    if self.classifier.classify(block) == "hot":
                        plans.append(MigrationPlan(
                            block_id=block.block_id,
                            from_tier=higher_tier,
                            to_tier=tier,
                            priority=90,  # Hot 块高优先级
                        ))
        
        # 按优先级排序
        plans.sort(key=lambda p: p.priority, reverse=True)
        return plans
    
    def _get_lower_tier(self, tier: StorageTier) -> Optional[StorageTier]:
        """获取更低层级"""
        if tier == StorageTier.HBM:
            return StorageTier.DDR
        elif tier == StorageTier.DDR:
            if StorageTier.NVME in self.storage.backends:
                return StorageTier.NVME
        return None
    
    def _get_higher_tier(self, tier: StorageTier) -> Optional[StorageTier]:
        """获取更高层级"""
        if tier == StorageTier.NVME:
            return StorageTier.DDR
        elif tier == StorageTier.DDR:
            return StorageTier.HBM
        return None
    
    def execute_migration(
        self,
        plan: MigrationPlan,
        block: KVBlockDescriptor,
    ) -> MigrationResult:
        """执行单个迁移
        
        Args:
            plan: 迁移计划
            block: KV 块描述符
            
        Returns:
            迁移结果
        """
        start_time = time.time()
        
        try:
            self.storage._migrate_block(block, plan.from_tier, plan.to_tier)
            success = True
        except Exception:
            success = False
        
        duration_ms = (time.time() - start_time) * 1000
        
        # 更新统计
        if success:
            self._stats["total_migrations"] += 1
            self._stats["total_bytes_migrated"] += block.size_bytes
            
            key = f"{plan.from_tier.name.lower()}_to_{plan.to_tier.name.lower()}"
            if key in self._stats:
                self._stats[key] += 1
        
        return MigrationResult(
            success=success,
            block_id=plan.block_id,
            from_tier=plan.from_tier,
            to_tier=plan.to_tier,
            duration_ms=duration_ms,
            size_bytes=block.size_bytes,
        )
    
    async def execute_migration_async(
        self,
        plans: List[MigrationPlan],
        blocks: Dict[int, KVBlockDescriptor],
        overlap_compute: bool = True,
    ) -> List[MigrationResult]:
        """异步批量执行迁移
        
        Args:
            plans: 迁移计划列表
            blocks: 块 ID 到描述符的映射
            overlap_compute: 是否与计算重叠
            
        Returns:
            迁移结果列表
        """
        results = []
        for plan in plans:
            block = blocks.get(plan.block_id)
            if block:
                result = self.execute_migration(plan, block)
                results.append(result)
        return results
    
    def get_stats(self) -> Dict:
        """获取迁移统计"""
        return self._stats.copy()
```

### 4. 实现跨请求 KV 复用 (`reuse/cross_request.py`)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import hashlib

from ..blocks.multi_granular import MultiGranularKVPool, KVBlockDescriptor


@dataclass
class ReuseResult:
    """复用结果"""
    reused: bool
    matched_blocks: List[KVBlockDescriptor]
    matched_tokens: int
    total_tokens: int
    
    @property
    def reuse_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.matched_tokens / self.total_tokens


@dataclass
class PrefixEntry:
    """前缀索引条目"""
    token_hash: str
    token_ids: List[int]
    block_ids: List[int]
    ref_count: int = 1
    tenant_id: Optional[str] = None


class CrossRequestKVCache:
    """跨请求 KV 缓存
    
    支持：
    1. 相同 prefix 的 KV 复用
    2. 多租户隔离
    3. 引用计数管理
    
    与 prefix_reuse 模块集成，提供更高层的复用接口。
    """
    
    def __init__(
        self,
        pool: MultiGranularKVPool,
        enable_tenant_isolation: bool = False,
    ):
        self.pool = pool
        self.enable_tenant_isolation = enable_tenant_isolation
        
        # 前缀索引：hash -> PrefixEntry
        self._prefix_index: Dict[str, PrefixEntry] = {}
        
        # Token 序列到 hash 的映射（加速查找）
        self._token_to_hash: Dict[tuple, str] = {}
        
        # 统计
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_reused_tokens": 0,
        }
    
    def _compute_hash(self, token_ids: List[int]) -> str:
        """计算 token 序列的 hash"""
        key = ",".join(map(str, token_ids))
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def try_reuse(
        self,
        request_id: str,
        token_ids: List[int],
        tenant_id: Optional[str] = None,
    ) -> ReuseResult:
        """尝试复用已有 KV
        
        Args:
            request_id: 请求 ID
            token_ids: Token 序列
            tenant_id: 租户 ID（用于隔离）
            
        Returns:
            复用结果
        """
        self._stats["total_queries"] += 1
        
        # 尝试找最长匹配前缀
        best_match = None
        best_length = 0
        
        # 从长到短尝试匹配
        for length in range(len(token_ids), 0, -1):
            prefix = token_ids[:length]
            prefix_tuple = tuple(prefix)
            
            # 检查缓存
            if prefix_tuple in self._token_to_hash:
                hash_key = self._token_to_hash[prefix_tuple]
                entry = self._prefix_index.get(hash_key)
                
                if entry:
                    # 检查租户隔离
                    if self.enable_tenant_isolation and entry.tenant_id != tenant_id:
                        continue
                    
                    best_match = entry
                    best_length = length
                    break
        
        if best_match is None:
            self._stats["cache_misses"] += 1
            return ReuseResult(
                reused=False,
                matched_blocks=[],
                matched_tokens=0,
                total_tokens=len(token_ids),
            )
        
        # 找到匹配
        self._stats["cache_hits"] += 1
        self._stats["total_reused_tokens"] += best_length
        
        # 增加引用计数
        best_match.ref_count += 1
        
        # 获取对应的 KV 块
        matched_blocks = [
            self.pool._blocks[bid]
            for bid in best_match.block_ids
            if bid in self.pool._blocks
        ]
        
        return ReuseResult(
            reused=True,
            matched_blocks=matched_blocks,
            matched_tokens=best_length,
            total_tokens=len(token_ids),
        )
    
    def commit(
        self,
        request_id: str,
        token_ids: List[int],
        blocks: List[KVBlockDescriptor],
        shareable: bool = True,
        tenant_id: Optional[str] = None,
    ) -> None:
        """提交新 KV 供后续复用
        
        Args:
            request_id: 请求 ID
            token_ids: Token 序列
            blocks: KV 块列表
            shareable: 是否可共享
            tenant_id: 租户 ID
        """
        if not shareable:
            return
        
        # 计算 hash
        hash_key = self._compute_hash(token_ids)
        
        # 创建索引条目
        entry = PrefixEntry(
            token_hash=hash_key,
            token_ids=token_ids.copy(),
            block_ids=[b.block_id for b in blocks],
            ref_count=1,
            tenant_id=tenant_id,
        )
        
        # 添加到索引
        self._prefix_index[hash_key] = entry
        self._token_to_hash[tuple(token_ids)] = hash_key
        
        # 标记块为共享
        for block in blocks:
            block.is_shared = True
    
    def release(
        self,
        request_id: str,
        token_ids: List[int],
    ) -> None:
        """释放复用的 KV
        
        当请求完成时调用，减少引用计数。
        """
        prefix_tuple = tuple(token_ids)
        if prefix_tuple not in self._token_to_hash:
            return
        
        hash_key = self._token_to_hash[prefix_tuple]
        entry = self._prefix_index.get(hash_key)
        
        if entry:
            entry.ref_count -= 1
            
            # 引用计数为 0 时清理
            if entry.ref_count <= 0:
                del self._prefix_index[hash_key]
                del self._token_to_hash[prefix_tuple]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = self._stats["total_queries"]
        hit_rate = self._stats["cache_hits"] / total if total > 0 else 0.0
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "index_size": len(self._prefix_index),
        }
```

---

## 单元测试要求

创建 `tests/unit/test_kv_runtime.py`：

```python
import pytest
import torch
from sageLLM.kv_runtime.blocks.multi_granular import (
    MultiGranularKVPool, KVPoolConfig, KVGranularity, StorageTier
)
from sageLLM.kv_runtime.hierarchy.tiered_storage import (
    TieredKVStorage, TierConfig
)
from sageLLM.kv_runtime.migration.hot_cold import (
    HotColdClassifier, KVMigrator
)
from sageLLM.kv_runtime.reuse.cross_request import CrossRequestKVCache


class TestMultiGranularKVPool:
    """多粒度 KV 池测试"""
    
    def test_allocate_block_granularity(self):
        """测试块粒度分配"""
        config = KVPoolConfig(block_size=16)
        pool = MultiGranularKVPool(config)
        
        blocks = pool.allocate(
            sequence_id=1,
            request_id="req_1",
            num_tokens=64,
            layer_ids=[0, 1, 2],
        )
        
        assert len(blocks) == 4  # 64 / 16 = 4 blocks
    
    def test_allocate_token_granularity(self):
        """测试 token 粒度分配"""
        config = KVPoolConfig()
        pool = MultiGranularKVPool(config)
        
        blocks = pool.allocate(
            sequence_id=1,
            request_id="req_1",
            num_tokens=10,
            layer_ids=[0],
            granularity=KVGranularity.TOKEN,
        )
        
        assert len(blocks) == 10
    
    def test_deallocate(self):
        """测试释放"""
        pool = MultiGranularKVPool(KVPoolConfig())
        blocks = pool.allocate(1, "req_1", 16, [0])
        
        assert pool.get_stats()["total_blocks"] == 1
        
        pool.deallocate(blocks)
        
        assert pool.get_stats()["total_blocks"] == 0


class TestTieredKVStorage:
    """三级存储测试"""
    
    @pytest.fixture
    def storage(self):
        return TieredKVStorage(
            hbm_config=TierConfig(
                tier=StorageTier.HBM,
                capacity_bytes=1024 * 1024,  # 1 MB
                bandwidth_gbps=900.0,
                latency_us=1.0,
                device_id=0,
            ),
            ddr_config=TierConfig(
                tier=StorageTier.DDR,
                capacity_bytes=4 * 1024 * 1024,  # 4 MB
                bandwidth_gbps=50.0,
                latency_us=100.0,
            ),
        )
    
    def test_tier_usage(self, storage):
        """测试层使用情况"""
        usage = storage.get_tier_usage(StorageTier.HBM)
        assert usage.capacity_bytes == 1024 * 1024
        assert usage.utilization == 0.0


class TestHotColdClassifier:
    """冷热分类器测试"""
    
    def test_classify_hot(self):
        """测试热块分类"""
        classifier = HotColdClassifier()
        
        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
        )
        block.access_frequency = 2.0  # 高频访问
        
        assert classifier.classify(block) == "hot"
    
    def test_classify_cold(self):
        """测试冷块分类"""
        classifier = HotColdClassifier(cold_timeout_s=1.0)
        
        block = KVBlockDescriptor(
            block_id=1,
            granularity=KVGranularity.BLOCK,
            layer_ids=[0],
            head_ids=[],
            token_range=(0, 16),
            sequence_id=1,
            request_id="req_1",
        )
        block.last_access_time = time.time() - 100  # 很久未访问
        block.access_frequency = 0.01
        
        assert classifier.classify(block) == "cold"


class TestCrossRequestKVCache:
    """跨请求缓存测试"""
    
    def test_reuse_exact_match(self):
        """测试精确匹配复用"""
        pool = MultiGranularKVPool(KVPoolConfig())
        cache = CrossRequestKVCache(pool)
        
        # 第一个请求
        token_ids = [1, 2, 3, 4, 5]
        blocks = pool.allocate(1, "req_1", len(token_ids), [0])
        cache.commit("req_1", token_ids, blocks)
        
        # 第二个请求尝试复用
        result = cache.try_reuse("req_2", token_ids)
        
        assert result.reused
        assert result.matched_tokens == 5
        assert result.reuse_ratio == 1.0
    
    def test_reuse_prefix_match(self):
        """测试前缀匹配复用"""
        pool = MultiGranularKVPool(KVPoolConfig())
        cache = CrossRequestKVCache(pool)
        
        # 提交前缀
        prefix = [1, 2, 3]
        blocks = pool.allocate(1, "req_1", len(prefix), [0])
        cache.commit("req_1", prefix, blocks)
        
        # 用更长的序列查询
        result = cache.try_reuse("req_2", [1, 2, 3, 4, 5])
        
        assert result.reused
        assert result.matched_tokens == 3
        assert result.reuse_ratio == 0.6
```

---

## 接口约定

### 输入接口

| 接口 | 来源 | 说明 |
|------|------|------|
| `PrefixIndex` | `prefix_reuse` | 前缀索引（可选集成） |
| `ScheduleOutput` | `runtime/scheduler` | 调度结果 |

### 输出接口

| 接口 | 目标 | 说明 |
|------|------|------|
| `KVBudget` | `runtime/scheduler` | KV 预算 |
| `KVMetrics` | `benchmarks` | KV 指标（命中率、迁移流量） |

---

## 验收标准

- [ ] 多粒度 KV 块支持 BLOCK/TOKEN/HEAD/LAYER 四种粒度
- [ ] 三级存储支持 HBM/DDR，NVMe 可选
- [ ] 冷热分类器准确率 > 90%（在模拟负载下）
- [ ] 跨请求复用与 prefix_reuse 模块正确集成
- [ ] 单元测试覆盖率 > 80%
- [ ] 代码通过 `ruff check` 和 `mypy`

---

## 输出物清单

```
kv_runtime/
├── __init__.py              # 更新导出
├── blocks/
│   ├── __init__.py
│   └── multi_granular.py    # ✅ 完整实现
├── hierarchy/
│   ├── __init__.py
│   └── tiered_storage.py    # ✅ 完整实现
├── migration/
│   ├── __init__.py
│   └── hot_cold.py          # ✅ 完整实现
└── reuse/
    ├── __init__.py
    └── cross_request.py     # ✅ 完整实现

tests/unit/
└── test_kv_runtime.py       # ✅ 测试文件
```
