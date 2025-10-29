# 拓扑感知配置文档

本文档详细介绍 sageLLM Control Plane 的拓扑感知功能，包括GPU拓扑检测、跨机调度优化、部署场景和配置指南。

## 目录

- [什么是拓扑感知](#什么是拓扑感知)
- [拓扑信息字段](#拓扑信息字段)
- [自动拓扑检测](#自动拓扑检测)
- [拓扑感知路由](#拓扑感知路由)
- [部署场景](#部署场景)
- [配置指南](#配置指南)
- [性能优化](#性能优化)
- [故障排查](#故障排查)

## 什么是拓扑感知

### 为什么需要拓扑感知

在多机多卡环境中，不同GPU/实例之间的通信性能差异巨大：

| 连接类型 | 带宽 | 延迟 | 适用场景 |
|---------|------|------|---------|
| **NVLINK** | 600 GB/s | < 1μs | 同GPU卡间通信 |
| **PCIe 4.0** | 64 GB/s | 1-5μs | 同机器内跨NUMA通信 |
| **10GbE网络** | 1.25 GB/s | 100-500μs | 跨机器通信 |
| **1GbE网络** | 125 MB/s | 1-5ms | 远程数据中心 |

**拓扑感知的核心目标**：
- 优先调度到**同机器**实例（避免网络延迟）
- 优先调度到**NVLINK连接**的实例（共享KV缓存）
- 考虑**NUMA亲和性**（避免跨NUMA访问内存）
- 感知**机架/区域**位置（容灾和故障隔离）

### 性能提升

启用拓扑感知后的性能提升（基于内部测试）：

```
同机 NVLINK 实例:
  - 延迟降低: 50-80%
  - 吞吐提升: 30-50%
  - KV缓存命中率: +60%

同机非 NVLINK 实例:
  - 延迟降低: 20-40%
  - 吞吐提升: 10-20%

跨机实例（需避免）:
  - 延迟增加: 100-500%
  - 带宽受限: 10-100x
```

## 拓扑信息字段

### ExecutionInstance 拓扑字段

```python
from control_plane.types import ExecutionInstance

instance = ExecutionInstance(
    # 基础字段
    instance_id="gpu-node1-0",
    host="192.168.1.10",
    port=8000,
    model_name="llama-2-7b",
    
    # 拓扑字段
    machine_id="node1",              # 机器唯一标识
    rack_id="rack-a",                # 机架标识（可选）
    datacenter_id="dc-us-west",      # 数据中心（可选，暂未使用）
    
    # GPU 拓扑
    gpu_bus_id="0000:17:00.0",       # GPU PCI 总线 ID
    gpu_device_id=0,                 # GPU 设备编号（0-7）
    nvlink_peers=["gpu-node1-1", "gpu-node1-2"],  # NVLINK 连接的其他实例
    
    # NUMA 拓扑
    numa_node=0,                     # NUMA 节点编号
    
    # 网络拓扑（可选）
    network_bandwidth_gbps=100,      # 网络带宽（Gbps）
    network_latency_ms=0.5,          # 网络延迟（毫秒）
    
    # 共享资源（可选）
    shared_memory_pool="node1-shm",  # 共享内存池标识
    shared_storage_path="/mnt/nvme/node1",  # 共享存储路径
)
```

### 字段说明

#### 1. machine_id（机器标识）

**用途**: 标识物理机器

**格式**: 字符串，建议使用主机名或UUID

**示例**:
```python
machine_id = "gpu-server-01"
machine_id = socket.gethostname()  # 自动获取
machine_id = "node-" + os.environ.get("SLURM_NODEID")  # Slurm 集群
```

**重要性**: ⭐⭐⭐⭐⭐（最重要的拓扑字段）

#### 2. rack_id（机架标识）

**用途**: 标识物理机架（用于容灾）

**格式**: 字符串

**示例**:
```python
rack_id = "rack-a-01"
rack_id = f"rack-{int(machine_id.split('-')[1]) // 4}"  # 每4台机器一个机架
```

**重要性**: ⭐⭐（可选，用于高可用部署）

#### 3. gpu_bus_id（GPU总线ID）

**用途**: 唯一标识GPU设备

**格式**: PCI总线地址（"domain:bus:device.function"）

**获取方法**:
```bash
nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader
# 输出: 0000:17:00.0
```

**Python获取**:
```python
from control_plane.topology import TopologyDetector

detector = TopologyDetector()
bus_id = detector.get_gpu_bus_id(gpu_device_id=0)
```

**重要性**: ⭐⭐⭐（用于NVLINK检测）

#### 4. nvlink_peers（NVLINK对等实例）

**用途**: 列出通过NVLINK连接的其他实例ID

**格式**: 字符串列表

**自动检测**:
```python
from control_plane.topology import TopologyDetector

detector = TopologyDetector()
topology = detector.detect_nvlink_topology()

# topology = {
#     0: [1, 2],     # GPU 0 与 GPU 1, 2 有 NVLINK
#     1: [0, 3],
#     ...
# }
```

**手动配置**:
```python
# GPU 0 实例
instance_0 = ExecutionInstance(
    instance_id="gpu-0",
    nvlink_peers=["gpu-1", "gpu-2"],  # 与 GPU 1, 2 有 NVLINK
    ...
)
```

**重要性**: ⭐⭐⭐⭐（显著提升性能）

#### 5. numa_node（NUMA节点）

**用途**: 标识GPU所属的NUMA节点

**格式**: 整数（通常0-1或0-3）

**获取方法**:
```bash
# 查看 GPU 0 的 NUMA 节点
cat /sys/bus/pci/devices/0000:17:00.0/numa_node
```

**Python获取**:
```python
detector = TopologyDetector()
numa_node = detector.detect_numa_nodes().get(0)  # GPU 0的NUMA节点
```

**重要性**: ⭐⭐⭐（避免跨NUMA访问）

## 自动拓扑检测

### TopologyDetector 使用

```python
from control_plane.topology import TopologyDetector

# 创建检测器
detector = TopologyDetector()

# 1. 检测 NVLINK 拓扑
nvlink_topology = detector.detect_nvlink_topology()
print(f"NVLINK topology: {nvlink_topology}")
# 输出: {0: [1, 2], 1: [0, 3], ...}

# 2. 检测 NUMA 节点
numa_nodes = detector.detect_numa_nodes()
print(f"NUMA nodes: {numa_nodes}")
# 输出: {0: 0, 1: 0, 2: 1, 3: 1, ...}

# 3. 获取 GPU 总线 ID
bus_id = detector.get_gpu_bus_id(gpu_device_id=0)
print(f"GPU 0 bus ID: {bus_id}")
# 输出: 0000:17:00.0
```

### 自动创建拓扑感知实例

```python
from control_plane.topology import TopologyDetector
from control_plane.types import ExecutionInstance

detector = TopologyDetector()

# 自动检测并创建实例
instance = detector.create_instance_with_topology(
    instance_id="gpu-0",
    host="localhost",
    port=8000,
    model_name="llama-2-7b",
    gpu_device_id=0,
    tensor_parallel_size=1
)

print(f"Machine ID: {instance.machine_id}")
print(f"GPU Bus ID: {instance.gpu_bus_id}")
print(f"NVLINK Peers: {instance.nvlink_peers}")
print(f"NUMA Node: {instance.numa_node}")
```

### 批量自动发现本地实例

```python
# 自动发现本地所有 GPU 实例
instances = detector.auto_detect_local_instances(
    base_port=8000,
    model_name="llama-2-7b"
)

for inst in instances:
    print(f"Discovered: {inst.instance_id} on GPU {inst.gpu_device_id}")
    await manager.register_instance(inst)
```

## 拓扑感知路由

### 启用拓扑感知路由

```python
from control_plane.manager import ControlPlaneManager
from control_plane.router import Router

# 方法1: 在 Manager 初始化时指定
manager = ControlPlaneManager(routing_strategy="topology_aware")

# 方法2: 动态切换路由策略
manager.router.strategy = "topology_aware"
```

### 路由策略行为

拓扑感知路由策略的选择优先级：

```
1. 同一机器 + NVLINK 连接 (亲和分数: 1.0)
   └─> 最优选择，延迟最低，可共享 KV 缓存

2. 同一机器 + 同一 NUMA 节点 (亲和分数: 0.5)
   └─> 次优选择，避免跨 NUMA 内存访问

3. 同一机器 + 不同 NUMA 节点 (亲和分数: 0.5)
   └─> 可接受，仍在本地

4. 同一机架 (亲和分数: 0.1)
   └─> 跨机，但网络延迟较低

5. 不同机架 (亲和分数: 0.01)
   └─> 最差选择，高延迟
```

### 亲和分数计算

```python
# ExecutionInstance 提供的方法
affinity_score = instance1.get_affinity_score(instance2)

# 示例
inst_a = ExecutionInstance(
    instance_id="gpu-0",
    machine_id="node1",
    nvlink_peers=["gpu-1"]
)

inst_b = ExecutionInstance(
    instance_id="gpu-1",
    machine_id="node1",
    nvlink_peers=["gpu-0"]
)

score = inst_a.get_affinity_score(inst_b)
# score = 1.0 (NVLINK 连接)
```

### 检查本地性

```python
# 检查两个实例是否在同一机器
is_local = instance1.is_local_to(instance2)

if is_local:
    print("Instances are on the same machine")
```

## 部署场景

### 场景1: 单机8卡部署

**硬件配置**:
- 1台服务器，8块 A100 80GB GPU
- NVLINK: 全连接拓扑（每个GPU连接其他所有GPU）
- NUMA: 2个节点，每个节点4块GPU

**实例配置**:
```python
from control_plane.topology import TopologyDetector

detector = TopologyDetector()

# 自动检测并注册所有8块GPU
instances = detector.auto_detect_local_instances(
    base_port=8000,  # 端口 8000-8007
    model_name="llama-2-70b"
)

for inst in instances:
    await manager.register_instance(inst)
```

**拓扑示意**:
```
NUMA 0: GPU 0, 1, 2, 3  ──┐
                          ├─ NVLINK 全连接
NUMA 1: GPU 4, 5, 6, 7  ──┘
```

**路由策略**: `topology_aware`
- 优先调度到同一 NUMA 节点的 GPU
- 充分利用 NVLINK 共享 KV 缓存

### 场景2: 4机32卡集群

**硬件配置**:
- 4台服务器，每台8块 A100 GPU
- 网络: 100 Gbps InfiniBand
- 机架: 全部在同一机架

**实例配置**:
```python
# 在每台机器上运行
import socket

machine_id = socket.gethostname()  # node1, node2, node3, node4
detector = TopologyDetector()

instances = detector.auto_detect_local_instances(
    base_port=8000,
    model_name="llama-2-70b"
)

# 所有实例共享同一 Control Plane
for inst in instances:
    # Control Plane 在 master 节点运行
    async with aiohttp.ClientSession() as session:
        await session.post(
            "http://master-node:9000/register_instance",
            json={
                "instance_id": inst.instance_id,
                "host": inst.host,
                "port": inst.port,
                # ... 其他字段
            }
        )
```

**拓扑示意**:
```
┌─ node1 ─────────┐  ┌─ node2 ─────────┐
│ GPU 0-7 NVLINK  │  │ GPU 8-15 NVLINK │
└─────────────────┘  └─────────────────┘
         │  100 Gbps IB  │
┌─ node3 ─────────┐  ┌─ node4 ─────────┐
│ GPU 16-23 NVLINK│  │ GPU 24-31 NVLINK│
└─────────────────┘  └─────────────────┘
```

**路由策略**: `topology_aware` + 用户亲和性
- 同一用户的连续请求调度到同一机器
- 避免跨机通信

### 场景3: 跨机架高可用部署

**硬件配置**:
- 8台服务器，分布在2个机架
- 每机架4台服务器，每台8卡
- 网络: 200 Gbps 跨机架链路

**实例配置**:
```python
# rack-a 机器配置
instances_rack_a = []
for node_id in ["node1", "node2", "node3", "node4"]:
    detector = TopologyDetector()
    local_instances = detector.auto_detect_local_instances(
        base_port=8000,
        model_name="llama-2-70b"
    )
    
    # 添加机架信息
    for inst in local_instances:
        inst.rack_id = "rack-a"
        inst.machine_id = node_id
        instances_rack_a.append(inst)

# rack-b 同理
instances_rack_b = [...]  # rack_id = "rack-b"
```

**容灾策略**:
```python
# 自定义策略：避免单机架故障
class RackAwarePolicy(SchedulingPolicy):
    def get_next_request(self, pending_queue, available_instances):
        if not pending_queue or not available_instances:
            return None
        
        request = pending_queue[0]
        
        # 检查机架分布
        racks = {inst.rack_id for inst in available_instances if inst.available}
        
        if len(racks) < 2:
            logger.warning("Only one rack available, high availability at risk")
        
        return request
```

### 场景4: PD分离 + 拓扑优化

**硬件配置**:
- Prefilling实例: 4机，每机4卡 A100（TP=4）
- Decoding实例: 8机，每机1卡 A100（TP=1）

**配置示例**:
```python
from control_plane.types import ExecutionInstanceType

# Prefilling 实例（高TP，优化吞吐）
for node in ["pf-node1", "pf-node2", "pf-node3", "pf-node4"]:
    detector = TopologyDetector()
    instances = detector.auto_detect_local_instances(
        base_port=8000,
        model_name="llama-2-70b",
        tensor_parallel_size=4  # 使用4卡并行
    )
    
    for inst in instances:
        inst.instance_type = ExecutionInstanceType.PREFILLING
        inst.machine_id = node
        await manager.register_instance(inst)

# Decoding 实例（低TP，优化延迟）
for node in ["dec-node1", ..., "dec-node8"]:
    detector = TopologyDetector()
    instances = detector.auto_detect_local_instances(
        base_port=8000,
        model_name="llama-2-70b",
        tensor_parallel_size=1  # 单卡
    )
    
    for inst in instances:
        inst.instance_type = ExecutionInstanceType.DECODING
        inst.machine_id = node
        await manager.register_instance(inst)

# 启用 PD 分离
manager = ControlPlaneManager(enable_pd_separation=True)
```

## 配置指南

### 最小配置（仅 machine_id）

对于简单部署，只需配置 `machine_id`：

```python
instance = ExecutionInstance(
    instance_id="gpu-0",
    host="192.168.1.10",
    port=8000,
    model_name="llama-2-7b",
    machine_id="node1",  # 最重要
    # 其他拓扑字段留空
)
```

路由器会优先选择同一 `machine_id` 的实例。

### 推荐配置（含 NVLINK）

对于生产环境，建议配置 NVLINK：

```python
from control_plane.topology import TopologyDetector

detector = TopologyDetector()

# 自动检测所有拓扑信息
instance = detector.create_instance_with_topology(
    instance_id="gpu-0",
    host="192.168.1.10",
    port=8000,
    model_name="llama-2-70b",
    gpu_device_id=0,
    tensor_parallel_size=1
)
# 自动填充: machine_id, gpu_bus_id, nvlink_peers, numa_node
```

### 完整配置（含机架/网络）

对于大规模集群，配置完整拓扑信息：

```python
instance = ExecutionInstance(
    instance_id="gpu-rack-a-node-1-0",
    host="192.168.1.10",
    port=8000,
    model_name="llama-2-70b",
    
    # 机器拓扑
    machine_id="node-1",
    rack_id="rack-a",
    
    # GPU 拓扑（自动检测）
    gpu_bus_id="0000:17:00.0",
    gpu_device_id=0,
    nvlink_peers=["gpu-rack-a-node-1-1", "gpu-rack-a-node-1-2"],
    numa_node=0,
    
    # 网络拓扑（手动配置）
    network_bandwidth_gbps=100,  # IB带宽
    network_latency_ms=0.5,      # 同机架延迟
    
    # 共享资源
    shared_memory_pool="node-1-shm",
    shared_storage_path="/mnt/nvme-shared",
)
```

## 性能优化

### 1. 用户亲和性调度

将同一用户的请求调度到同一机器：

```python
from control_plane.router import Router

router = Router(strategy="affinity")
manager = ControlPlaneManager(router=router)

# 提交请求时指定 user_id
request = RequestMetadata(
    request_id="req-123",
    user_id="user-alice",  # 关键
    prompt="Hello",
    ...
)
await manager.submit_request(request)
```

路由器会记住 `user-alice` 上次使用的实例，优先调度到同一实例（如果在同一机器上）。

### 2. KV 缓存亲和性

对于多轮对话，保持在同一 NVLINK 组内：

```python
# 第一轮对话
request1 = RequestMetadata(
    request_id="conv-1-turn-1",
    user_id="user-bob",
    conversation_id="conv-1",  # 会话ID
    prompt="What is the capital of France?",
    ...
)
await manager.submit_request(request1)

# 第二轮对话（路由到相同NVLINK组）
request2 = RequestMetadata(
    request_id="conv-1-turn-2",
    user_id="user-bob",
    conversation_id="conv-1",  # 相同会话ID
    prompt="What about Germany?",
    ...
)
await manager.submit_request(request2)
```

拓扑感知路由会优先选择 NVLINK 连接的实例，提高 KV 缓存命中率。

### 3. 负载均衡 + 拓扑优化

结合负载均衡和拓扑亲和性：

```python
# 自定义策略
class BalancedTopologyRouter(Router):
    def select_instance(self, request, instances):
        # 1. 过滤同机器实例
        local_instances = [
            i for i in instances
            if i.machine_id == self.preferred_machine_id
        ]
        
        if not local_instances:
            local_instances = instances  # 降级到全局
        
        # 2. 在本地实例中选择负载最低的
        return min(local_instances, key=lambda i: i.current_load)
```

## 故障排查

### 问题1: NVLINK 检测失败

**症状**:
```python
detector = TopologyDetector()
topology = detector.detect_nvlink_topology()
# 返回空字典 {}
```

**原因**:
- `nvidia-smi` 不可用
- GPU 驱动版本过低
- 机器上确实没有 NVLINK

**解决方法**:
```bash
# 检查 nvidia-smi
nvidia-smi topo -m

# 输出应包含 NVLINK 信息:
#     GPU0  GPU1  GPU2  ...
# GPU0  X    NV12  NV12  ...
# GPU1  NV12  X    NV12  ...

# 如果全是 "PHB" 或 "SOC"，说明没有 NVLINK
```

**降级方案**: 不配置 `nvlink_peers`，仅使用 `machine_id`

### 问题2: 跨机调度延迟高

**症状**:
```
实例 inst-node1-0 的 P95 延迟: 2000ms
实例 inst-node2-0 的 P95 延迟: 150ms
```

**原因**: 路由器未启用拓扑感知，随机跨机调度

**解决方法**:
```python
# 检查路由策略
print(manager.router.strategy)  # 应为 "topology_aware"

# 切换策略
manager.router.strategy = "topology_aware"

# 或在初始化时指定
manager = ControlPlaneManager(routing_strategy="topology_aware")
```

### 问题3: machine_id 不一致

**症状**:
```python
# node1 上的实例
instance1.machine_id = "node1"

# node1 上的另一个实例
instance2.machine_id = "gpu-server-1"

# 路由器认为它们不在同一机器
```

**原因**: `machine_id` 命名不一致

**解决方法**: 统一使用主机名
```python
import socket

machine_id = socket.gethostname()  # 在所有实例中使用
```

### 问题4: NUMA 节点错误

**症状**: 性能低于预期，内存带宽低

**调试**:
```bash
# 检查 GPU 0 的 NUMA 节点
cat /sys/bus/pci/devices/$(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader -i 0 | tr '[:upper:]' '[:lower:]')/numa_node

# 检查进程绑定
numactl --show

# 绑定到正确的 NUMA 节点
numactl --cpunodebind=0 --membind=0 python -m vllm.entrypoints.openai.api_server ...
```

## 相关文档

- [自定义调度策略开发](CUSTOM_SCHEDULING.md) - 如何在策略中利用拓扑信息
- [监控指标文档](METRICS.md) - 拓扑感知对性能指标的影响
- [部署指南](DEPLOYMENT.md) - 多机部署的完整流程

## 总结

拓扑感知是 sageLLM 在多机多卡环境下实现高性能的关键技术：

1. **自动检测**: 使用 `TopologyDetector` 自动发现 NVLINK、NUMA 拓扑
2. **核心字段**: 最重要的是 `machine_id` 和 `nvlink_peers`
3. **路由优化**: 启用 `topology_aware` 路由策略
4. **性能提升**: 同机 NVLINK 调度可降低 50-80% 延迟

开始配置您的拓扑感知部署吧！
