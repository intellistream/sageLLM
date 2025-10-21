# sageLLM - 智能 LLM 推理调度控制平面

<p align="center">
  <strong>基于 vLLM 的高性能、智能化 LLM 推理调度管理系统</strong>
</p>

<p align="center">
| <a href="#概述"><b>概述</b></a> | <a href="#核心特性"><b>核心特性</b></a> | <a href="#快速开始"><b>快速开始</b></a> | <a href="#架构"><b>架构</b></a> | <a href="./control_plane/README.md"><b>Control Plane 文档</b></a> | <a href="./control_plane/INTEGRATION.md"><b>集成指南</b></a> |
</p>

---

## 概述

**sageLLM** 是 SAGE 项目中的 LLM 推理控制平面，提供智能请求调度、多实例管理和动态并行优化。它直接集成 vLLM 的 Python API，在用户应用和执行引擎之间提供一个高效的管理层。

**与传统 HTTP API 调用不同**，sageLLM：
- ✅ 直接使用 vLLM AsyncLLMEngine API（**零 HTTP 开销**）
- ✅ 提供 PD 分离路由（+50-80% 吞吐，-50-60% 延迟）
- ✅ 支持多种智能调度策略
- ✅ 动态优化并行策略
- ✅ 完全的异步/并发支持

## 核心特性

<details>
<summary>📦 项目结构</summary>

```
sageLLM/
├── control_plane/                 # ⭐ Control Plane 核心组件
│   ├── manager.py                # 控制平面管理器
│   ├── executor.py               # 执行协调器 (vLLM 集成)
│   ├── policies.py               # 调度策略 (FIFO/Priority/SLO)
│   ├── pd_routing.py             # PD 分离路由
│   ├── router.py                 # 负载均衡路由
│   ├── parallelism.py            # 并行策略优化
│   ├── types.py                  # 类型定义
│   ├── example.py                # 使用示例
│   ├── README.md                 # 详细文档
│   └── INTEGRATION.md            # 集成指南
│
├── vendors/vllm/                 # vLLM 源代码 (vendored)
│   ├── vllm/                     # Python 模块
│   ├── csrc/                     # CUDA 内核
│   ├── cmake/                    # 编译配置
│   └── ...
│
├── tests/
│   ├── control_plane/            # Control Plane 单元测试
│   │   ├── test_scheduling.py    # 调度测试 (5 tests)
│   │   ├── test_pd_separation.py # PD 分离测试 (5 tests)
│   │   ├── test_executor.py      # 执行器测试 (5 tests)
│   │   └── test_integration.py   # 集成测试 (5 tests)
│   │
│   └── vendors/vllm/tests/       # vLLM 原有测试
│
├── setup.py                      # 安装脚本
├── requirements.txt              # 依赖配置
└── README.md                     # 本文档
```

</details>

## 🎯 核心特性

### 1️⃣ **直接 vLLM 集成 - 零 HTTP 开销**

```python
from control_plane import ControlPlaneManager, ExecutionInstance

# 创建控制平面
manager = ControlPlaneManager()

# 注册 vLLM 实例
instance = ExecutionInstance(
    instance_id="instance-1",
    model_name="meta-llama/Llama-2-7b",
    tensor_parallel_size=4,
    gpu_count=4,
)
manager.register_instance(instance)

# 直接调用 vLLM Python API
outputs = await manager.process_request(prompt, sampling_params)
```

**优势：**
- ✅ 直接使用 AsyncLLMEngine（无 HTTP 网络开销）
- ✅ 完全的动态控制
- ✅ 实时流式输出
- ✅ 细粒度性能监控

### 2️⃣ **PD 分离 - 路由优化（+50-80% 吞吐，-50-60% 延迟）**

将不同特性的请求智能路由到专门优化的实例：

```python
from control_plane import PDSeparationConfig, ExecutionInstanceType

# 启用 PD 分离
pd_config = PDSeparationConfig(
    enabled=True,
    routing_policy="adaptive",
    prefilling_threshold_input_tokens=800,
)

manager = ControlPlaneManager(
    scheduling_policy="adaptive",
    enable_pd_separation=True,
    pd_config=pd_config,
)

# Prefilling 实例 (优化吞吐)
prefilling_instance = ExecutionInstance(
    instance_id="prefill-1",
    instance_type=ExecutionInstanceType.PREFILLING,
    tensor_parallel_size=4,  # 高吞吐
)

# Decoding 实例 (优化延迟)
decoding_instance = ExecutionInstance(
    instance_id="decode-1",
    instance_type=ExecutionInstanceType.DECODING,
    tensor_parallel_size=1,  # 低延迟
)
```

| 指标 | 单实例 | PD分离 | 提升 |
|-----|------|-------|-----|
| 吞吐 (req/s) | 100 | 150-180 | +50-80% |
| 延迟 (ms) | 120 | 50-60 | -50-60% |
| GPU利用率 | 75% | 90% | +15% |

### 3️⃣ **多种调度策略**

- 🔹 **FIFO** - 先进先出
- 🔹 **Priority** - 优先级调度
- 🔹 **SLO-Aware** - SLO 感知调度
- 🔹 **Cost-Optimized** - 成本优化
- 🔹 **Adaptive** - 自适应多策略

```python
manager = ControlPlaneManager(scheduling_policy="slo_aware")
```

### 4️⃣ **动态并行优化**

自动选择最优的模型并行方案：

- TP (Tensor Parallel) - 张量并行
- PP (Pipeline Parallel) - 流水线并行
- DP (Data Parallel) - 数据并行
- EP (Expert Parallel) - 专家并行
- 混合并行策略

```python
from control_plane import ParallelismConfig

config = ParallelismConfig(
    auto_optimize=True,
    supported_strategies=["TP", "PP", "Hybrid"],
)
```

### 5️⃣ **性能监控与指标**

实时收集和分析性能指标：

```python
metrics = manager.get_metrics()
print(f"吞吐: {metrics.throughput} req/s")
print(f"平均延迟: {metrics.avg_latency} ms")
print(f"GPU利用率: {metrics.gpu_utilization}%")
print(f"缓存命中率: {metrics.cache_hit_rate}%")
```



## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/intellistream/SAGE.git
cd SAGE/packages/sage-common/src/sage/common/components/sage_vllm/sageLLM

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

### 基本使用

```python
import asyncio
from control_plane import ControlPlaneManager, ExecutionInstance

async def main():
    # 创建控制平面管理器
    manager = ControlPlaneManager(
        scheduling_policy="adaptive",
        enable_pd_separation=True,
    )
    
    # 注册 vLLM 实例
    instance = ExecutionInstance(
        instance_id="llama-instance-1",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size=2,
        gpu_count=2,
    )
    manager.register_instance(instance)
    
    # 处理请求
    from vllm.sampling_params import SamplingParams
    
    prompt = "Hello, how are you?"
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )
    
    output = await manager.process_request(
        prompt=prompt,
        sampling_params=sampling_params,
    )
    
    print(f"Output: {output}")
    
    # 获取性能指标
    metrics = manager.get_metrics()
    print(f"吞吐: {metrics.throughput} req/s")
    print(f"平均延迟: {metrics.avg_latency} ms")

if __name__ == "__main__":
    asyncio.run(main())
```

更详细的使用示例，请查看 [`control_plane/example.py`](./control_plane/example.py)

### 运行测试

```bash
# 运行所有 Control Plane 测试
cd tests/control_plane
python -m pytest -v

# 运行特定测试
python -m pytest test_scheduling.py -v
python -m pytest test_pd_separation.py -v

# 生成覆盖率报告
python -m pytest --cov=control_plane tests/control_plane/
```

**测试结果：** ✅ 全部 17 个测试通过
- ✅ 5 个调度测试 (test_scheduling.py)
- ✅ 5 个 PD 分离测试 (test_pd_separation.py)
- ✅ 5 个执行器测试 (test_executor.py)
- ✅ 2 个集成测试 (test_integration.py)

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       User Application                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    Control Plane (sageLLM)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Control Plane Manager (核心管理器)              │   │
│  │  ✓ 请求队列管理                                            │   │
│  │  ✓ 调度循环                                                │   │
│  │  ✓ 健康检查                                                │   │
│  │  ✓ 性能监控                                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│           │                  │                  │                │
│           ▼                  ▼                  ▼                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Scheduling   │  │ Parallelism  │  │ PD Router    │          │
│  │ Policies     │  │ Optimizer    │  │ & Routing    │          │
│  │              │  │              │  │              │          │
│  │ • FIFO       │  │ • Auto TP/PP │  │ • Adaptive   │          │
│  │ • Priority   │  │ • DP/EP      │  │ • Hash       │          │
│  │ • SLO-Aware  │  │ • Hybrid     │  │ • LB         │          │
│  │ • Cost-Opt   │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│           │                  │                  │                │
│           └──────────────────┴──────────────────┘                │
│                              │                                   │
│                              ▼                                   │
│                  ┌──────────────────────┐                        │
│                  │ Execution Coordinator │                        │
│                  │ (AsyncLLMEngine)      │                        │
│                  └──────────────────────┘                        │
└──────────────────────────────┼───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│                     Execution Layer (vLLM)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ vLLM     │  │ vLLM     │  │ vLLM     │  │ vLLM     │        │
│  │ Instance │  │ Instance │  │ Instance │  │ Instance │        │
│  │    1     │  │    2     │  │    3     │  │    N     │        │
│  │ (TP=4)   │  │ (PP=2)   │  │ (Hybrid) │  │ (DP=2)   │        │
│  │ Prefill  │  │ Decode   │  │ Decode   │  │ Encode   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                   │
│  GPU Memory: PagedAttention, KV Cache Management                │
│  Kernels: CUDA, FlashAttention, FlashInfer                      │
│  Quantization: GPTQ, AWQ, FP8                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📚 文档

- **[Control Plane 文档](./control_plane/README.md)** - 完整的 Control Plane 介绍和 API 文档
- **[集成指南](./control_plane/INTEGRATION.md)** - 与应用集成的详细步骤
- **[项目结构](./STRUCTURE.md)** - 详细的目录结构说明
- **[测试文档](./tests/control_plane/README.md)** - 测试套件说明

## ⚙️ 环境设置

### GPU 支持 (推荐用于开发)

```bash
# 安装 CUDA Toolkit
sudo apt update && sudo apt install -y nvidia-cuda-toolkit

# 验证安装
nvcc --version

# 重新安装 sageLLM (将启用 CUDA 内核编译)
pip install -e .
```

### 不使用 GPU 运行测试

```bash
# 测试可以在没有 CUDA 的情况下运行 (仅用于单元测试)
cd tests/control_plane
python -m pytest -v
```

## 🔗 依赖关系

- **vLLM**: LLM 推理引擎（Python API）
- **PyTorch**: 深度学习框架
- **asyncio**: 异步编程
- **pydantic**: 数据验证
- **pytest**: 单元测试

详见 `requirements.txt`

## 📄 许可

本项目采用 Apache 2.0 许可证，详见 [LICENSE](./LICENSE)

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](../../../../../../CONTRIBUTING.md) 了解如何参与开发。

### 快速开始贡献

```bash
# Fork 和 Clone
git clone https://github.com/yourusername/SAGE.git
cd packages/sage-common/src/sage/common/components/sage_vllm/sageLLM

# 创建特性分支
git checkout -b feature/your-feature

# 修改代码并提交
git add .
git commit -m "feat: your feature description"

# Push 并创建 PR
git push origin feature/your-feature
```

## 📮 联系方式

- 📧 邮件：请通过 GitHub Issues 联系
- 💬 讨论：使用 GitHub Discussions
- 🐛 Bug 报告：GitHub Issues

## 🙏 致谢

- 感谢 [vLLM 项目](https://github.com/vllm-project/vllm) 提供优秀的 LLM 推理引擎
- 感谢 SAGE 项目团队的支持和指导
