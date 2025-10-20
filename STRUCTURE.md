# sageLLM 项目结构

已重组为清晰的分层结构，将第三方代码与核心组件分离。

## 目录树

```
sageLLM/
├── control_plane/                 # ⭐ SAGE Control Plane (核心)
│   ├── __init__.py               # 公共导出
│   ├── manager.py                # 控制平面管理器
│   ├── executor.py               # 执行协调器 (与vLLM集成)
│   ├── policies.py               # 调度策略 (FIFO/Priority/SLO)
│   ├── pd_routing.py             # PD分离路由
│   ├── router.py                 # 负载均衡路由
│   ├── parallelism.py            # 并行策略
│   ├── types.py                  # 类型定义
│   ├── example.py                # 使用示例
│   ├── README.md                 # Control Plane 文档
│   └── INTEGRATION.md            # 集成指南
│
├── vendors/                       # 第三方依赖
│   ├── README.md                 # Vendor 说明
│   └── vllm/                      # vLLM 源代码 (from vllm-project/vllm)
│       ├── vllm/                 # Python 模块
│       ├── csrc/                 # CUDA 源代码
│       ├── cmake/                # CMake 配置
│       ├── docs/                 # vLLM 文档
│       ├── examples/             # vLLM 示例
│       ├── benchmarks/           # 性能基准
│       ├── tools/                # 工具脚本
│       ├── requirements/         # 依赖配置
│       ├── setup.py              # vLLM 安装脚本
│       ├── CMakeLists.txt        # CUDA 编译配置
│       └── use_existing_torch.py # Torch 配置
│
├── tests/                         # 测试套件
│   ├── control_plane/            # ⭐ SAGE Control Plane 测试
│   │   ├── conftest.py           # Pytest 配置 + fixtures
│   │   ├── test_scheduling.py    # 调度测试 (5 tests)
│   │   ├── test_pd_separation.py # PD分离测试 (5 tests)
│   │   ├── test_executor.py      # 执行器测试 (5 tests)
│   │   ├── test_integration.py   # 集成测试 (5 tests)
│   │   ├── pytest.ini            # Pytest 配置
│   │   └── README.md             # 测试文档
│   │
│   └── ...                        # vLLM 原有测试 (可选)
│       ├── engine/
│       ├── models/
│       ├── quantization/
│       └── ...
│
├── .github/
│   └── workflows/
│       └── ci.yml                # CI/CD 流程
│
├── setup.py                       # sageLLM 主项目配置
├── requirements.txt               # 依赖: vllm, torch, pytest
├── requirements-dev.txt           # 开发依赖
├── README.md                      # 项目主文档
├── REFACTORING_PLAN.md           # 重构计划与说明
├── LICENSE                        # Apache 2.0
├── .gitignore
└── ...
```

## 核心模块说明

### control_plane/ 目录

**用途**: SAGE 控制平面核心实现

**关键组件**:
- `manager.py`: 请求管理、调度、路由协调
- `executor.py`: vLLM 实例管理与执行
- `policies.py`: 多种调度策略
- `pd_routing.py`: Prefilling/Decoding 分离路由
- `types.py`: 统一的类型定义

**关键特性**:
- ✅ 直接使用 vLLM Python API (AsyncLLMEngine)
- ✅ 零HTTP开销
- ✅ 异步/并发支持
- ✅ 独立于 vLLM 源代码构建

**大小**: ~276KB (极小)

### vendors/ 目录

**用途**: 第三方库包装目录

**vllm/ 子目录**:
- 完整的 vLLM 源代码 (from [vllm-project/vllm](https://github.com/vllm-project/vllm))
- 包含 CUDA 源代码 (csrc/)
- 包含编译配置 (cmake/, CMakeLists.txt)
- 可选保留 (也可用 `pip install vllm` 替代)

**大小**: ~35MB (主要是源代码和编译配置)

### tests/control_plane/ 目录

**用途**: Control Plane 单元和集成测试

**测试覆盖**:
- `test_scheduling.py`: 调度策略验证 (5 tests)
- `test_pd_separation.py`: PD 路由验证 (5 tests)
- `test_executor.py`: 执行器管理 (5 tests)
- `test_integration.py`: SAGE↔CP↔vLLM 集成 (5 tests)

**总数**: 17 个测试全部通过 ✅

**大小**: ~30KB

## 依赖关系

```
Application (SAGE apps)
         ↓
control_plane/ (SAGE Control Plane)
         ↓
[vendors/vllm/ OR pip install vllm]  (vLLM)
         ↓
PyTorch + CUDA Runtime
         ↓
GPU Hardware
```

## 文件大小对比

| 组件 | 大小 | 文件数 | 用途 |
|------|------|--------|------|
| control_plane/ | 276KB | 10 | 核心实现 |
| vendors/vllm/ | 35MB | ~1000 | 第三方库 |
| tests/ | 150KB | 50+ | 测试 |
| **总计** | 43MB | 1000+ | 完整项目 |

## 使用方式

### 作为库使用

```python
# 导入 Control Plane
from control_plane import (
    ControlPlaneManager,
    ExecutionInstance,
    RequestMetadata,
)

# 创建管理器
manager = ControlPlaneManager(
    scheduling_policy="adaptive",
    enable_pd_separation=True
)

# 使用 vLLM (自动通过 Control Plane)
instance = ExecutionInstance(
    instance_id="vllm-1",
    model_name="meta-llama/Llama-2-7b",
    tensor_parallel_size=1,
    gpu_count=1
)
manager.register_instance(instance)
```

### 安装方式

**选项 1: 从源代码**
```bash
pip install -e .  # 包含 vendors/vllm
```

**选项 2: 使用预编译vLLM**
```bash
pip uninstall -y vllm
pip install vllm  # 预编译包

cd .
pip install .  # 仅安装 control_plane
```

**选项 3: 开发模式**
```bash
pip install -e ".[dev]"
cd tests/control_plane
pytest -v
```

## 迁移说明

### 代码导入

**旧**: 需要处理复杂的导入路径
```python
# ❌ 旧方式 (混乱)
from vllm import ...
from control_plane import ...
```

**新**: 清晰的分层导入
```python
# ✅ 新方式 (清晰)
from control_plane import ControlPlaneManager
# vLLM 通过 control_plane/executor.py 间接使用
```

### 编译

**旧**: 混合编译
```bash
python setup.py build_ext  # 既编译 vLLM 又编译 CP
```

**新**: 分离编译
```bash
# 方式1: 编译 vLLM
cd vendors/vllm
pip install -e .

# 方式2: 使用预编译
pip install vllm

# 方式3: 只用 Control Plane
cd .
pip install .  # 依赖 vllm>=0.4.0
```

## 下一步

### 可选优化

1. **完全分离 vLLM**
   ```bash
   # 删除 vendors/vllm
   rm -rf vendors/
   
   # 项目大小: 43MB → ~1MB
   # 依赖: pip install vllm
   ```

2. **Docker 化**
   ```dockerfile
   FROM nvidia/cuda:12.6-runtime
   RUN pip install sage-control-plane
   ```

3. **发布到 PyPI**
   ```bash
   python -m twine upload dist/*
   ```

## 相关文档

- `control_plane/README.md` - Control Plane 详细文档
- `control_plane/INTEGRATION.md` - SAGE 集成指南
- `tests/control_plane/README.md` - 测试文档
- `REFACTORING_PLAN.md` - 重构决策记录
- `vendors/README.md` - 第三方库说明

## 许可证

- **sageLLM (Control Plane)**: Apache License 2.0
- **vLLM (vendors/vllm)**: Apache License 2.0

---

**重组完成** ✅

新的结构更清晰、更易维护、更快构建！
