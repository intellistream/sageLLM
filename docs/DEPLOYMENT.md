# vLLM 部署指南

本文档说明如何启动 vLLM 服务器实例，以便 Control Plane 进行统一调度。

## 架构说明

Control Plane 通过 HTTP API 与所有 vLLM 实例通信，因此：

1. **每个 GPU 启动一个独立的 vLLM 服务器**
2. **Control Plane 通过 HTTP 调用 vLLM 的 OpenAI-compatible API**
3. **本地 GPU 和远程 GPU 对 Control Plane 来说是透明的**

```
Control Plane (HTTP 客户端)
    ↓
    ├── localhost:8000 (GPU 0, 本机)
    ├── localhost:8001 (GPU 1, 本机)
    ├── localhost:8002 (GPU 2, 本机)
    ├── localhost:8003 (GPU 3, 本机)
    ├── 192.168.1.100:8000 (远程机器 A, GPU 0)
    └── 192.168.1.100:8001 (远程机器 A, GPU 1)
```

---

## 一体机部署（单机多卡）

### 前提条件

```bash
# 安装 vLLM
pip install vllm

# 确认 GPU 可用
nvidia-smi
```

### 启动多个 vLLM 服务器

假设一体机有 **4 个 GPU**，需要启动 4 个独立的 vLLM 服务器，每个绑定一个 GPU：

```bash
# GPU 0 - 端口 8000
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1

# GPU 1 - 端口 8001
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 1

# GPU 2 - 端口 8002
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b \
    --host 0.0.0.0 \
    --port 8002 \
    --tensor-parallel-size 1

# GPU 3 - 端口 8003
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b \
    --host 0.0.0.0 \
    --port 8003 \
    --tensor-parallel-size 1
```

### 使用 systemd 管理（推荐）

创建服务文件 `/etc/systemd/system/vllm-gpu@.service`:

```ini
[Unit]
Description=vLLM Server on GPU %i
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="CUDA_VISIBLE_DEVICES=%i"
ExecStart=/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b \
    --host 0.0.0.0 \
    --port 800%i \
    --tensor-parallel-size 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
# 启动 4 个 GPU 的 vLLM 服务
sudo systemctl start vllm-gpu@0
sudo systemctl start vllm-gpu@1
sudo systemctl start vllm-gpu@2
sudo systemctl start vllm-gpu@3

# 设置开机自启
sudo systemctl enable vllm-gpu@{0,1,2,3}

# 查看状态
sudo systemctl status vllm-gpu@0
```

### Control Plane 配置

```python
from control_plane import ControlPlaneManager, ExecutionInstance

async def setup_local_cluster():
    cp = ControlPlaneManager(scheduling_policy="adaptive")
    
    # 注册本机 4 个 GPU
    for gpu_id in range(4):
        instance = ExecutionInstance(
            instance_id=f"local-gpu-{gpu_id}",
            host="localhost",  # 或 "127.0.0.1"
            port=8000 + gpu_id,
            model_name="meta-llama/Llama-2-7b",
            tensor_parallel_size=1,
            gpu_count=1,
        )
        cp.register_instance(instance)
    
    await cp.start()
    return cp
```

---

## 多机部署（跨机器调度）

### 机器配置示例

**机器 A** (IP: 192.168.1.100, 4 GPUs)
**机器 B** (IP: 192.168.1.101, 4 GPUs)
**机器 C** (IP: 192.168.1.102, 无 GPU, 运行 Control Plane)

### 在每台 GPU 机器上启动 vLLM

**机器 A 和机器 B 都执行：**

```bash
# 启动 4 个 vLLM 服务器，监听所有网络接口
for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b \
        --host 0.0.0.0 \
        --port 800$i \
        --tensor-parallel-size 1 \
        > /var/log/vllm-gpu-$i.log 2>&1 &
done
```

### Control Plane 配置（在机器 C）

```python
from control_plane import ControlPlaneManager, ExecutionInstance

async def setup_multi_machine_cluster():
    cp = ControlPlaneManager(
        scheduling_policy="cost_optimized",
        routing_strategy="load_balanced",
    )
    
    # 机器 A 的 4 个 GPU
    for gpu_id in range(4):
        instance = ExecutionInstance(
            instance_id=f"machine-a-gpu-{gpu_id}",
            host="192.168.1.100",
            port=8000 + gpu_id,
            model_name="meta-llama/Llama-2-7b",
            tensor_parallel_size=1,
            gpu_count=1,
        )
        cp.register_instance(instance)
    
    # 机器 B 的 4 个 GPU
    for gpu_id in range(4):
        instance = ExecutionInstance(
            instance_id=f"machine-b-gpu-{gpu_id}",
            host="192.168.1.101",
            port=8000 + gpu_id,
            model_name="meta-llama/Llama-2-7b",
            tensor_parallel_size=1,
            gpu_count=1,
        )
        cp.register_instance(instance)
    
    await cp.start()
    
    # Control Plane 现在统一调度 8 个 GPU
    return cp
```

---

## 验证部署

### 1. 检查 vLLM 服务健康状态

```bash
# 本地
curl http://localhost:8000/health
curl http://localhost:8001/health

# 远程
curl http://192.168.1.100:8000/health
curl http://192.168.1.101:8000/health
```

### 2. 测试 vLLM API

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### 3. 通过 Control Plane 测试

```python
import asyncio
from control_plane import ControlPlaneManager, ExecutionInstance, RequestMetadata

async def test_control_plane():
    cp = ControlPlaneManager()
    
    # 注册实例
    instance = ExecutionInstance(
        instance_id="test-gpu",
        host="localhost",
        port=8000,
        model_name="meta-llama/Llama-2-7b",
        gpu_count=1,
    )
    cp.register_instance(instance)
    
    await cp.start()
    
    # 提交测试请求
    request = RequestMetadata(
        request_id="test-001",
        prompt="Tell me about artificial intelligence",
        max_tokens=100,
    )
    
    await cp.submit_request(request)
    
    await asyncio.sleep(5)
    await cp.stop()

asyncio.run(test_control_plane())
```

---

## 常见问题

### Q1: 如何使用 TP > 1 的配置？

如果要用 4 个 GPU 做 Tensor Parallel：

```bash
# 使用 GPU 0-3 做 TP=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b \
    --host 0.0.0.0 \
    --port 9000 \
    --tensor-parallel-size 4
```

Control Plane 注册：

```python
instance = ExecutionInstance(
    instance_id="tp4-instance",
    host="localhost",
    port=9000,
    model_name="meta-llama/Llama-2-70b",
    tensor_parallel_size=4,
    gpu_count=4,  # 使用 4 个 GPU
)
```

### Q2: 如何配置防火墙？

确保 vLLM 端口可访问：

```bash
# Ubuntu/Debian
sudo ufw allow 8000:8010/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8000-8010/tcp --permanent
sudo firewall-cmd --reload
```

### Q3: 如何监控 vLLM 实例？

vLLM 暴露 Prometheus metrics：

```bash
curl http://localhost:8000/metrics
```

可以集成到 Prometheus + Grafana 进行监控。

---

## 最佳实践

1. **使用 systemd 或 supervisor 管理 vLLM 进程**，确保崩溃后自动重启
2. **配置日志滚动**，避免日志文件过大
3. **定期健康检查**，Control Plane 会自动剔除不健康的实例
4. **资源监控**，监控 GPU 利用率、内存、温度
5. **负载均衡**，合理配置 Control Plane 的调度策略

---

## 下一步

- 阅读 [control_plane/README.md](../control_plane/README.md) 了解调度策略
- 阅读 [control_plane/INTEGRATION.md](../control_plane/INTEGRATION.md) 了解 API 使用
- 实现自定义调度策略，专注于优化调度算法
