# ANHCAPE — An Huynh Custom Adaptive Policy with Embedding

> **DRL-based Mobile Robot Navigation using ROS 2 + Gazebo**

<div align="center">
  <img src="/docs/simulation.gif" alt="ANHCAPE Simulation" />
</div>

**ANHCAPE** là kiến trúc Deep Reinforcement Learning tùy chỉnh của **An Huynh**, xây dựng trên nền TD3 với các cải tiến:
- **Adaptive** — Exploration decay, Q-value clipping, adaptive checkpointing
- **Policy** — Actor-Critic policy optimization tối ưu cho robot navigation
- **Embedding** — Dual Encoder system (Encoder + Fixed Encoder) cho state representation

---

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup (Docker)](#setup-docker)
- [Build](#build)
- [Training](#training)
- [Testing](#testing)
- [Visualization](#visualization)
- [Additional Demos](#additional-demos)

---

## Project Structure

```txt
anhc_agent_ws/
└── src/
    ├── 📂 anhc_agent/              ← Main ANHCAPE DRL agent package
    │   ├── 📂 config/              ← train_config.yaml, test_config.yaml
    │   ├── 📂 launch/              ← test_anhcape.launch.py
    │   ├── 📂 scripts/
    │   │   ├── 📂 policy/          ← anhcape_agent.py, train_anhcape_agent.py, test_anhcape_agent.py
    │   │   ├── 📂 environment/     ← environment.py, environment_interface.py
    │   │   └── 📂 utils/           ← buffer.py, file_manager.py, plot_*.py
    │   └── 📂 temp/               ← Checkpoints, logs, eval results
    │
    ├── 📂 anhc_agent_description/  ← Robot URDF, meshes, sensor models
    │   ├── 📂 launch/
    │   ├── 📂 meshes/
    │   ├── 📂 models/
    │   └── 📂 urdf/
    │
    ├── 📂 anhc_agent_gazebo/       ← Gazebo simulation worlds & launch files
    │   ├── 📂 config/              ← Pedsim, SLAM configs
    │   ├── 📂 launch/              ← simulation.launch.py, training_env*.launch.py
    │   ├── 📂 models/
    │   └── 📂 worlds/
    │
    ├── 📂 anhc_agent_interfaces/   ← Custom ROS 2 service/message definitions
    │   ├── 📂 srv/                 ← Step, Reset, Seed, GetDimensions, SampleActionSpace
    │   └── 📂 action/
    │
    ├── 📂 pedsim_ros2/             ← Pedestrian simulation (submodule)
    └── 📂 velodyne_simulator/      ← Velodyne LiDAR simulation (submodule)
```

---

## Requirements

- **OS**: Ubuntu 22.04
- **ROS 2**: Humble (recommended) hoặc Jazzy
- **GPU**: NVIDIA (khuyến nghị, cần CUDA ≥ 11.8)
- **Docker**: Docker + NVIDIA Container Toolkit (cho setup Docker)
- **Python**: 3.10+, PyTorch, squaternion, tensorboard

---

## Setup (Docker)

### 1) Cài Docker và NVIDIA Container Toolkit

```bash
# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2) Build Docker image

```bash
cd ~/anhc_agent_ws/src
docker build -t anhcape_ros2_image .devcontainer/
```

### 3) Cho phép Docker hiển thị GUI (Gazebo)

```bash
xhost +local:docker
```

### 4) Chạy container

```bash
docker run -it --rm \
  --name anhcape_ros2 \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --env="ANHC_AGENT_SRC_PATH=/root/anhc_agent_ws/src/src/" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/anhc_agent_ws:/root/anhc_agent_ws" \
  --network=host \
  --gpus all \
  anhcape_ros2_image \
  bash
```

---

## Build

### Lần đầu (bên trong container hoặc native):

```bash
cd ~/anhc_agent_ws

# Cài dependencies
rosdep init || true
rosdep update
rosdep install --from-path src -yi --rosdistro humble

# Cài Python packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install squaternion tensorboard
pip install -r src/requirements.txt

# Build
source /opt/ros/humble/setup.bash   # hoặc jazzy
rm -rf build/ install/ log/
colcon build --symlink-install
source install/setup.bash
```

### Build lại một package:

```bash
cd ~/anhc_agent_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select anhc_agent_gazebo --symlink-install
source install/setup.bash
```

### Cấp lại quyền nếu bị permission error:

```bash
sudo chown -R $USER:$USER ~/anhc_agent_ws/build ~/anhc_agent_ws/install ~/anhc_agent_ws/log
```

---

## Training

> **Đặt environment variable trước khi chạy bất kỳ terminal nào:**
> ```bash
> export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
> ```
> Hoặc thêm vào `~/.bashrc`:
> ```bash
> echo 'export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/' >> ~/.bashrc
> source ~/.bashrc
> ```

### Terminal 1 — Khởi động Gazebo simulation

```bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
cd ~/anhc_agent_ws && source install/setup.bash
source /usr/share/gazebo/setup.bash

# Static environment
ros2 launch anhc_agent_gazebo training_env.launch.py

# Dynamic environment (với pedestrians)
ros2 launch anhc_agent_gazebo training_env_dynamic.launch.py \
  pedsim_start_delay:=4.0 \
  bridge_start_delay:=6.0
```

### Terminal 2 — DRL environment node

```bash
docker exec -it anhcape_ros2 bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
cd ~/anhc_agent_ws && source install/setup.bash
ros2 run anhc_agent environment.py
```

### Terminal 3 — Train ANHCAPE agent

```bash
docker exec -it anhcape_ros2 bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
cd ~/anhc_agent_ws && source install/setup.bash
ros2 run anhc_agent train_anhcape_agent.py
```

### Terminal 4 — TensorBoard (host machine)

```bash
source ~/anhc_agent_ws/.venv/bin/activate  # nếu dùng venv
tensorboard --logdir ~/anhc_agent_ws/src/src/anhc_agent/temp/logs --bind_all
# Mở trình duyệt: http://localhost:6006
```

---

### Cấu hình Training (`train_config.yaml`)

File: `src/src/anhc_agent/config/train_config.yaml`

```yaml
train_settings:
  seed: 40
  use_checkpoints: true
  load_model: true                  # true = resume từ checkpoint
  timesteps_before_training: 25000
  eval_freq: 5000
  eval_eps: 10
  max_timesteps: 5000000
  max_episode_steps: 500
  base_file_name: "anhcape_agent"
  save_date: ""                     # "" = dùng checkpoint mới nhất
                                    # "20260524_073634" = resume phiên cụ thể
```

---

### Resume Training từ Checkpoint

Checkpoint được lưu tự động tại:
```
src/src/anhc_agent/temp/pytorch_models/
src/src/anhc_agent/temp/train_state/
```

Để resume, đặt `load_model: true` và `save_date` tương ứng trong `train_config.yaml`, sau đó chạy lại Terminal 3.

---

## Testing

### Terminal 1 — Gazebo simulation

```bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
cd ~/anhc_agent_ws && source install/setup.bash
ros2 launch anhc_agent_gazebo training_env.launch.py
```

### Terminal 2 — Environment node

```bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
cd ~/anhc_agent_ws && source install/setup.bash
ros2 run anhc_agent environment.py
```

### Terminal 3 — Test agent (launch file)

```bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
cd ~/anhc_agent_ws && source install/setup.bash
ros2 launch anhc_agent test_anhcape.launch.py
```

Hoặc chạy trực tiếp:

```bash
ros2 run anhc_agent test_anhcape_agent.py
```

### Cấu hình Testing (`test_config.yaml`)

```yaml
test_settings:
  seed: 40
  base_file_name: "anhcape_agent"
  save_date: "20241019"          # Chỉnh theo tên checkpoint muốn test
  max_episode_steps: 500
  use_checkpoints: true
```

---

## Visualization

### Plot training metrics

```bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
python3 src/src/anhc_agent/scripts/utils/plot_metrics.py
```

### Plot reward curves

```bash
python3 src/src/anhc_agent/scripts/utils/plot_reward.py
```

### Plot trajectories on map

```bash
python3 src/src/anhc_agent/scripts/utils/plot_trajectories_on_map.py
```

---

## Pedsim (Pedestrian Simulation) — Tuning

Để điều chỉnh hành vi người đi bộ, chỉnh file:
```
src/src/anhc_agent_gazebo/config/pedsim_training_env.xml
```

Sau khi chỉnh, rebuild:
```bash
cd ~/anhc_agent_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select anhc_agent_gazebo --symlink-install
source install/setup.bash
ros2 launch anhc_agent_gazebo training_env_dynamic.launch.py
```

---

## Additional Demos

<table width="100%">
  <tr>
    <td align="center" width="50%">
      <img src="/docs/slam.gif" alt="SLAM" width="90%">
    </td>
    <td align="center" width="50%">
      <img src="/docs/dynamic_environment.gif" alt="Dynamic Environment" width="90%">
    </td>
  </tr>
  <tr>
    <td align="center"><em>SLAM Mapping</em></td>
    <td align="center"><em>Dynamic Environment Navigation</em></td>
  </tr>
</table>

---

## Architecture Overview

| Component | Description |
|-----------|-------------|
| `AnhcapeAgent` | Main agent class (Actor + Critic + Encoder) |
| `AnhcapeActor` | Policy network với AvgL1Norm normalization |
| `AnhcapeCritic` | Twin Q-networks với value clipping |
| `AnhcapeEncoder` | State embedding (fixed + trainable) |
| LAP Buffer | Prioritized replay buffer với Huber loss |
| Distance Reward | Multi-component reward shaping |

---

## Maintainer

**An Huynh** — ANHCAPE Project  
Email: `anhuynh@example.com`  
Architecture: ANHCAPE v1.0 (TD3 + LAP Buffer + Dual Encoder + Distance Reward)
