# TD3 Baseline — Hướng Dẫn Training

> **Mục tiêu:** Chạy TD3 Baseline trong **cùng môi trường** với AnhcApe để benchmark và đánh giá hiệu năng.

---

## 📁 Cấu Trúc File TD3

```
anhc_agent/
├── config/
│   ├── td3_hyperparameters.yaml     ← Hyperparameters TD3 (chỉnh tại đây)
│   └── train_td3_config.yaml        ← Cài đặt training (seed, timesteps, ...)
├── scripts/
│   ├── policy/
│   │   ├── td3_agent.py             ← TD3Actor + TD3Critic + TD3Agent
│   │   └── train_td3_agent.py       ← Training loop
│   └── utils/
│       └── td3_uniform_buffer.py    ← Uniform Replay Buffer
└── launch/
    └── train_td3.launch.py          ← Launch file
```

**Output được lưu tách biệt hoàn toàn với AnhcApe:**

```
anhc_agent/temp/
├── pytorch_models_td3/      ← Checkpoint định kỳ
├── pytorch_models_best_td3/ ← Model tốt nhất
├── final_models_td3/        ← Model cuối cùng
├── results_td3/             ← Eval rewards (.npy)
└── logs_td3/                ← TensorBoard logs
```

---

## ⚙️ Bước 1 — Cấu Hình Training

### 1.1 Cài đặt cơ bản (`train_td3_config.yaml`)

```yaml
train_settings:
  seed: 40                        # Cùng seed với AnhcApe → benchmark công bằng
  use_checkpoints: false          # TD3 gốc KHÔNG dùng checkpoint mechanism
  timesteps_before_training: 25000  # Random exploration trước khi train
  eval_freq: 5000                 # Evaluation mỗi 5000 timesteps
  eval_eps: 10                    # Số episode đánh giá mỗi lần eval
  max_timesteps: 5000000          # Tổng số bước training
  max_episode_steps: 500          # Giới hạn bước mỗi episode
  base_file_name: "td3_agent"     # Tiền tố tên file lưu
```

### 1.2 Hyperparameters (`td3_hyperparameters.yaml`)

```yaml
hyperparameters:
  batch_size: 256
  buffer_size: 1000000
  discount: 0.99
  tau: 0.005              # Polyak averaging (soft target update)
  policy_freq: 2          # Actor update mỗi 2 critic steps
  exploration_noise: 1.0
  exploration_noise_min: 0.1
  exploration_noise_decay_steps: 750000
  target_policy_noise: 0.2
  noise_clip: 0.5
  hidden_dim: 256
  actor_lr: 0.0003
  critic_lr: 0.0003
```

> ⚠️ Đây là các giá trị **chuẩn từ paper TD3 gốc**. Không nên thay đổi khi benchmark.

---

## 🚀 Bước 2 — Chạy Training (3 Terminal, giống hệt AnhcApe)

> ✅ **TD3 dùng ĐÚNG quy trình 3-terminal như AnhcApe.**
> Chỉ thay đổi duy nhất: Terminal 3 dùng `train_td3_agent.py` thay vì `train_anhcape_agent.py`.

### 2.1 Đảm bảo `train_td3_config.yaml` ở chế độ NEW

```yaml
load_model: false   # ← PHẢI là false khi train mới
save_date: ""       # ← Để trống
```

---

### Terminal 1 — Khởi động Gazebo simulation (không thay đổi)

```bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
cd ~/anhc_agent_ws && source install/setup.bash
source /usr/share/gazebo/setup.bash

# Dynamic environment (với pedestrians) — dùng lệnh GIỐNG HỆT khi train AnhcApe
ros2 launch anhc_agent_gazebo training_env_dynamic.launch.py \
  pedsim_start_delay:=4.0 \
  bridge_start_delay:=6.0
```

---

### Terminal 2 — DRL Environment Node (không thay đổi)

```bash
docker exec -it anhcape_ros2 bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
cd ~/anhc_agent_ws && source install/setup.bash
ros2 run anhc_agent environment.py
```

---

### Terminal 3 — Train TD3 Agent ← **Thay đổi duy nhất**

```bash
docker exec -it anhcape_ros2 bash
export ANHC_AGENT_SRC_PATH=~/anhc_agent_ws/src/src/
cd ~/anhc_agent_ws && source install/setup.bash

# Thay vì: ros2 run anhc_agent train_anhcape_agent.py
ros2 run anhc_agent train_td3_agent.py
```

---

## 🔄 Bước 3 — Resume Training (Tiếp tục sau khi bị ngắt)

### 3.1 Resume tự động (khuyến nghị)

Chỉnh `train_td3_config.yaml`:

```yaml
load_model: true   # ← Bật resume
save_date: ""      # ← Để trống → tự động tìm checkpoint MỚI NHẤT
```

Sau đó chạy lại lệnh launch như bình thường. Agent sẽ tự tìm và load file `*_train_state.json` mới nhất.

### 3.2 Resume phiên cụ thể

Nếu muốn resume một phiên training cụ thể (không phải mới nhất):

```yaml
load_model: true
save_date: "20260529_153000"   # ← Timestamp của phiên muốn resume
```

> 💡 Tìm timestamp: xem tên file trong `temp/pytorch_models_td3/`:
> ```bash
> ls ~/anhc_agent_ws/src/src/anhc_agent/temp/pytorch_models_td3/
> # td3_agent_seed_40_20260529_153000_actor.pth
> # td3_agent_seed_40_20260529_153000_train_state.json
> ```

---

## 📊 Bước 4 — Theo Dõi Training

### 4.1 TensorBoard

```bash
tensorboard --logdir ~/anhc_agent_ws/src/src/anhc_agent/temp/logs_td3
```

Truy cập: `http://localhost:6006`

**Metrics được log:**

| Tag | Ý nghĩa |
|-----|----------|
| `loss/critic` | TD3 Critic loss (MSE) |
| `loss/actor` | Actor loss (deterministic policy gradient) |
| `Q/value` | Giá trị Q trung bình |
| `train/episode_reward` | Reward mỗi training episode |
| `train/episode_steps` | Số bước mỗi training episode |
| `eval/average_reward` | Reward trung bình khi evaluation |
| `eval/best_reward` | Reward tốt nhất từ trước đến nay |

### 4.2 So sánh TD3 vs AnhcApe trên cùng TensorBoard

```bash
tensorboard --logdir_spec \
  td3:~/anhc_agent_ws/src/src/anhc_agent/temp/logs_td3,\
  anhcape:~/anhc_agent_ws/src/src/anhc_agent/temp/logs
```

---

## 💾 Bước 5 — Quản Lý Checkpoint

### Cấu trúc file được lưu

Mỗi lần evaluation, các file sau được lưu vào `pytorch_models_td3/`:

```
td3_agent_seed_40_<timestamp>_actor.pth
td3_agent_seed_40_<timestamp>_actor_target.pth
td3_agent_seed_40_<timestamp>_actor_optimizer.pth
td3_agent_seed_40_<timestamp>_critic.pth
td3_agent_seed_40_<timestamp>_critic_target.pth
td3_agent_seed_40_<timestamp>_critic_optimizer.pth
td3_agent_seed_40_<timestamp>_buffer.npz       ← Replay buffer (~4GB nếu đầy)
td3_agent_seed_40_<timestamp>_train_state.json ← Training state
```

Model tốt nhất được lưu riêng vào `pytorch_models_best_td3/`:

```
td3_agent_seed_40_<timestamp>_best_actor.pth
td3_agent_seed_40_<timestamp>_best_critic.pth
...
```

---

## ⚖️ So Sánh TD3 vs AnhcApe

| Đặc điểm | TD3 Baseline | AnhcApe |
|-----------|-------------|---------|
| **Replay Buffer** | Uniform (i.i.d. sampling) | LAP (Prioritized) |
| **Target Update** | Polyak τ=0.005 (soft) | Hard update mỗi 250 steps |
| **Encoder** | Không có | State Encoder + Fixed Encoder |
| **Critic Input** | `[state, action]` | `[state, action, zs, zsa]` |
| **Actor Input** | `state` | `[state, zs]` |
| **Q-value Clipping** | Không | Có (dynamic bounds) |
| **Checkpoint Mechanism** | Không | Có (reset on degradation) |
| **Output Dir** | `*_td3/` | `pytorch_models/` |

---

## 🛠️ Xử Lý Sự Cố

### Lỗi: "td3_hyperparameters.yaml not found"

```bash
# Kiểm tra đường dẫn
echo $ANHC_AGENT_SRC_PATH
ls $ANHC_AGENT_SRC_PATH/anhc_agent/config/td3_hyperparameters.yaml
```

### Lỗi: "train_td3_agent.py not found"

Package chưa được rebuild sau khi thêm file mới:

```bash
cd ~/anhc_agent_ws
colcon build --packages-select anhc_agent
source install/setup.bash
```

### Lỗi: "Buffer file không tìm thấy" khi Resume

Buffer (`.npz`) có thể đã bị xóa hoặc chưa được lưu. Trong trường hợp này agent sẽ tự bắt đầu với buffer trống và tiếp tục từ weights đã load. Đây là hành vi **an toàn** — training vẫn tiếp tục được.

### Kiểm tra trạng thái training đang chạy

```bash
# Xem log real-time
ros2 topic echo /rosout

# Kiểm tra node đang chạy
ros2 node list

# Kiểm tra timestep hiện tại
cat ~/anhc_agent_ws/src/src/anhc_agent/temp/pytorch_models_td3/*_train_state.json | python3 -m json.tool
```
