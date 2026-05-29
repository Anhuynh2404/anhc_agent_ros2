#!/usr/bin/env python3
"""
TD3 Baseline Agent — Twin Delayed Deep Deterministic Policy Gradient
=====================================================================
Implementation thuần túy theo paper gốc:
  "Addressing Function Approximation Error in Actor-Critic Methods"
  Fujimoto et al., ICML 2018  (https://arxiv.org/abs/1802.09477)

Drop-in replacement cho AnhcapeAgent trong cùng training loop.
Không sử dụng bất kỳ thành phần nào của AnhcApe (Encoder, LAP, Fixed-Encoder, v.v.).
"""

import os
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ament_index_python.packages import get_package_share_directory

PACKAGE_SHARE_DIR = get_package_share_directory('anhc_agent')
UTILS_DIR = os.path.join(PACKAGE_SHARE_DIR, 'scripts', 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

import td3_uniform_buffer as uniform_buffer


# ──────────────────────────────────────────────────────────────────────────────
# Actor Network
# ──────────────────────────────────────────────────────────────────────────────
class TD3Actor(nn.Module):
    """
    Policy network π_θ(s): s → a ∈ [-1, 1]^action_dim

    Kiến trúc: 3 FC layers với ReLU activation, tanh output.
    Đây là kiến trúc chuẩn từ paper gốc TD3.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(TD3Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


# ──────────────────────────────────────────────────────────────────────────────
# Twin Critic Network
# ──────────────────────────────────────────────────────────────────────────────
class TD3Critic(nn.Module):
    """
    Twin Q-networks Q_φ1(s, a) và Q_φ2(s, a).

    Cả hai critic được train song song nhưng trong cùng một module,
    giúp tái sử dụng forward pass. Điều này là standard trong TD3.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(TD3Critic, self).__init__()

        # Q1 network
        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)

        # Q2 network (twin)
        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả về cả hai Q-values (dùng để tính critic loss)."""
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)

        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Trả về chỉ Q1 (dùng để tính actor loss — gradient chỉ qua Q1)."""
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        return self.q1_l3(q1)


# ──────────────────────────────────────────────────────────────────────────────
# TD3 Agent
# ──────────────────────────────────────────────────────────────────────────────
class TD3Agent(object):
    """
    Standard TD3 Agent — drop-in replacement cho AnhcapeAgent.

    Các tính năng chính của TD3 gốc:
      1. Twin Critics (giảm overestimation bias)
      2. Delayed Policy Updates (actor cập nhật mỗi policy_freq bước)
      3. Target Policy Smoothing (thêm noise vào target action)
      4. Soft target updates (polyak averaging)

    API tương thích hoàn toàn với AnhcapeAgent:
      - select_action(state, use_checkpoint=False, use_exploration=True)
      - train()
      - train_and_checkpoint(ep_timesteps, ep_return)  [no-op, giữ interface]
      - save(directory, filename)
      - load(directory, filename)
      - get_agent_state() / restore_agent_state(state_dict)
      - replay_buffer.add(s, a, s', r, done)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hp: dict,
        log_dir: str = None,
    ):
        self.hyperparameters = hp

        # ── Generic hyperparameters ─────────────────────────────────────────
        self.discount          = hp["discount"]
        self.batch_size        = hp["batch_size"]
        self.buffer_size       = hp["buffer_size"]
        self.tau               = hp["tau"]           # polyak averaging coeff
        self.policy_freq       = hp["policy_freq"]   # delayed policy update

        # ── Exploration noise (decay schedule) ──────────────────────────────
        self.exploration_noise           = hp["exploration_noise"]
        self.exploration_noise_min       = hp["exploration_noise_min"]
        self.exploration_noise_decay_steps = hp["exploration_noise_decay_steps"]

        # ── Target policy smoothing ─────────────────────────────────────────
        self.target_policy_noise = hp["target_policy_noise"]
        self.noise_clip          = hp["noise_clip"]

        # ── Network architecture ─────────────────────────────────────────────
        hidden_dim   = hp["hidden_dim"]
        actor_lr     = hp["actor_lr"]
        critic_lr    = hp["critic_lr"]

        self.max_action = max_action
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Actor ───────────────────────────────────────────────────────────
        self.actor = TD3Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )

        # ── Twin Critic ──────────────────────────────────────────────────────
        self.critic = TD3Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # ── Uniform Replay Buffer ────────────────────────────────────────────
        self.replay_buffer = uniform_buffer.UniformReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            max_size=self.buffer_size,
            batch_size=self.batch_size,
            max_action=max_action,
            normalize_actions=True,
        )

        # ── Internal counters ────────────────────────────────────────────────
        self.training_steps = 0

        # ── TensorBoard writer ───────────────────────────────────────────────
        self.log_dir = log_dir
        if self.log_dir is not None:
            from datetime import datetime
            run_subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_dir = os.path.join(log_dir, run_subdir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Action Selection
    # ──────────────────────────────────────────────────────────────────────────
    def select_action(
        self,
        state,
        use_checkpoint: bool = False,   # kept for API compatibility (unused)
        use_exploration: bool = True,
    ):
        """
        Chọn action dựa trên policy hiện tại.

        use_checkpoint: Tham số giữ lại để tương thích API với AnhcapeAgent
                        (TD3 gốc không có checkpoint mechanism).
        """
        with torch.no_grad():
            state_t = torch.tensor(
                state.reshape(1, -1), dtype=torch.float, device=self.device
            )
            action = self.actor(state_t)

            if use_exploration:
                # Linear decay exploration noise (giống AnhcapeAgent)
                if self.exploration_noise > self.exploration_noise_min:
                    self.exploration_noise -= (
                        1.0 - self.exploration_noise_min
                    ) / self.exploration_noise_decay_steps
                action = action + torch.randn_like(action) * self.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    # ──────────────────────────────────────────────────────────────────────────
    # Training Step
    # ──────────────────────────────────────────────────────────────────────────
    def train(self):
        """
        Một bước gradient update theo thuật toán TD3 gốc.

        Sequence:
          1. Sample minibatch từ Uniform Replay Buffer
          2. Tính target Q-value với target policy smoothing
          3. Update twin critics bằng MSE loss
          4. Delayed actor update (mỗi policy_freq bước)
          5. Soft target network update (polyak averaging)
        """
        if self.replay_buffer.size < self.batch_size:
            return

        self.training_steps += 1
        state, action, next_state, reward, not_done = self.replay_buffer.sample()

        # ── Step 1: Compute target Q-value ──────────────────────────────────
        with torch.no_grad():
            # Target Policy Smoothing: thêm clipped noise vào next action
            noise = (
                torch.randn_like(action) * self.target_policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            # Twin critic targets: lấy min để tránh overestimation
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # ── Step 2: Update Twin Critics ──────────────────────────────────────
        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Step 3: Delayed Actor Update ────────────────────────────────────
        if self.training_steps % self.policy_freq == 0:
            # Maximize Q1 (deterministic policy gradient)
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ── Step 4: Soft Target Network Update (Polyak Averaging) ───────
            with torch.no_grad():
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

            # Log actor loss
            self.writer.add_scalar("loss/actor", actor_loss.item(), self.training_steps)

        # ── Logging ──────────────────────────────────────────────────────────
        self.writer.add_scalar("loss/critic", critic_loss.item(), self.training_steps)
        self.writer.add_scalar(
            "Q/value", torch.min(current_Q1, current_Q2).mean().item(), self.training_steps
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Compatibility shim: train_and_checkpoint
    # (AnhcapeAgent sử dụng trong loop khi use_checkpoints=True)
    # TD3 gốc không có checkpoint mechanism → chạy train() mỗi timestep.
    # Để tương thích, ta gọi train() cho mỗi bước đã tích lũy.
    # ──────────────────────────────────────────────────────────────────────────
    def train_and_checkpoint(self, ep_timesteps: int, ep_return: float):
        """
        API compatibility shim.

        AnhcapeAgent dùng method này khi use_checkpoints=True.
        TD3 gốc không có checkpoint reset mechanism → ta gọi train()
        cho mỗi timestep trong episode vừa kết thúc.
        """
        for _ in range(ep_timesteps):
            self.train()

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence: Save / Load
    # ──────────────────────────────────────────────────────────────────────────
    def save(self, directory: str, filename: str):
        """Lưu tất cả model weights, optimizer states và replay buffer."""
        torch.save(
            self.actor.state_dict(),
            os.path.join(directory, f"{filename}_actor.pth"),
        )
        torch.save(
            self.actor_target.state_dict(),
            os.path.join(directory, f"{filename}_actor_target.pth"),
        )
        torch.save(
            self.actor_optimizer.state_dict(),
            os.path.join(directory, f"{filename}_actor_optimizer.pth"),
        )
        torch.save(
            self.critic.state_dict(),
            os.path.join(directory, f"{filename}_critic.pth"),
        )
        torch.save(
            self.critic_target.state_dict(),
            os.path.join(directory, f"{filename}_critic_target.pth"),
        )
        torch.save(
            self.critic_optimizer.state_dict(),
            os.path.join(directory, f"{filename}_critic_optimizer.pth"),
        )

        # Save replay buffer
        self.replay_buffer.save(directory, filename)

    def load(self, directory: str, filename: str):
        """Load tất cả model weights, optimizer states và replay buffer."""
        map_loc = self.device

        self.actor.load_state_dict(
            torch.load(
                os.path.join(directory, f"{filename}_actor.pth"),
                map_location=map_loc,
            )
        )
        self.actor_target.load_state_dict(
            torch.load(
                os.path.join(directory, f"{filename}_actor_target.pth"),
                map_location=map_loc,
            )
        )
        self.actor_optimizer.load_state_dict(
            torch.load(
                os.path.join(directory, f"{filename}_actor_optimizer.pth"),
                map_location=map_loc,
            )
        )
        self.critic.load_state_dict(
            torch.load(
                os.path.join(directory, f"{filename}_critic.pth"),
                map_location=map_loc,
            )
        )
        self.critic_target.load_state_dict(
            torch.load(
                os.path.join(directory, f"{filename}_critic_target.pth"),
                map_location=map_loc,
            )
        )
        self.critic_optimizer.load_state_dict(
            torch.load(
                os.path.join(directory, f"{filename}_critic_optimizer.pth"),
                map_location=map_loc,
            )
        )

        # Load replay buffer
        self.replay_buffer.load(directory, filename)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal State for Resume
    # ──────────────────────────────────────────────────────────────────────────
    def get_agent_state(self) -> dict:
        """Lấy toàn bộ internal state để lưu vào train_state.json khi resume."""
        return {
            "training_steps": self.training_steps,
            "exploration_noise": self.exploration_noise,
        }

    def restore_agent_state(self, state_dict: dict):
        """Khôi phục internal state từ train_state.json khi resume."""
        self.training_steps   = state_dict.get("training_steps", 0)
        self.exploration_noise = state_dict.get(
            "exploration_noise", self.exploration_noise
        )
