"""
Uniform Replay Buffer — Standard Experience Replay cho TD3 Baseline
====================================================================
Buffer tiêu chuẩn không có prioritization, đúng như thiết kế gốc
của TD3 (Fujimoto et al., 2018).

API tương thích với LAP buffer của AnhcapeAgent:
  - add(state, action, next_state, reward, done)
  - sample() → (s, a, s', r, not_done)  [tensors trên device]
  - save(directory, filename)
  - load(directory, filename)

Không có:
  - update_priority()     (chỉ có trong LAP)
  - reset_max_priority()  (chỉ có trong LAP)
"""

import os
import numpy as np
import torch


class UniformReplayBuffer:
    """
    Standard Uniform Experience Replay Buffer.

    Lưu trữ transitions (s, a, s', r, done) trong một circular buffer
    và sample ngẫu nhiên đều (uniform) khi training.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        max_size: int = int(1e6),
        batch_size: int = 256,
        max_action: float = 1.0,
        normalize_actions: bool = True,
    ):
        """
        Args:
            state_dim:         Số chiều của state vector.
            action_dim:        Số chiều của action vector.
            device:            torch.device ('cuda' hoặc 'cpu').
            max_size:          Dung lượng tối đa của buffer (số transitions).
            batch_size:        Số samples mỗi lần gọi sample().
            max_action:        Giá trị tối đa của action (dùng để normalize).
            normalize_actions: Nếu True, normalize action về [-1, 1] khi lưu.
        """
        max_size = int(max_size)
        self.max_size   = max_size
        self.batch_size = batch_size
        self.device     = device
        self.ptr        = 0   # con trỏ vị trí ghi tiếp theo
        self.size       = 0   # số transitions hiện có

        # Pre-allocate numpy arrays (efficient CPU storage)
        self.state      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.action     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.reward     = np.zeros((max_size, 1),          dtype=np.float32)
        self.not_done   = np.zeros((max_size, 1),          dtype=np.float32)

        # Action normalization scale
        self._action_scale = max_action if normalize_actions else 1.0

    # ──────────────────────────────────────────────────────────────────────────
    # Add Transition
    # ──────────────────────────────────────────────────────────────────────────
    def add(
        self,
        state,
        action,
        next_state,
        reward: float,
        done: float,
    ):
        """
        Thêm một transition vào buffer.

        action được normalize về [-1, 1] nếu normalize_actions=True.
        done=1.0 khi episode kết thúc (terminal), 0.0 nếu không.
        """
        self.state[self.ptr]      = state
        self.action[self.ptr]     = action / self._action_scale
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr]     = reward
        self.not_done[self.ptr]   = 1.0 - done

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # ──────────────────────────────────────────────────────────────────────────
    # Sample Minibatch (Uniform Random)
    # ──────────────────────────────────────────────────────────────────────────
    def sample(self) -> tuple:
        """
        Sample một minibatch ngẫu nhiên đều từ buffer.

        Returns:
            Tuple (state, action, next_state, reward, not_done) — torch.Tensor
            trên self.device, shape: (batch_size, dim).
        """
        idx = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.tensor(self.state[idx],      dtype=torch.float, device=self.device),
            torch.tensor(self.action[idx],     dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[idx], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[idx],     dtype=torch.float, device=self.device),
            torch.tensor(self.not_done[idx],   dtype=torch.float, device=self.device),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────────────
    def save(self, save_folder: str, file_name: str):
        """Lưu nội dung buffer ra file .npz (compressed)."""
        save_path = os.path.join(save_folder, f"{file_name}_buffer.npz")
        np.savez_compressed(
            save_path,
            state      = self.state[:self.size],
            action     = self.action[:self.size],
            next_state = self.next_state[:self.size],
            reward     = self.reward[:self.size],
            not_done   = self.not_done[:self.size],
            ptr        = np.array([self.ptr]),
            size       = np.array([self.size]),
        )

    def load(self, save_folder: str, file_name: str) -> bool:
        """
        Load buffer từ file .npz.

        Returns:
            True nếu load thành công, False nếu file không tồn tại.
        """
        load_path = os.path.join(save_folder, f"{file_name}_buffer.npz")
        if not os.path.exists(load_path):
            print(
                f"[UniformReplayBuffer] File không tìm thấy: {load_path}. "
                "Bắt đầu với buffer trống."
            )
            return False

        data = np.load(load_path)
        self.size = int(data["size"][0])
        self.ptr  = int(data["ptr"][0])

        self.state[:self.size]      = data["state"]
        self.action[:self.size]     = data["action"]
        self.next_state[:self.size] = data["next_state"]
        self.reward[:self.size]     = data["reward"]
        self.not_done[:self.size]   = data["not_done"]

        return True
