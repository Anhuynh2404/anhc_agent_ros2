#!/usr/bin/env python3

import os
import sys
import glob
import json
import rclpy
import time
from datetime import datetime

import torch
import numpy as np
from ament_index_python.packages import get_package_share_directory

PACKAGE_SHARE_DIR = get_package_share_directory('anhc_agent')
UTILS_DIR = os.path.join(PACKAGE_SHARE_DIR, 'scripts', 'utils')
ENV_DIR = os.path.join(PACKAGE_SHARE_DIR, 'scripts', 'environment')
POLICY_DIR = os.path.join(PACKAGE_SHARE_DIR, 'scripts', 'policy')
for module_dir in (UTILS_DIR, ENV_DIR, POLICY_DIR):
    if module_dir not in sys.path:
        sys.path.append(module_dir)

from anhcape_agent import AnhcapeAgent
from environment_interface import EnvInterface
from file_manager import DirectoryManager, load_yaml


class TrainAnhcape(EnvInterface):
    def __init__(self):
        super().__init__("train_anhcape_node")

        # Load training config parameters
        anhc_agent_src_path_env = "ANHC_AGENT_SRC_PATH"
        anhc_agent_src_path = os.getenv(anhc_agent_src_path_env)
        if anhc_agent_src_path is None:
            self.get_logger().error(
                f"Environment variable: {anhc_agent_src_path_env}, is not set"
            )
        anhc_agent_pkg_path = os.path.join(anhc_agent_src_path, "anhc_agent")

        self.hyperparameters_path = os.path.join(
            anhc_agent_pkg_path, "config", "hyperparameters.yaml"
        )
        self.train_config_file_path = os.path.join(
            anhc_agent_pkg_path, "config", "train_config.yaml"
        )

        # Load config file
        try:
            training_settings = load_yaml(self.train_config_file_path)["train_settings"]
        except Exception as e:
            self.get_logger().error(f"Unable to load config file: {e}")
            sys.exit(-1)
        # Extract training settings
        self.seed = training_settings["seed"]
        self.max_episode_steps = training_settings["max_episode_steps"]
        self.load_model = training_settings["load_model"]
        self.max_timesteps = training_settings["max_timesteps"]
        self.use_checkpoints = training_settings["use_checkpoints"]
        self.eval_freq = training_settings["eval_freq"]
        self.timesteps_before_training = training_settings["timesteps_before_training"]
        self.eval_eps = training_settings["eval_eps"]
        self.base_file_name = training_settings["base_file_name"]
        
        save_date = training_settings.get("save_date", "").strip()
        if save_date:
            # Dùng tên chỉ định: resume từ phiên cụ thể theo save_date
            self.file_name = f"{self.base_file_name}_seed_{self.seed}_{save_date}"
        else:
            # Tự động tạo timestamp mới (dùng khi train mới)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file_name = f"{self.base_file_name}_seed_{self.seed}_{current_time}"

        # Setup directories for saving models, results and logs
        temp_dir_path = os.path.join(anhc_agent_src_path, "anhc_agent", "temp")
        self.pytorch_models_dir = os.path.join(temp_dir_path, "pytorch_models")
        self.pytorch_models_best_dir = os.path.join(temp_dir_path, "pytorch_models_best")
        self.final_models_dir = os.path.join(temp_dir_path, "final_models")
        self.results_dir = os.path.join(temp_dir_path, "results")
        self.log_dir = os.path.join(temp_dir_path, "logs")
        # Make directories
        self.create_directories()

        # Set seed
        self.set_env_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Initialize the agent
        try:
            hyperparameters = load_yaml(self.hyperparameters_path)["hyperparameters"]
        except Exception as e:
            self.get_logger().error(f"Unable to load config file: {e}")
            sys.exit(-1)
        self.state_dim, self.action_dim, self.max_action = self.get_dimensions()
        self.rl_agent = AnhcapeAgent(
            self.state_dim,
            self.action_dim,
            self.max_action,
            hyperparameters,
            self.log_dir,
        )

        # ── Load model để resume ─────────────────────────────────────────────
        if self.load_model:
            # Nếu save_date trống → tự động tìm checkpoint mới nhất
            if not save_date:
                detected = self._auto_detect_file_name()
                if detected:
                    self.file_name = detected
                    self.get_logger().info(
                        f"[RESUME] Tự động phát hiện checkpoint: {self.file_name}"
                    )
                else:
                    self.get_logger().warning(
                        "[RESUME] Không tìm thấy checkpoint nào → bắt đầu training mới"
                    )
                    self.load_model = False
            # Load weights
            if self.load_model:
                try:
                    self.rl_agent.load(self.pytorch_models_dir, self.file_name)
                    self.get_logger().info(f"[RESUME] Weights đã load: {self.file_name}")
                except Exception as e:
                    self.get_logger().error(
                        f"[RESUME] THẤT BẠI khi load model '{self.file_name}': {e}\n"
                        f"         → Kiểm tra lại base_file_name và save_date trong train_config.yaml"
                    )
                    sys.exit(-1)
        else:
            self.get_logger().info(f"[NEW] Training mới từ đầu: {self.file_name}")
        # ─────────────────────────────────────────────────────────────────────

        # Flag to indicate that the training is done
        self.done_training = False

        self.log_training_setting_data()

    def log_training_setting_data(self):
        """Log general info at the start of training"""
        self.border = "+" + "-" * 80 + "+"
        self.get_logger().info(self.border)
        self.get_logger().info(f"| File name: {self.file_name}: Seed: {self.seed}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Results will be saved in:")
        self.get_logger().info(f"|  {self.pytorch_models_dir}")
        self.get_logger().info(f"|  {self.final_models_dir}")
        self.get_logger().info(f"|  {self.results_dir}")
        self.get_logger().info(f"|  {self.log_dir}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Environment")
        self.get_logger().info(self.border)
        self.get_logger().info(f"| State Dim: {self.state_dim}")
        self.get_logger().info(f"| Action Dim: {self.action_dim}")
        self.get_logger().info(f"| Max Action: {self.max_action}")
        self.get_logger().info(self.border)

    def create_directories(self):
        """Create directories for saving models (preserve existing data for resume)"""
        directories = [
            self.pytorch_models_dir,
            self.pytorch_models_best_dir,
            self.final_models_dir,
            self.results_dir,
            self.log_dir,
        ]
        for dir in directories:
            dir_manager = DirectoryManager(dir)
            dir_manager.create()  # create only if not present, do NOT delete

    @property
    def training_state_path(self):
        return os.path.join(self.pytorch_models_dir, f"{self.file_name}_train_state.json")

    def _auto_detect_file_name(self):
        """Tự động tìm file train_state.json mới nhất trong pytorch_models_dir.
        Đọc field 'file_name' bên trong để lấy tên chính xác.
        """
        pattern = os.path.join(
            self.pytorch_models_dir,
            f"{self.base_file_name}_seed_{self.seed}_*_train_state.json",
        )
        matches = glob.glob(pattern)
        if not matches:
            return None
        # Lấy file được chỉnh sửa gần nhất
        newest = max(matches, key=os.path.getmtime)
        try:
            with open(newest, "r") as f:
                state = json.load(f)
            # Ưu tiên đọc file_name lưu bên trong JSON (tuyệt đối chính xác)
            stored_name = state.get("file_name")
            if stored_name:
                return stored_name
            # Fallback: suy ra từ tên file nếu chưa có field file_name
            return os.path.basename(newest).replace("_train_state.json", "")
        except Exception as e:
            self.get_logger().warning(f"Không đọc được {newest}: {e}")
            return None

    def save_training_state(self, t, ep_num, epoch, evals):
        """Lưu toàn bộ training state để resume chính xác.
        Bao gồm: file_name, timestep, agent internal state.
        """
        state = {
            "file_name": self.file_name,       # ← KEY: lưu tên để tự detect khi resume
            "total_timesteps": t,
            "episode_num": ep_num,
            "epoch": epoch,
            "evals": evals,
            "agent_state": self.rl_agent.get_agent_state(),  # ← exploration noise, Q-bounds...
        }
        with open(self.training_state_path, "w") as f:
            json.dump(state, f, indent=4)
        self.get_logger().info(f"Training state saved at T={t}, Episode={ep_num}")

    def load_training_state(self):
        """Load toàn bộ training state để resume.
        Khôi phục: timestep, episode, epoch, eval history, và agent internal state.
        """
        if not self.load_model:
            # Training mới → bắt đầu từ đầu, không cần đọc state
            return 1, 1, 1, []

        if os.path.exists(self.training_state_path):
            with open(self.training_state_path, "r") as f:
                state = json.load(f)
            t_start = state.get("total_timesteps", 1)
            ep_num = state.get("episode_num", 1)
            epoch = state.get("epoch", 1)
            evals = state.get("evals", [])

            # Khôi phục agent internal state (exploration noise, Q-bounds, v.v.)
            agent_state = state.get("agent_state", {})
            if agent_state:
                self.rl_agent.restore_agent_state(agent_state)
                self.get_logger().info(
                    f"[RESUME] Agent state khôi phục: "
                    f"training_steps={agent_state.get('training_steps', 0)}, "
                    f"exploration_noise={agent_state.get('exploration_noise', '?'):.4f}"
                )
            else:
                self.get_logger().warning(
                    "[RESUME] train_state.json không có 'agent_state' → "
                    "exploration_noise và Q-bounds sẽ reset về mặc định"
                )

            self.get_logger().info(
                f"[RESUME] ✓ Tiếp tục từ T={t_start}, Episode={ep_num}, Epoch={epoch}"
            )
            return t_start, ep_num, epoch, evals

        self.get_logger().error(
            f"[RESUME] KHÔNG TÌM THẤY: {self.training_state_path}\n"
            f"         Kiểm tra lại load_model và save_date trong train_config.yaml"
        )
        sys.exit(-1)

    def save_models(self, directory, file_name):
        """Save the models at the given step"""
        self.rl_agent.save(directory, file_name)
        self.get_logger().info("Models updated")

    def train_online(self):
        """Train the agent online"""
        start_time = time.time()

        # Load training state if resuming
        t_start, ep_num, epoch, evals = self.load_training_state()
        # Khôi phục kỷ lục eval reward từ lịch sử (quan trọng khi resume)
        self.best_eval_reward = max(evals) if evals else float("-inf")
        self.get_logger().info(f"| Best eval reward so far: {self.best_eval_reward:.3f}")
        timesteps_since_eval = 0
        allow_train = t_start > self.timesteps_before_training

        state, ep_finished = self.reset(), False
        ep_total_reward, ep_timesteps = 0, 0

        for t in range(t_start, self.max_timesteps + 1):

            if allow_train:
                action = self.rl_agent.select_action(np.array(state))
            else:
                action = self.sample_action_space()

            # Act
            next_state, reward, ep_finished, _ = self.step(action)

            ep_total_reward += reward
            ep_timesteps += 1

            done = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0
            self.rl_agent.replay_buffer.add(state, action, next_state, reward, done)

            state = next_state

            if allow_train and not self.use_checkpoints:
                self.rl_agent.train()

            if ep_finished or ep_timesteps == self.max_episode_steps:
                self.get_logger().info(
                    f"Total T: {t} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}"
                )
                # Log episode reward and length to TensorBoard
                self.rl_agent.writer.add_scalar("train/episode_reward", ep_total_reward, t)
                self.rl_agent.writer.add_scalar("train/episode_steps", ep_timesteps, t)

                if allow_train and self.use_checkpoints:
                    self.rl_agent.train_and_checkpoint(ep_timesteps, ep_total_reward)

                if allow_train and timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    # Save models and training state
                    self.save_models(self.pytorch_models_dir, self.file_name)
                    self.save_training_state(t, ep_num, epoch, evals)
                    self.evaluate_and_print(evals, epoch, start_time, t)
                    epoch += 1

                if t >= self.timesteps_before_training:
                    allow_train = True

                state, done = self.reset(), False
                ep_total_reward, ep_timesteps = 0, 0
                ep_num += 1

            timesteps_since_eval += 1
        # Indicate that the training is done
        self.done_training = True

    def evaluate_and_print(self, evals, epoch, start_time, t):
        """Evaluate the agent and print the results"""

        self.get_logger().info(self.border)
        self.get_logger().info(f"| Evaluation at epoch: {epoch}")
        self.get_logger().info(
            f"| Total time passed: {round((time.time()-start_time)/60.,2)} min(s)"
        )

        total_reward = np.zeros(self.eval_eps)
        for ep in range(self.eval_eps):
            state, done = self.reset(), False
            ep_timesteps = 0
            while not done and ep_timesteps < self.max_episode_steps:
                action = self.rl_agent.select_action(
                    np.array(state), self.use_checkpoints, use_exploration=False
                )
                # Act
                state, reward, done, _ = self.step(action)
                total_reward[ep] += reward
                ep_timesteps += 1

        avg_reward = total_reward.mean()
        self.get_logger().info(
            f"| Average reward over {self.eval_eps} episodes: {avg_reward:.3f}"
        )

        # ── Lưu model tốt nhất ──────────────────────────────────────────────
        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            best_file_name = f"{self.file_name}_best"
            self.save_models(self.pytorch_models_best_dir, best_file_name)
            self.get_logger().info(
                f"| ★ NEW BEST MODEL saved! Reward: {avg_reward:.3f} at T={t}, Epoch={epoch}"
            )
            self.rl_agent.writer.add_scalar("eval/best_reward", avg_reward, t)
        # ────────────────────────────────────────────────────────────────────

        self.get_logger().info(f"| Best reward so far: {self.best_eval_reward:.3f}")
        self.get_logger().info(self.border)
        evals.append(avg_reward)
        np.save(f"{self.results_dir}/{self.file_name}", evals)
        self.rl_agent.writer.add_scalar("eval/average_reward", avg_reward, t)


def main(args=None):
    # Initialize the ROS2 communication
    rclpy.init(args=args)
    # Initialize the node
    train_anhcape_node = TrainAnhcape()
    # Start training
    train_anhcape_node.train_online()
    try:
        while rclpy.ok() and not train_anhcape_node.done_training:
            rclpy.spin_once(train_anhcape_node)
    except KeyboardInterrupt as e:
        train_anhcape_node.get_logger().warning(f"KeyboardInterrupt: {e}")
    finally:
        train_anhcape_node.get_logger().info("rclpy, shutting down...")
        train_anhcape_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
