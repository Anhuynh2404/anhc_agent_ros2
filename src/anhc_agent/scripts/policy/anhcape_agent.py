import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ament_index_python.packages import get_package_share_directory

PACKAGE_SHARE_DIR = get_package_share_directory('anhc_agent')
UTILS_DIR = os.path.join(PACKAGE_SHARE_DIR, 'scripts', 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

import buffer


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class AnhcapeActor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
        super(AnhcapeActor, self).__init__()

        self.activ = activ

        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class AnhcapeEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(AnhcapeEncoder, self).__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class AnhcapeCritic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(AnhcapeCritic, self).__init__()

        self.activ = activ

        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)


class AnhcapeAgent(object):
    def __init__(self, state_dim, action_dim, max_action, hp, log_dir=None):
        # Hyperparameters
        self.hyperparameters = self.prep_hyperparameters(hp)

        # Generic
        self.discount = self.hyperparameters["discount"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.buffer_size = self.hyperparameters["buffer_size"]
        self.target_update_rate = self.hyperparameters["target_update_rate"]
        self.exploration_noise = self.hyperparameters["exploration_noise"]
        self.exploration_noise_min = self.hyperparameters["exploration_noise_min"]
        self.exploration_noise_decay_steps = self.hyperparameters[
            "exploration_noise_decay_steps"
        ]

        # TD3
        self.noise_clip = self.hyperparameters["noise_clip"]
        self.policy_freq = self.hyperparameters["policy_freq"]
        self.target_policy_noise = self.hyperparameters["target_policy_noise"]

        # LAP
        self.alpha = self.hyperparameters["alpha"]
        self.min_priority = self.hyperparameters["min_priority"]

        # Checkpointing
        self.reset_weight = self.hyperparameters["reset_weight"]
        self.steps_before_checkpointing = self.hyperparameters[
            "steps_before_checkpointing"
        ]
        self.max_eps_when_checkpointing = self.hyperparameters[
            "max_eps_when_checkpointing"
        ]

        # Encoder Model
        self.zs_dim = self.hyperparameters["zs_dim"]
        self.enc_hdim = self.hyperparameters["enc_hdim"]
        self.enc_activ = self.hyperparameters["enc_activ"]
        self.encoder_lr = self.hyperparameters["encoder_lr"]

        # Actor Model
        self.actor_hdim = self.hyperparameters["actor_hdim"]
        self.actor_activ = self.hyperparameters["actor_activ"]
        self.actor_lr = self.hyperparameters["actor_lr"]

        # Critic Model
        self.critic_hdim = self.hyperparameters["critic_hdim"]
        self.critic_activ = self.hyperparameters["critic_activ"]
        self.critic_lr = self.hyperparameters["critic_lr"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = AnhcapeActor(
            state_dim, action_dim, self.zs_dim, self.actor_hdim, self.actor_activ
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr
        )
        self.actor_target = AnhcapeActor(
            state_dim, action_dim, self.zs_dim, self.actor_hdim, self.actor_activ
        ).to(self.device)

        self.critic = AnhcapeCritic(
            state_dim, action_dim, self.zs_dim, self.critic_hdim, self.critic_activ
        ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )
        self.critic_target = AnhcapeCritic(
            state_dim, action_dim, self.zs_dim, self.critic_hdim, self.critic_activ
        ).to(self.device)

        self.encoder = AnhcapeEncoder(
            state_dim, action_dim, self.zs_dim, self.enc_hdim, self.enc_activ
        ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=self.encoder_lr
        )
        self.fixed_encoder = AnhcapeEncoder(
            state_dim, action_dim, self.zs_dim, self.enc_hdim, self.enc_activ
        ).to(self.device)
        self.fixed_encoder_target = AnhcapeEncoder(
            state_dim, action_dim, self.zs_dim, self.enc_hdim, self.enc_activ
        ).to(self.device)

        self.checkpoint_actor = AnhcapeActor(
            state_dim, action_dim, self.zs_dim, self.actor_hdim, self.actor_activ
        ).to(self.device)
        self.checkpoint_encoder = AnhcapeEncoder(
            state_dim, action_dim, self.zs_dim, self.enc_hdim, self.enc_activ
        ).to(self.device)

        self.replay_buffer = buffer.LAP(
            state_dim,
            action_dim,
            self.device,
            self.buffer_size,
            self.batch_size,
            max_action,
            normalize_actions=True,
            prioritized=True,
        )

        self.max_action = max_action

        self.training_steps = 0

        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0

        # Writer
        self.log_dir = log_dir
        if self.log_dir is not None:
            from datetime import datetime
            run_subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_dir = os.path.join(log_dir, run_subdir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    @staticmethod
    def prep_hyperparameters(hyperparameters):
        """Pre-proccess hyperparameters"""
        activation_functions = {
            "elu": F.elu,
            "relu": F.relu,
        }
        hyperparameters["enc_activ"] = activation_functions[
            hyperparameters["enc_activ"]
        ]
        hyperparameters["critic_activ"] = activation_functions[
            hyperparameters["critic_activ"]
        ]
        hyperparameters["actor_activ"] = activation_functions[
            hyperparameters["actor_activ"]
        ]
        return hyperparameters

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(
                state.reshape(1, -1), dtype=torch.float, device=self.device
            )

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs)

            if use_exploration:
                if self.exploration_noise > self.exploration_noise_min:
                    self.exploration_noise -= (
                        1 - self.exploration_noise_min
                    ) / self.exploration_noise_decay_steps
                action = action + torch.randn_like(action) * self.exploration_noise

            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def train(self):
        self.training_steps += 1

        state, action, next_state, reward, not_done = self.replay_buffer.sample()

        """******************************************
		** Update Encoder
		******************************************"""
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)

        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        """******************************************
		** Update Critic
		******************************************"""
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (
                self.actor_target(next_state, fixed_target_zs) + noise
            ).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(
                fixed_target_zs, next_action
            )

            Q_target = self.critic_target(
                next_state, next_action, fixed_target_zsa, fixed_target_zs
            ).min(1, keepdim=True)[0]
            Q_avg = Q_target
            Q_target = reward + not_done * self.discount * Q_target.clamp(
                self.min_target, self.max_target
            )

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)
        td_loss = (Q - Q_target).abs()
        critic_loss = LAP_huber(td_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        """******************************************
		** Update LAP
		******************************************"""
        priority = td_loss.max(1)[0].clamp(min=self.min_priority).pow(self.alpha)
        self.replay_buffer.update_priority(priority)

        """******************************************
		** Update Actor
		******************************************"""
        if self.training_steps % self.policy_freq == 0:
            actor = self.actor(state, fixed_zs)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
            Q = self.critic(state, actor, fixed_zsa, fixed_zs)

            actor_loss = -Q.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        """******************************************
		** Update Iteration
		******************************************"""
        if self.training_steps % self.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        """******************************************
		** Write new values to tensorboard
		******************************************"""
        self.writer.add_scalar("loss", critic_loss, self.training_steps)
        self.writer.add_scalar("Q", torch.mean(Q_avg), self.training_steps)
        self.writer.add_scalar("Q_max", self.max, self.training_steps)

    def train_and_checkpoint(self, ep_timesteps, ep_return):
        """If using checkpoints: run when each episode terminates"""
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset()

    def train_and_reset(self):
        """Batch training"""
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.steps_before_checkpointing:
                self.best_min_return *= self.reset_weight
                self.max_eps_before_update = self.max_eps_when_checkpointing

            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

    def save(self, directory, filename):
        """Save model parameters"""
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(
            self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth"
        )
        torch.save(
            self.actor_optimizer.state_dict(),
            f"{directory}/{filename}_actor_optimizer.pth",
        )

        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(
            self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth"
        )
        torch.save(
            self.critic_optimizer.state_dict(),
            f"{directory}/{filename}_critic_optimizer.pth",
        )

        torch.save(self.encoder.state_dict(), f"{directory}/{filename}_encoder.pth")
        torch.save(
            self.fixed_encoder.state_dict(), f"{directory}/{filename}_fixed_encoder.pth"
        )
        torch.save(
            self.fixed_encoder_target.state_dict(),
            f"{directory}/{filename}_fixed_encoder_target.pth",
        )
        torch.save(
            self.encoder_optimizer.state_dict(),
            f"{directory}/{filename}_encoder_optimizer.pth",
        )

        torch.save(
            self.checkpoint_actor.state_dict(),
            f"{directory}/{filename}_checkpoint_actor.pth",
        )
        torch.save(
            self.checkpoint_encoder.state_dict(),
            f"{directory}/{filename}_checkpoint_encoder.pth",
        )
        
        # Save replay buffer
        self.replay_buffer.save(directory, filename)

    def load(self, directory, filename):
        """Load model parameters"""
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.actor_target.load_state_dict(
            torch.load(f"{directory}/{filename}_actor_target.pth")
        )
        self.actor_optimizer.load_state_dict(
            torch.load(f"{directory}/{filename}_actor_optimizer.pth")
        )

        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.critic_target.load_state_dict(
            torch.load(f"{directory}/{filename}_critic_target.pth")
        )
        self.critic_optimizer.load_state_dict(
            torch.load(f"{directory}/{filename}_critic_optimizer.pth")
        )

        self.encoder.load_state_dict(torch.load(f"{directory}/{filename}_encoder.pth"))
        self.fixed_encoder.load_state_dict(
            torch.load(f"{directory}/{filename}_fixed_encoder.pth")
        )
        self.fixed_encoder_target.load_state_dict(
            torch.load(f"{directory}/{filename}_fixed_encoder_target.pth")
        )
        self.encoder_optimizer.load_state_dict(
            torch.load(f"{directory}/{filename}_encoder_optimizer.pth")
        )

        self.checkpoint_actor.load_state_dict(
            torch.load(f"{directory}/{filename}_checkpoint_actor.pth")
        )
        self.checkpoint_encoder.load_state_dict(
            torch.load(f"{directory}/{filename}_checkpoint_encoder.pth")
        )
        
        # Load replay buffer
        self.replay_buffer.load(directory, filename)

    def get_agent_state(self):
        """Lấy toàn bộ internal state để lưu vào train_state.json khi resume"""
        return {
            "training_steps": self.training_steps,
            "exploration_noise": self.exploration_noise,
            # Value clipping state (Q-value bounds)
            "max": self.max,
            "min": self.min,
            "max_target": self.max_target,
            "min_target": self.min_target,
            # Checkpoint tracking state
            "best_min_return": self.best_min_return,
            "min_return": self.min_return,
            "eps_since_update": self.eps_since_update,
            "timesteps_since_update": self.timesteps_since_update,
            "max_eps_before_update": self.max_eps_before_update,
        }

    def restore_agent_state(self, state_dict):
        """Khôi phục internal state từ train_state.json khi resume"""
        self.training_steps = state_dict.get("training_steps", 0)
        self.exploration_noise = state_dict.get(
            "exploration_noise", self.exploration_noise
        )
        self.max = state_dict.get("max", -1e8)
        self.min = state_dict.get("min", 1e8)
        self.max_target = state_dict.get("max_target", 0)
        self.min_target = state_dict.get("min_target", 0)
        self.best_min_return = state_dict.get("best_min_return", -1e8)
        self.min_return = state_dict.get("min_return", 1e8)
        self.eps_since_update = state_dict.get("eps_since_update", 0)
        self.timesteps_since_update = state_dict.get("timesteps_since_update", 0)
        self.max_eps_before_update = state_dict.get("max_eps_before_update", 1)
