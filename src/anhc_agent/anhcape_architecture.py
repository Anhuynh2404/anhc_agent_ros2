import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#0d1117'

def draw_box(ax, x, y, w, h, label, sublabel='', color='#1f6feb', alpha=0.9, fontsize=11, subfontsize=9):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05", linewidth=2,
                         edgecolor='white', facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.12 if sublabel else 0), label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=4)
    if sublabel:
        ax.text(x, y - 0.18, sublabel, ha='center', va='center',
                fontsize=subfontsize, color='#c9d1d9', zorder=4, style='italic')

def arrow(ax, x1, y1, x2, y2, color='#58a6ff', lw=1.5, label='', style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.05, my, label, fontsize=8, color=color, zorder=5)

# ============================================================
# FIGURE 1: Overall TD7 Architecture
# ============================================================
fig1, ax = plt.subplots(figsize=(20, 12))
ax.set_xlim(0, 20); ax.set_ylim(0, 12)
ax.axis('off')
ax.set_facecolor('#0d1117')
fig1.patch.set_facecolor('#0d1117')

ax.text(10, 11.4, 'ANHCAPE Agent — Tổng Quan Kiến Trúc', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#f0f6fc')
ax.text(10, 10.9, 'ANHCAPE: TD3 + LAP Buffer + Encoder (State Embedding)', ha='center',
        fontsize=13, color='#8b949e')

# Environment box
draw_box(ax, 2.5, 6, 3.5, 1.2, 'Gazebo Environment', 'State s, Next_State s\'', '#238636')
# LAP Buffer
draw_box(ax, 7, 6, 3, 1.2, 'LAP Replay Buffer', 'Prioritized Sampling', '#9e6a03')
# Encoder
draw_box(ax, 12, 8.5, 3, 1.2, 'Encoder', 'zs(s) → z_s\nzsa(z_s,a) → z_sa', '#1f6feb')
# Fixed Encoder
draw_box(ax, 12, 6, 3, 1.2, 'Fixed Encoder\n(frozen copy)', 'sync mỗi N steps', '#388bfd', alpha=0.7)
# Fixed Encoder Target
draw_box(ax, 12, 3.5, 3, 1.2, 'Fixed Encoder Target', 'dùng cho Critic target', '#388bfd', alpha=0.5)

# Actor
draw_box(ax, 7, 9.2, 3, 1.2, 'Actor π(s,z_s)', 'Linear: s→hdim\ncat(z_s)→action', '#8957e5')
# Actor Target
draw_box(ax, 7, 3.5, 3, 1.2, 'Actor Target π\'', 'sync mỗi N steps', '#8957e5', alpha=0.6)

# Critic
draw_box(ax, 17, 6, 3, 1.8, 'Critic (Q1, Q2)', 'input: s, a, z_sa, z_s\ncat → 2×zs_dim+hdim\noutput: Q-value', '#da3633')
# Critic Target
draw_box(ax, 17, 3.5, 3, 1.2, 'Critic Target', 'min(Q1,Q2) target', '#da3633', alpha=0.6)

# Checkpoint actor / encoder
draw_box(ax, 2.5, 3.5, 3, 1.2, 'Checkpoint\nActor + Encoder', 'dùng khi eval\nuse_checkpoint=True', '#3fb950', alpha=0.7)

# Arrows: env → buffer
arrow(ax, 4.25, 6, 5.5, 6, '#58a6ff', label='(s,a,s\',r,done)')
# buffer → encoder
arrow(ax, 8.5, 6.5, 10.5, 8.0, '#ffa657', label='batch')
arrow(ax, 8.5, 6.0, 10.5, 6.0, '#ffa657')
arrow(ax, 8.5, 5.5, 10.5, 3.8, '#ffa657')
# encoder → actor
arrow(ax, 10.5, 8.5, 8.5, 9.2, '#d2a8ff', label='z_s')
# encoder fixed → critic
arrow(ax, 13.5, 6, 15.5, 6.3, '#ff7b72', label='z_s, z_sa')
arrow(ax, 13.5, 3.5, 15.5, 5.7, '#ff7b72', label='z_s_t, z_sa_t')
# actor → critic
arrow(ax, 8.5, 9.0, 15.5, 6.8, '#d2a8ff', label='action a', lw=1.2)
# actor target → critic target
arrow(ax, 8.5, 3.5, 15.5, 3.5, '#d2a8ff', label='next_action', lw=1.2)
# critic target → update
arrow(ax, 17, 4.8, 17, 5.1, '#ff7b72', label='Q_target', style='->')
# checkpoint
arrow(ax, 5.5, 9.0, 4.0, 4.1, '#3fb950', lw=1, label='copy weights')

# Legend
legend_items = [
    mpatches.Patch(color='#238636', label='Environment (Gazebo/ROS2)'),
    mpatches.Patch(color='#9e6a03', label='LAP Replay Buffer'),
    mpatches.Patch(color='#1f6feb', label='Encoder (trainable)'),
    mpatches.Patch(color='#388bfd', label='Fixed/Target Encoders'),
    mpatches.Patch(color='#8957e5', label='Actor / Actor Target'),
    mpatches.Patch(color='#da3633', label='Critic / Critic Target'),
    mpatches.Patch(color='#3fb950', label='Checkpoint Models'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=9,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='white',
          framealpha=0.9, ncol=2)

plt.tight_layout()
plt.savefig('/tmp/td7_overview.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved td7_overview.png")

# ============================================================
# FIGURE 2: Neural Network Layers Detail
# ============================================================
fig2, axes = plt.subplots(1, 3, figsize=(20, 10))
fig2.patch.set_facecolor('#0d1117')
fig2.suptitle('TD7 — Chi Tiết Các Mạng Nơ-ron', fontsize=18, fontweight='bold', color='#f0f6fc', y=0.98)

colors_enc = ['#1a3a5c','#1f6feb','#1a3a5c','#0d2e5c','#1f6feb','#1a3a5c']
colors_act = ['#3b1f6e','#8957e5','#8957e5','#8957e5']
colors_cri = ['#5c1a1a','#da3633','#da3633','#da3633']

def draw_nn(ax, layers, colors, title, subtitle):
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(title, color='#f0f6fc', fontsize=14, fontweight='bold', pad=10)
    ax.text(5, 9.4, subtitle, ha='center', color='#8b949e', fontsize=9)
    n = len(layers)
    ys = np.linspace(8.5, 1.0, n)
    for i, (lname, lsize, lact) in enumerate(layers):
        c = colors[i % len(colors)]
        box = FancyBboxPatch((1, ys[i]-0.35), 8, 0.7,
                             boxstyle="round,pad=0.04", linewidth=1.5,
                             edgecolor='#58a6ff', facecolor=c, alpha=0.9, zorder=3)
        ax.add_patch(box)
        ax.text(5, ys[i]+0.05, lname, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)
        ax.text(5, ys[i]-0.15, f'dim={lsize}  |  {lact}', ha='center', va='center',
                fontsize=8, color='#c9d1d9', zorder=4)
        if i < n-1:
            ax.annotate('', xy=(5, ys[i+1]+0.35), xytext=(5, ys[i]-0.35),
                       arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=1.2), zorder=2)

# Encoder
enc_layers = [
    ('Input: state s', 'state_dim', 'raw'),
    ('zs1: Linear(state_dim → hdim)', '256', 'ELU'),
    ('zs2: Linear(hdim → hdim)', '256', 'ELU'),
    ('zs3: Linear(hdim → zs_dim)', '256', 'AvgL1Norm ✓'),
    ('[cat(zs, action)] → zsa1', '512', 'ELU'),
    ('zsa2: Linear(hdim → hdim)', '256', 'ELU'),
    ('zsa3: Linear(hdim → zs_dim)', '256', 'raw (pred next zs)'),
]
draw_nn(axes[0], enc_layers, colors_enc, '🔵 Encoder', 'Học embedding z_s và z_sa từ state/action')

# Actor
act_layers = [
    ('Input: state s', 'state_dim', 'raw'),
    ('l0: Linear(state_dim → hdim)', '256', 'AvgL1Norm ✓'),
    ('[cat(l0_out, z_s)] concat', '512', 'concat'),
    ('l1: Linear(zs_dim+hdim → hdim)', '256', 'ReLU'),
    ('l2: Linear(hdim → hdim)', '256', 'ReLU'),
    ('l3: Linear(hdim → action_dim)', '2', 'tanh → [-1,1]'),
]
draw_nn(axes[1], act_layers, colors_act, '🟣 Actor', 'Chính sách π(s, z_s) → action (vel, angular)')

# Critic
cri_layers = [
    ('[cat(state, action)] s+a', 'state+action', 'concat'),
    ('q01/q02: Linear(s+a → hdim)', '256', 'AvgL1Norm ✓'),
    ('[cat(q0_out, z_sa, z_s)]', '256+512', 'concat embeddings'),
    ('q1/q4: Linear(2*zs+hdim → hdim)', '256', 'ELU'),
    ('q2/q5: Linear(hdim → hdim)', '256', 'ELU'),
    ('q3/q6: Linear(hdim → 1)', '1', 'Q-value'),
]
draw_nn(axes[2], cri_layers, colors_cri, '🔴 Critic (Q1 & Q2)', 'Ước lượng Q(s, a, z_sa, z_s)')

plt.tight_layout()
plt.savefig('/tmp/td7_networks.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved td7_networks.png")

# ============================================================
# FIGURE 3: Training Loop
# ============================================================
fig3, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#0d1117')
fig3.patch.set_facecolor('#0d1117')
ax.text(8, 9.6, 'TD7 — Training Loop & Update Flow', ha='center', fontsize=18,
        fontweight='bold', color='#f0f6fc')

steps = [
    (8, 8.7, 'STEP 1: Collect Experience', 'agent.select_action(state) → action', '#238636'),
    (8, 7.3, 'STEP 2: Store Transition', 'buffer.add(s, a, s\', r, done)', '#9e6a03'),
    (8, 5.9, 'STEP 3: Update Encoder', 'Loss = MSE(pred_zs, next_zs)\nEncoder học dự đoán next state embedding', '#1f6feb'),
    (8, 4.5, 'STEP 4: Update Critic', 'Q_target = r + γ·min(Q1_t, Q2_t)\nCritic_loss = LAP_Huber(|Q - Q_target|)', '#da3633'),
    (8, 3.1, 'STEP 5: Update LAP Priority', 'priority = TD_error^α  (LAP mechanism)', '#9e6a03'),
    (8, 1.9, 'STEP 6: Update Actor (every policy_freq)', 'Actor_loss = -mean(Q(s, π(s,z_s), ...))', '#8957e5'),
    (8, 0.7, 'STEP 7: Sync Targets (every target_update_rate)', 'actor_target ← actor\ncritic_target ← critic\nfixed_encoder ← encoder', '#3fb950'),
]

for x, y, title, desc, color in steps:
    box = FancyBboxPatch((1.5, y-0.45), 13, 0.9,
                         boxstyle="round,pad=0.04", linewidth=1.5,
                         edgecolor=color, facecolor=color+'22', alpha=1.0, zorder=3)
    ax.add_patch(box)
    ax.text(3.5, y+0.1, title, ha='left', va='center',
            fontsize=11, fontweight='bold', color=color, zorder=4)
    ax.text(3.5, y-0.18, desc, ha='left', va='center',
            fontsize=9, color='#c9d1d9', zorder=4)

for i in range(len(steps)-1):
    y1 = steps[i][1] - 0.45
    y2 = steps[i+1][1] + 0.45
    ax.annotate('', xy=(8, y2), xytext=(8, y1),
               arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=1.5), zorder=2)

plt.tight_layout()
plt.savefig('/tmp/td7_training_loop.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved td7_training_loop.png")

print("All 3 figures generated successfully!")
