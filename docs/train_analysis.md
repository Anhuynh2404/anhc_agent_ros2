# Phân Tích Convergence & Khi Nào Dừng Training AnhcApe

## 📊 Trạng Thái Training Hiện Tại

| Metric | Giá trị |
|--------|---------|
| **Tổng timesteps** | 685,555 / 5,000,000 (**13.7%**) |
| **Training steps** | 660,555 |
| **Eval epochs** | 131 |
| **Exploration noise** | 0.2073 (sẽ giảm đến min 0.1) |
| **Best eval reward** | **109.03** (epoch 121) |

### Tiến trình qua các Phase

| Phase | Epochs | Mean Reward | Std Dev | Nhận xét |
|-------|--------|------------|---------|----------|
| Phase 1 (Exploration) | 1–20 | **-39.97** | 37.50 | Agent gần như random |
| Phase 2 (Learning) | 21–60 | **+8.13** | 37.03 | Bắt đầu học, rất noisy |
| Phase 3 (Improving) | 61–100 | **+22.72** | 42.05 | Reward tăng nhưng vẫn biến động mạnh |
| Phase 4 (Recent) | 101–131 | **+44.89** | 35.07 | ✅ Mean tốt nhất, std giảm |

### Gần đây (dấu hiệu tích cực)

| Window | Mean | Std | Positive Rate |
|--------|------|-----|---------------|
| Last 20 epochs | 61.37 | 27.86 | **100%** ✅ |
| Last 10 epochs | 60.91 | 26.52 | **100%** ✅ |
| Last 5 epochs | 60.61 | 24.43 | **100%** ✅ |

---

## 🔍 Cách Đánh Giá Convergence Ổn Định

### 1. Moving Average (MA) — Chỉ số chính

> [!TIP]
> Mở TensorBoard để xem trực quan: `tensorboard --logdir=temp/logs/`

**MA10** (trung bình 10 epoch gần nhất) và **MA20** là hai chỉ số quan trọng nhất:

```
MA10 gần đây: [60.5, 65.1, 63.5, 64.5, 60.9]  ← khá phẳng ✅
MA20 gần đây: [54.1, 58.5, 58.0, 60.2, 61.4]  ← vẫn đang tăng nhẹ 📈
```

**Convergence ổn định = khi MA20 không còn tăng đáng kể trong 50+ epochs liên tiếp.**

### 2. Standard Deviation — Mức độ ổn định

- Phase 3 std: 42.05 → Phase 4 std: 35.07 → **Giảm dần ✅**
- Nhưng std ~25–35 vẫn cao → Agent chưa hoàn toàn ổn định

### 3. Positive Reward Rate

- Tổng: 63.4% → Last 20: **100%** → Rõ ràng đang cải thiện

### 4. Best Reward Plateau

- Best = 109.03 xuất hiện ở epoch 121 (gần đây)
- Nếu best không cải thiện trong 100+ epochs → có thể đã plateau

---

## 🎯 Tiêu Chí Dừng Training (Stopping Criteria)

### Tình huống 1: "Đủ tốt" — Dừng sớm ⏸️

Dừng khi **TẤT CẢ** điều kiện sau được thỏa mãn liên tục trong **≥ 50 eval epochs** (~250K timesteps):

- [ ] MA20 biến động < 5% so với giá trị trung bình
- [ ] Std deviation < 20 (hiện tại: 24–35)
- [ ] Positive reward rate > 90%
- [ ] Best reward không cải thiện trong 50+ epochs

### Tình huống 2: "Tối ưu" — Train hết budget ⏳

- Chạy hết `max_timesteps = 5,000,000` (hiện mới 13.7%)
- Lấy **best model** từ `pytorch_models_best/` làm final model

### Tình huống 3: "Overfitting / Degradation" — Dừng ngay ⚠️

> [!WARNING]
> Dừng ngay nếu thấy **bất kỳ** dấu hiệu nào sau:

- MA20 giảm liên tục > 30 epochs (performance degradation)
- Critic loss tăng không kiểm soát (divergence)
- Q-value explodes (hiện tại max=206, min=-268 — OK)

---

## 📈 Đánh Giá Hiện Tại: AnhcApe CHƯA Converge

> [!IMPORTANT]
> **Kết luận: Agent đang ở giai đoạn học tốt nhưng CHƯA converge ổn định. Không nên dừng lúc này.**

**Lý do:**

1. **Mới train 13.7%** budget (685K/5M) — còn rất sớm
2. **MA20 vẫn đang tăng** (54 → 61) — chưa plateau
3. **Exploration noise = 0.207** — chưa decay hết (min = 0.1, sẽ hết ở ~750K steps)
4. **Std dev vẫn cao** (~25–35) — performance chưa ổn định
5. **Trend rõ ràng tích cực** — 100% positive reward trong 20 epoch gần nhất

### 🗺️ Dự Kiến Timeline

```
Timesteps      Giai đoạn                     Noise level
────────────────────────────────────────────────────────────
    0 -  25K   Random exploration             1.0 (max)
   25K - 750K  ← BẠN Ở ĐÂY (685K)           0.207 ← đang giảm
  750K - 1.5M  Exploitation phase bắt đầu    0.1 (min)
  1.5M - 3M    Peak learning / convergence    0.1
    3M - 5M    Fine-tuning / plateau          0.1
```

---

## 🛠️ Khuyến Nghị Cụ Thể

### Ngắn hạn (bây giờ → 1.5M timesteps)
1. **Tiếp tục train** — không dừng
2. **Theo dõi TensorBoard** mỗi ~50K timesteps
3. Quan sát khi noise chạm 0.1 (~750K) — performance thường tăng đột ngột

### Trung hạn (1.5M → 3M timesteps)
4. Bắt đầu theo dõi convergence criteria ở trên
5. Nếu MA20 plateau > 50 epochs liên tiếp → có thể cân nhắc dừng

### Dài hạn (3M → 5M timesteps)
6. So sánh với TD3 Baseline tại cùng timestep
7. Nếu không còn cải thiện → sử dụng best model và dừng

---

## 📋 Checklist Theo Dõi (Copy để sử dụng)

```
Mốc kiểm tra:
[ ] 750K steps  — Noise decay hoàn tất, benchmark lại
[ ] 1M steps    — So sánh AnhcApe vs TD3 baseline lần đầu
[ ] 1.5M steps  — Kiểm tra convergence indicators
[ ] 2M steps    — Quyết định có tiếp tục hay dừng
[ ] 3M steps    — Final convergence check
[ ] 5M steps    — Training hoàn tất (nếu chưa dừng sớm)
```

---

## 🔧 Các Metric Bạn Đang Track Trên TensorBoard

| Metric | Ý nghĩa | Dấu hiệu tốt |
|--------|----------|---------------|
| `train/episode_reward` | Reward mỗi episode training | Xu hướng tăng |
| `train/episode_steps` | Số bước mỗi episode | Tăng dần (survive lâu hơn) |
| `eval/average_reward` | Reward eval (10 eps, no noise) | **Chỉ số quan trọng nhất** |
| `eval/best_reward` | Best model checkpoint | Nấc thang đi lên |
| `loss` (critic loss) | LAP Huber loss | Ổn định, không tăng vọt |
| `Q` | Giá trị Q trung bình | Tăng dần, ổn định |
| `Q_max` | Q-value ceiling | Không explode (< 500 là OK) |

> [!NOTE]
> **Tóm lại**: AnhcApe đang train tốt! Xu hướng hoàn toàn tích cực. Chỉ mới dùng ~14% budget. Hãy tiếp tục train ít nhất đến 1.5M–2M timesteps rồi đánh giá lại.
