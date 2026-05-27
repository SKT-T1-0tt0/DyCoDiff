# MCFL Gate 统一标定策略（基于三数据集 BEATs pooled norm）

## 1. 三数据集 BEATs pooled norm 统计（16 段 clip + mean-pool 8 tokens）

| 数据集 | mean | std | p5 | p95 | min | max |
|--------|------|-----|-----|-----|-----|-----|
| post_audioset_drums | 8.57 | 0.16 | 8.30 | 8.82 | 8.26 | 8.86 |
| post_URMP           | 8.08 | 0.69 | 7.44 | 9.79 | 7.33 | 9.96 |
| post_landscape      | 8.58 | 0.23 | 8.19 | 8.90 | 8.18 | 9.07 |

- **结论**：drums/landscape 很集中，URMP 分布最宽（std 0.69）、p5–p95 跨 [7.44, 9.79]。

---

## 2. 统一标定策略（同时覆盖三数据集、无需手调）

### 策略 A：绝对区间（三角形 conf）

- **推荐默认**（覆盖三数据集 p5–p95 并留余量）：
  - `mcfl_gate_norm_low = 7.2`
  - `mcfl_gate_norm_high = 10.0`
- 中心 ≈ 8.6，norm 在 [7.2, 10.0] 内 conf 从 0→1→0，超出则 conf→0。
- **仅 drums 紧标定**（若只关心 drums、且希望 conf 更“可控”）：low=8.3, high=8.9。

### 策略 B：Z-score 标定（更稳，跨数据集只更新统计）

- 训练阶段用 `scripts/audio_beats_stats.py` 在**训练集**上跑一遍，得到 pooled norm 的 **μ_train, σ_train**（或合并多数据集的加权均值/方差）。
- 推理时：
  - `z = (norm - μ_train) / (σ_train + eps)`
  - `conf = clamp((z - z_low)/(z_high - z_low), 0, 1)`，例如 `z_low=-1.5, z_high=1.5`。
- 基于当前三数据集近似：
  - `mcfl_gate_norm_mu = 8.4`
  - `mcfl_gate_norm_sigma = 0.5`
  - `mcfl_gate_z_low = -1.5`, `mcfl_gate_z_high = 1.5`
- 换数据集时只需重新跑 stats 更新 μ/σ，不必改 low/high。

---

## 3. 默认参数建议（“不炸、泛化更稳”）

| 参数 | 推荐默认 | 说明 |
|------|----------|------|
| mcfl_gate_norm_low | **7.2** | 统一覆盖三数据集 p5 以下 |
| mcfl_gate_norm_high | **10.0** | 统一覆盖三数据集 p95 以上 |
| mcfl_gate_ema | **0.9** | gate 时间平滑更强，减轻 flicker |
| mcfl_gate_use_zscore | False（可选 True） | 用策略 B 时设为 True 并设 mu/sigma |

若使用 **z-score**（`mcfl_gate_use_zscore=True`）：

| 参数 | 推荐默认 |
|------|----------|
| mcfl_gate_norm_mu | 8.4 |
| mcfl_gate_norm_sigma | 0.5 |
| mcfl_gate_z_low | -1.5 |
| mcfl_gate_z_high | 1.5 |

---

## 4. 单次压缩 + 验证矩阵（快速定位问题）

### 原则：只压一次，避免“数据侧 soft_clip + BEATs 前 response”双重非线性。

- **方案 A（推荐）**：`audio_norm_mode=peak`, `audio_soft_clip=none`, `audio_response=compand`
- **方案 B**：`audio_norm_mode=peak`, `audio_soft_clip=compand`, `audio_response=none`

### Random Gain 顺序（训练，当前实现已按此顺序）

1. **数据侧**：`audio_norm_mode=peak`（保证不爆，再乘 g 时意义稳定）
2. **Train loop**：Random Gain（peak 之后）：`x *= g`, `g ~ log_uniform(audio_gain_low, audio_gain_high)`
   - 起步建议 `--audio_gain_low 0.5 --audio_gain_high 2`，不稳再扩到 `[0.25, 4]`
3. **单次压缩**：`audio_response=compand` 或 `tanh`（只做一次，数据侧 `audio_soft_clip=none`）
4. BEATs

### 收尾验证矩阵（最少 4 组，三数据集各跑一遍小评估）

按下面顺序跑，**每组都在三数据集上跑一遍小评估**，把风险压到最低：

| 组 | 配置 | 目标 |
|----|------|------|
| **1** | **稳健默认**：peak + response=compand（单次）、gain [0.5, 2]、gate 7.2–10.0 + EMA 0.9、dropout 0.2 | drums 不炸、flicker 不爆、URMP/landscape 无明显退化 |
| **2** | 只改压缩类型：其他同组 1，**response=tanh**（随机 k 依旧开） | 看 compand vs tanh 谁对 BEATs 更“分布友好” |
| **3** | 关掉随机化：**audio_random_gain False**、**audio_random_response_strength False**，其他同组 1 | 确认随机化是在帮泛化，而不是引入训练不稳定；若关掉后 eval 更稳但跨域差 → 可把 gain 收窄为 [0.7, 1.5] |
| **4** | Gate 标定方式：训练仍用 absolute；**推理**对三数据集分别试 **absolute / z-score** | 确认 z-score 不会引入新偏置（尤其 URMP std 大时，absolute 可能更“偏开”，z-score 更“偏保守”） |

---

## 5. 防炸护栏（成本低、收益高）

| 护栏 | 实现 | 说明 |
|------|------|------|
| **护栏 1** | `mcfl_gate_lambda_max=0.2`（默认） | gate 硬上限，即使 conf=1 也不超过此值；可选退避：若检测到生成异常（如 early steps latent/residual 极端大）可临时 gate×0.5 |
| **护栏 2** | eval 时把 **pooled norm 的 p1/p99、min/max** 与 **conf 的 p1/p99** 写入 JSON | 跑 `audio_beats_stats.py --with_beats --out xxx.json` 会输出 `pooled_norm` 与 `conf` 分布；若出现“pooled norm 整体偏移”或“conf 全 0/全 1”可立刻定位 |
| **护栏 3** | `mcfl_gate_norm_clip_clamp=True`（默认） | 对 per-frame norm 按**当前 clip 的 p5–p95** 限幅后再算 conf，避免单帧 spike 把 conf 拉到 0 或 1 |

---

## 6. drums 专用紧标定建议

- 窄区间 [8.3, 8.9] 会让 conf 对小偏移非常敏感，**必须配 EMA=0.9 或更强**，否则易把瞬态差异放大成 gate 抖动。
- 数据处理链（resample、peak、gain、response）稍有变化，pooled norm 可能整体平移，**窄区间更脆弱**。
- **更推荐**：drums-only 时用 **z-score**（μ/σ 来自 drums stats），比固定 low/high 更抗分布漂移。

---

## 7. 当出现 FID 爆炸 / flicker 爆炸时怎么排查（3 步）

1. **看 pooled norm 分布是否漂移**  
   用 `audio_beats_stats.py --with_beats` 在出问题的数据集上跑一遍，看 `pooled_norm` 的 p5/p95、p1/p99 是否与训练集或之前统计差很多；若整体偏移 → 调 gate 区间或改用 z-score 并更新 μ/σ。

2. **看 conf 是否全 0/全 1 或长尾**  
   同一份 stats JSON 里有 `conf` 的 p1/p99；若 conf 全 0（gate 几乎关闭）或全 1（gate 一直满）、或长尾极端 → 说明标定区间或 z 范围不合适，或存在异常样本。

3. **看是否发生双重压缩**  
   确认数据侧 `audio_soft_clip=none`、只在 BEATs 前做一次 `audio_response=compand`（或 tanh）；若两边都开了非线性 → 先改成只压一次再测。

---

## 8. 推荐默认的选择理由

- **为什么默认 compand（方案 A）**：对瞬态更线性、可解释，鼓点等强瞬态不易被压扁；tanh 更激进，有时更稳，可通过组 2 对比。
- **为什么 EMA=0.9**：对 flicker 是“强力保险”，比 0.8 更稳；唯一风险是条件响应略慢（动作/节奏跟随可能滞后）。若 AV_ALIGN 下降但 flicker 已很稳，可把 EMA 降到 0.85 做权衡。
- **为什么 gain 先 [0.5, 2] 再扩**：起步保守，避免大增益引入训练不稳定；若跨域仍差再扩到 [0.25, 4]。

---

## 9. 如何生成/更新统计

```bash
# 各数据集分别跑（含 BEATs）
python scripts/audio_beats_stats.py --data_path datasets/post_audioset_drums --with_beats --out stats_drums.json
python scripts/audio_beats_stats.py --data_path datasets/post_URMP --with_beats --out stats_urmp.json
python scripts/audio_beats_stats.py --data_path datasets/post_landscape --with_beats --out stats_landscape.json

# 合并得到统一 low/high 与 z-score 默认（见 scripts/unify_gate_calibration.py）
python scripts/unify_gate_calibration.py stats_drums.json stats_urmp.json stats_landscape.json

# 护栏2：输出含 pooled_norm 与 conf 分布（可合并进 eval 报告）
python scripts/audio_beats_stats.py --data_path datasets/post_audioset_drums --with_beats --out eval_condition_stats.json
```
