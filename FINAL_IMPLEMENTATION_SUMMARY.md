# 🎯 SRD-RL 최종 구현 요약

## 핵심 질문과 해결책

### ❓ "동일 이미지에 대해 다른 trajectory를 어떻게 생성하나?"

**문제**: 초기 구현에서는 `gt_trajectory + random_noise`로 샘플링했는데, 이건:
- 모델의 실제 출력이 아님
- Gradient가 제대로 흐르지 않음
- 다양성 부족

**✅ 해결책**: **실제 Diffusion 모델에서 Stochastic Sampling**

```python
# ❌ 기존 (잘못된 방식)
for _ in range(N):
    noisy_traj = gt_trajectory + torch.randn_like(gt_trajectory) * 0.5

# ✅ 개선 (올바른 방식)
for i in range(N):
    # Diffusion의 다양한 noise seed로 실제 샘플링
    sampled_action = self.diffusion.sample(
        self.expert,
        batch_size=B,
        action_dim=action_dims,
        device=device,
        dtype=dtype,
        init_noise=torch.randn(...) * (1.0 + i * 0.2)  # 다양한 초기값
    )

    # Action을 trajectory로 변환
    sampled_traj = self.action_space.action_to_traj(
        sampled_action,
        traj_history_xyz=ego_history_xyz,
        traj_history_rot=ego_history_rot,
    )['traj_future_xyz']
```

**왜 이게 작동하는가?**

1. **Diffusion의 Stochasticity**: Flow matching/diffusion은 `x_0 ~ N(0, 1)`에서 시작
2. **다른 초기 노이즈** → **다른 최종 trajectory**
3. **모델의 실제 분포**에서 샘플링됨

---

## 알고리즘 흐름도

```
Input: Image, GT trajectory, Ego history
│
├─> 1. Diffusion에서 N개 trajectory 샘플링
│    ├─ Sample 1: GT (baseline)
│    ├─ Sample 2: diffusion.sample(noise_seed=1)
│    ├─ Sample 3: diffusion.sample(noise_seed=2)
│    └─ Sample N: diffusion.sample(noise_seed=N)
│
├─> 2. 각 샘플의 보상 계산
│    │
│    ├─ Visual Safety Score
│    │   ├─ Laplacian variance (texture)
│    │   └─ Color consistency
│    │
│    ├─ GT Similarity Score
│    │   └─ exp(-distance(sampled, GT))
│    │
│    ├─ Reasoning Alignment Score
│    │   └─ Language-action consistency
│    │
│    └─ **Trust Gate** (핵심!)
│        │
│        ├─ IF (reasoning has "danger") AND (safety < 0.3):
│        │   └─ gt_weight = 0.1  ← GT 무시!
│        │
│        └─ ELSE:
│            └─ gt_weight = 1.0  ← GT 신뢰
│
│    Final Reward = safety_weight * safety +
│                    gt_weight * gt_similarity +
│                    reasoning_weight * alignment
│
├─> 3. 최고 보상 trajectory 선택
│    └─ best_idx = argmax(rewards)
│
└─> 4. AWR Loss 계산 (Gradient 흐름!)
     │
     ├─ Advantage weight: w = exp(advantage / temp)
     │
     └─ Flow matching loss:
         Forward model(best_trajectory) → Weighted MSE
```

---

## 핵심 개선 사항

### 1. 실제 모델 샘플링

**Before**:
```python
sampled = gt + noise  # 모델 아님!
```

**After**:
```python
sampled = diffusion.sample(expert, init_noise=...)  # 실제 모델!
```

### 2. AWR vs REINFORCE

**REINFORCE (불안정)**:
```python
loss = -log_prob * advantage  # High variance!
```

**AWR (안정적)**:
```python
best_traj = trajectories[argmax(reward)]
weight = exp(advantage / temperature)
loss = weighted_mse(model(best_traj), best_traj, weight)  # Low variance!
```

### 3. Gradient 흐름

**Before**:
```python
with torch.no_grad():
    samples = [gt + noise for _ in range(N)]
# ❌ No gradient!
```

**After**:
```python
# Sampling (no grad)
with torch.no_grad():
    samples = [diffusion.sample(...) for _ in range(N)]

# Training (with grad!)
target = samples[best_idx]
loss = flow_matching_loss(model(target), target)  # ✅ Gradient flows!
```

---

## 주요 메트릭 해석

학습 중 나타나는 메트릭:

```
rl_reward_mean: 2.456     # 평균 보상
rl_reward_best: 2.789     # 최고 보상
rl_gt_is_best: 0.65       # ⭐ 핵심 메트릭!
rl_safety_mean: 0.678     # 시각적 안전성
rl_gt_sim_mean: 0.823     # GT 유사도
rl_weight_mean: 1.23      # AWR 가중치
```

### 🎯 `rl_gt_is_best`: 성공의 지표

- **1.0 (100%)**: 항상 GT가 최선 → 모델이 개선 못함
- **0.65 (65%)**: 35%는 모델이 GT보다 나은 경로 발견!
- **0.30 (30%)**: 70%는 모델이 GT 능가 → 🎉 목표 달성!

**이상적 학습 곡선**:
```
Epoch 1:  gt_is_best = 0.90  (모델이 아직 약함)
Epoch 3:  gt_is_best = 0.70  (개선 중)
Epoch 5:  gt_is_best = 0.50  (절반은 모델이 나음!)
Epoch 10: gt_is_best = 0.30  (목표 달성!)
```

---

## 실전 예제

### 시나리오: 진흙탕 직진 GT

```python
# GT: "진흙탕으로 직진" (잘못된 데이터!)
gt_trajectory = [[1, 0], [2, 0], [3, 0], ...]  # 직진

# 1. 모델이 4개 샘플 생성
samples = [
    gt_trajectory,           # Sample 0: GT (직진)
    [[1, 0.5], [2, 1.0], ...], # Sample 1: 약간 우회
    [[1, 1.5], [2, 3.0], ...], # Sample 2: 크게 우회
    [[1, -0.3], [2, -0.5], ...], # Sample 3: 약간 좌회전
]

# 2. 보상 계산
rewards = [
    1.2,  # GT: safety=0.2 (진흙), gt_sim=1.0 → gt_weight=0.1 적용!
    2.8,  # Sample 1: safety=0.9 (안전), gt_sim=0.7
    3.1,  # Sample 2: safety=1.0 (매우 안전), gt_sim=0.4 ← BEST!
    2.3,  # Sample 3: safety=0.8, gt_sim=0.8
]

# 3. Trust Gate 작동
# Reasoning: "I see mud ahead"
# Safety: 0.2 < 0.3 (danger!)
# → gt_weight = 0.1 (GT 무시!)

# 4. AWR 학습
best_idx = 2  # Sample 2 선택
advantage = 3.1 - 2.35 = 0.75
weight = exp(0.75 / 2.0) = 1.45

# 모델은 Sample 2 (크게 우회)를 학습
# → 다음부터는 진흙 피함!
```

---

## 코드 위치

핵심 코드: `finetune_consistency.py`

```
Line 105-249:  Visual Safety Scoring
Line 256-323:  Reasoning Analysis
Line 903-1192: _compute_rl_loss (핵심!)
  ├─ 949-988:  Diffusion 샘플링
  ├─ 990-1043: Log prob 계산
  ├─ 1045-1097: Trust-aware reward 계산
  └─ 1107-1175: AWR loss 계산
```

---

## 검증 체크리스트

학습 시작 전 확인:

- [ ] `rl_loss` 값이 합리적? (0.1 ~ 10 범위)
- [ ] `rl_gt_is_best` 초기값이 높음? (0.8 ~ 1.0)
- [ ] `rl_safety_mean` 계산됨? (NaN 아님)
- [ ] 메모리 OOM 안남? (샘플링 4개 → 메모리 4배)

학습 중 모니터링:

- [ ] `rl_gt_is_best`가 **감소** 추세? ← 핵심!
- [ ] `rl_reward_mean`이 **증가** 추세?
- [ ] `rl_safety_mean`이 증가?
- [ ] `rl_loss` 안정적? (폭발 안함)

학습 후 검증:

- [ ] `rl_gt_is_best` < 0.5? (절반 이상 모델이 나음)
- [ ] 추론 시 진흙/장애물 피함?
- [ ] Reasoning 텍스트가 일관적?

---

## 트러블슈팅

### Q1: `rl_loss`가 NaN 됨

**원인**: AWR weight 폭발 (`exp(advantage)`가 너무 큼)

**해결**:
```python
weight = torch.exp(advantage / 2.0).clamp(0.1, 5.0)  # Clamp 추가!
```

### Q2: 메모리 부족 (OOM)

**원인**: 샘플링 4개 → 4배 메모리

**해결**:
```bash
--num_trajectory_samples 2  # 4 → 2로 줄이기
--per_device_train_batch_size 1
--gradient_accumulation_steps 8  # 늘리기
```

### Q3: `rl_gt_is_best`가 안 줄어듦

**원인 1**: GT가 실제로 좋음 (데이터 품질 높음)

**해결**: `--safety_reward_weight` 높이기

**원인 2**: 샘플 다양성 부족

**해결**: `--num_trajectory_samples` 늘리기 (4 → 8)

### Q4: 학습이 너무 느림

**원인**: Diffusion 샘플링이 expensive

**해결**:
- `--num_trajectory_samples 2`로 줄이기
- `--rl_loss_weight 0.3`으로 줄여서 RL 빈도 감소

---

## 다음 단계

1. **학습 실행**
   ```bash
   bash run_srd_rl.sh basic
   ```

2. **메트릭 모니터링**
   - TensorBoard로 `rl_gt_is_best` 추적

3. **평가**
   - GT 틀린 케이스에서 모델 추론
   - 시각화: 모델 trajectory vs GT

4. **논문화**
   - 이 접근법은 novelty 충분!
   - Learning from Noisy Labels + Safe RL 결합

---

**구현 완료! 🎉**

이제 실제 학습을 돌려보고 `rl_gt_is_best`가 감소하는지 확인하세요!
