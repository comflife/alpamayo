# Self-Reflective Denoising RL (SRD-RL) for Alpamayo-R1

## 🎯 핵심 아이디어

**"GT(정답 데이터)가 틀렸을 때, 모델이 스스로 깨닫고 고치는 것"**

기존 SFT는 "진흙인데 직진"이라고 되어 있으면 무조건 그대로 배웁니다. 하지만 SRD-RL은:

1. **시각적 증거** (텍스처 분석)
2. **논리적 추론** (reasoning 텍스트)
3. **GT와의 비교**

이 세 가지를 종합하여 **"GT가 틀렸다!"**고 판단하면 과감히 무시합니다.

---

## 🧠 핵심 기술

### 1. Trust Gate (신뢰도 게이트)

```python
IF (reasoning에 "진흙/위험/장애물" 언급) AND (visual_safety < 0.3):
    gt_weight = 0.1  # GT를 거의 무시!
ELSE:
    gt_weight = 1.0  # GT를 신뢰
```

### 2. Visual Safety Scoring (Depth 없이!)

- **Laplacian Variance**: 거친 지형(진흙, 돌)은 텍스처가 복잡 → 높은 분산
- **Color Consistency**: 안전한 도로와 색상 차이 → 위험 감지

### 3. Reasoning-Action Alignment

- Reasoning: "좌측 피해" → Trajectory: 우측으로 회피
- 언어와 행동이 일치하지 않으면 감점

### 4. AWR-style Multi-Sampling (개선!)

**핵심 변경사항**: 단순 노이즈가 아닌 **실제 모델 샘플링**

```python
# ❌ 기존 (문제): GT + random noise → 모델 출력 아님!
noisy_traj = gt_trajectory + torch.randn_like(gt_trajectory)

# ✅ 개선: 실제 diffusion에서 다양한 trajectory 샘플링
for i in range(N_samples):
    sampled_action = diffusion.sample(
        expert,
        init_noise=torch.randn(...) * (1.0 + i * 0.2)  # 다양한 시드
    )
    sampled_traj = action_to_traj(sampled_action)
```

- 모델이 직접 여러 trajectory 생성 (stochastic)
- 각 샘플의 보상 계산
- **AWR (Advantage Weighted Regression)**: 최고 보상 trajectory로 학습
  - REINFORCE보다 안정적!
  - Gradient가 올바르게 흐름!

---

## 🚀 사용법

### 기본 학습

```bash
cd /home/byounggun/alpamayo/src

torchrun --nproc_per_node=2 -m alpamayo_r1.alignment.finetune_consistency \
    --data_path /home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl \
    --output_dir /home/byounggun/alpamayo/outputs/alpamayo_srd_rl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 5e-6 \
    --consistency_loss_weight 0.2 \
    --safety_reward_weight 1.5 \
    --gt_reward_weight 0.5 \
    --reasoning_reward_weight 0.3 \
    --num_trajectory_samples 4 \
    --rl_loss_weight 0.5
```

### 초공격적 모드 (데이터 품질이 매우 낮을 때)

GT를 거의 무시하고 시각적 안전성을 최우선으로:

```bash
torchrun --nproc_per_node=2 -m alpamayo_r1.alignment.finetune_consistency \
    --data_path /path/to/noisy_data.jsonl \
    --output_dir /path/to/output \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 5e-6 \
    --safety_reward_weight 2.0 \
    --gt_reward_weight 0.3 \
    --gt_trust_min 0.05 \
    --rl_loss_weight 0.7
```

### 보수적 모드 (데이터가 깨끗할 때)

GT를 더 신뢰하고 RL은 보조적으로:

```bash
torchrun --nproc_per_node=2 -m alpamayo_r1.alignment.finetune_consistency \
    --data_path /path/to/clean_data.jsonl \
    --output_dir /path/to/output \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 5e-6 \
    --safety_reward_weight 0.5 \
    --gt_reward_weight 1.0 \
    --rl_loss_weight 0.2
```

---

## 📊 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `safety_reward_weight` | 1.5 | 시각적 안전성 보상 가중치 (↑ = GT보다 안전성 우선) |
| `gt_reward_weight` | 0.5 | GT 유사도 보상 가중치 (↓ = GT 덜 신뢰) |
| `reasoning_reward_weight` | 0.3 | 언어-행동 일관성 보상 가중치 |
| `num_trajectory_samples` | 4 | 샘플링할 trajectory 개수 (↑ = 더 나은 탐색) |
| `rl_loss_weight` | 0.5 | RL loss vs SFT loss 비율 |
| `danger_keyword_threshold` | 0.3 | 위험 감지 임계값 (↓ = 더 민감) |
| `gt_trust_min` | 0.1 | 최소 GT 신뢰도 |
| `gt_trust_max` | 1.0 | 최대 GT 신뢰도 |

---

## 📈 기대 효과

이 방법으로 학습하면:

1. ✅ **데이터 오류 감지**: "GT가 진흙으로 가라고 하는데, 이건 틀렸어!"
2. ✅ **Super-human 성능**: 인간 운전자보다 안전한 경로 생성
3. ✅ **설명 가능성**: "GT는 직진이지만 진흙이 보여서 우회했습니다"
4. ✅ **노이즈 강건성**: 품질 낮은 데이터셋에서도 안정적 학습

---

## 🔍 학습 중 로그 예시

```
Epoch 1: 100%|██████████| 250/250 [12:34<00:00,  3.02s/it]
loss: 2.345, language_loss: 1.234, traj_loss: 0.567, rl_loss: 0.234,
consistency_loss: 0.089, reward: 2.456, reward_best: 2.789, gt_is_best: 0.65,
safety: 0.678, gt_sim: 0.823, reasoning: 0.712, weight: 1.23
```

주목할 메트릭:
- `reward`: 평균 보상 (↑ = 더 나은 trajectory)
- `reward_best`: 최고 샘플 보상 (모델이 찾은 최선)
- **`gt_is_best`**: GT가 최선인 비율 (↓ = 모델이 GT보다 나은 경로 발견!)
  - **0.65 = 65%만 GT가 최선** → 35%는 모델이 더 나음!
  - **목표: 학습이 진행되며 이 값이 낮아지면 성공!**
- `safety`: 시각적 안전성 (1에 가까울수록 좋음)
- `gt_sim`: GT와 유사도 (낮아도 safety가 높으면 OK!)
- `reasoning`: 언어-행동 일관성
- `weight`: AWR 가중치 평균 (advantage 크기 반영)

---

## 🧪 실험 제안

### 실험 1: 노이즈 레벨에 따른 비교

데이터셋을 의도적으로 오염시켜 테스트:

1. Clean (0% 노이즈)
2. Light (10% 잘못된 GT)
3. Heavy (30% 잘못된 GT)

각각에 대해:
- Baseline SFT
- SRD-RL (이 코드)

성능 비교 지표: 안전성, 충돌률, GT 편차

### 실험 2: Ablation Study

각 컴포넌트의 기여도 측정:

1. Full SRD-RL
2. w/o Trust Gate
3. w/o Visual Safety
4. w/o Reasoning Alignment

---

## 🎓 이론적 배경

이 접근법은 다음 연구들의 아이디어를 결합:

1. **Learning from Noisy Labels** (신뢰도 가중 학습)
2. **RLHF/GRPO** (다중 샘플 비교)
3. **Multimodal Alignment** (Vision-Language-Action 일치)
4. **Self-Supervised Denoising** (스스로 노이즈 감지)

핵심 차별점: **모델이 GT의 옳고 그름을 판단**한다는 점!

---

## 💡 디버깅 팁

### 만약 RL loss가 불안정하다면:

1. `rl_loss_weight` 줄이기 (0.5 → 0.3)
2. `num_trajectory_samples` 늘리기 (4 → 8)
3. Learning rate 줄이기 (5e-6 → 2e-6)

### 만약 모델이 GT를 너무 무시한다면:

1. `gt_reward_weight` 높이기 (0.5 → 0.8)
2. `danger_keyword_threshold` 낮추기 (0.3 → 0.2)
3. `safety_reward_weight` 줄이기 (1.5 → 1.0)

### 만약 모델이 GT를 맹신한다면:

1. `safety_reward_weight` 높이기 (1.5 → 2.5)
2. `gt_trust_min` 낮추기 (0.1 → 0.05)
3. `rl_loss_weight` 높이기 (0.5 → 0.8)

---

## 🚧 향후 개선 방향

1. **실제 Depth 모델 통합**: RGB만으로도 작동하지만, Depth 추가하면 더 정확
2. **Online Learning**: 실제 주행 중 수집한 안전/위험 신호로 계속 학습
3. **Curriculum Learning**: 쉬운 케이스 → 어려운 케이스 순서로 학습
4. **Human-in-the-Loop**: 모델이 GT를 의심할 때 인간에게 확인 요청

---

## 📝 인용

이 코드를 연구에 사용하시면 다음과 같이 인용해주세요:

```bibtex
@misc{alpamayo_srd_rl_2025,
  title={Self-Reflective Denoising RL for Autonomous Driving},
  author={Your Name},
  year={2025},
  howpublished={https://github.com/yourusername/alpamayo}
}
```

---

## 🤝 기여

개선 아이디어나 버그 리포트는 언제나 환영합니다!

---

**Happy Training! 🚗💨**
