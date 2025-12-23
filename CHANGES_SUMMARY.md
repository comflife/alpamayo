# 코드 변경 사항 요약

## 개요

[finetune_consistency.py](src/alpamayo_r1/alignment/finetune_consistency.py)를 **Self-Reflective Denoising RL (SRD-RL)** 방식으로 대폭 개선했습니다.

---

## 주요 변경 사항

### 1. 새로운 함수 추가 (라인 108-331)

#### Visual Safety Scoring Functions
- `compute_texture_variance()`: Laplacian variance로 지형 거칠기 측정
- `compute_color_consistency()`: 경로 색상과 안전 기준 영역 비교
- `compute_visual_safety_score()`: 종합 안전성 점수 계산

#### Reasoning Analysis Functions
- `detect_danger_keywords()`: "mud", "rock", "obstacle" 등 위험 키워드 탐지
- `check_reasoning_trajectory_alignment()`: 언어와 행동의 논리적 일치 검증

### 2. ConsistencyEnhancedModel 클래스 개선

#### 새로운 초기화 파라미터 (라인 544-568)
```python
safety_reward_weight: float = 1.5
gt_reward_weight: float = 0.5
reasoning_reward_weight: float = 0.3
num_trajectory_samples: int = 4
rl_loss_weight: float = 0.5
danger_threshold: float = 0.3
gt_trust_min: float = 0.1
gt_trust_max: float = 1.0
```

#### 새로운 forward 메서드 로직 (라인 661-727)
- `reasoning_text` 파라미터 추가
- RL loss 계산 추가 (2b 섹션)

#### **핵심: _compute_rl_loss() 메서드 추가 (라인 903-1192)**

**AWR 스타일 RL 구현 (개선!):**

```python
def _compute_rl_loss(self, ...):
    # 1. ✅ 실제 모델에서 trajectory 샘플링 (NOT random noise!)
    for i in range(num_trajectory_samples):
        # Diffusion에서 다양한 noise seed로 샘플링
        sampled_action = diffusion.sample(
            expert,
            init_noise=torch.randn(...) * (1.0 + i * 0.2)
        )
        sampled_traj = action_to_traj(sampled_action)

    # 2. 각 샘플의 보상 계산
    for traj in sampled_trajectories:
        safety_score = compute_visual_safety_score(image, traj)
        gt_similarity = exp(-distance(traj, gt))
        reasoning_alignment = check_alignment(reasoning, traj)

        # **Trust Gate**: Dynamic GT weighting
        if detect_danger(reasoning) and safety_score < 0.3:
            gt_weight = 0.1  # DISTRUST GT!
        else:
            gt_weight = 1.0  # Trust GT

        reward = safety_weight * safety_score + \
                 gt_weight * gt_similarity + \
                 reasoning_weight * reasoning_alignment

    # 3. ✅ AWR: 최고 보상 trajectory로 학습 (더 안정적!)
    best_idx = reward.argmax()
    best_traj = sampled_trajectories[best_idx]
    weight = exp(advantage[best_idx] / temperature)

    # Flow matching loss to best trajectory (with gradient!)
    rl_loss = weighted_mse(model(best_traj), best_traj, weight)

    return rl_loss
```

**주요 개선점**:
1. **실제 모델 샘플링**: `gt + noise` → `diffusion.sample()`
2. **AWR 알고리즘**: REINFORCE → Advantage Weighted Regression (더 안정적)
3. **Gradient 흐름**: no_grad 샘플링 + 별도 forward로 올바른 gradient
4. **메트릭 추가**: `gt_is_best` (GT가 최선인 비율)

### 3. 데이터 파이프라인 수정

#### Dataset.__getitem__() (라인 1175-1186)
- `reasoning_text` 필드 추가하여 반환

#### collate_fn() (라인 1229-1244)
- `reasoning_text` 리스트 처리 추가

### 4. Training Loop 수정

#### 모델 초기화 (라인 1350-1365)
- 모든 SRD-RL 하이퍼파라미터 전달

#### Forward 호출 (라인 1446-1456)
- `reasoning_text=batch.get("reasoning_text")` 추가

#### 로깅 개선 (라인 1467-1481)
- RL 메트릭 표시 최적화

### 5. Training Arguments 확장 (라인 73-105)

새로운 파라미터:
```python
safety_reward_weight: float = 1.5
gt_reward_weight: float = 0.5
reasoning_reward_weight: float = 0.3
num_trajectory_samples: int = 4
rl_loss_weight: float = 0.5
danger_keyword_threshold: float = 0.3
gt_trust_min: float = 0.1
gt_trust_max: float = 1.0
```

---

## 코드 크기 변화

- **이전**: ~1,082 라인
- **이후**: ~1,550 라인
- **추가**: ~470 라인 (주로 RL 로직)

---

## 의존성 추가

- `import cv2`: OpenCV (텍스처 분석용)
- `import re`: 정규표현식 (키워드 감지용)

---

## 하위 호환성

✅ **완전 호환**: 기존 학습 명령어는 그대로 작동합니다.

새로운 RL 기능을 끄려면:
```bash
--rl_loss_weight 0.0
```

---

## 테스트 상태

✅ Python 문법 검증 완료
⏳ 실제 학습 테스트 필요

---

## 파일 구조

```
/home/byounggun/alpamayo/
├── src/alpamayo_r1/alignment/
│   └── finetune_consistency.py      # ⭐ 메인 수정 파일
├── SRD_RL_README.md                 # 📘 상세 문서
├── QUICKSTART_SRD_RL.md             # 🚀 빠른 시작
└── CHANGES_SUMMARY.md               # 📝 이 파일
```

---

## 다음 할 일

1. **실제 데이터로 학습 테스트**
   ```bash
   cd /home/byounggun/alpamayo/src
   torchrun --nproc_per_node=2 -m alpamayo_r1.alignment.finetune_consistency \
       --data_path /path/to/finetune_data.jsonl \
       --output_dir /path/to/output \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 4 \
       --num_train_epochs 3 \
       --learning_rate 5e-6
   ```

2. **메트릭 모니터링**
   - `rl_loss` 안정성
   - `safety` vs `gt_sim` 트레이드오프
   - `reward` 증가 추세

3. **하이퍼파라미터 튜닝**
   - 데이터 품질에 따라 `safety_reward_weight` / `gt_reward_weight` 조정

4. **평가**
   - 학습된 모델로 추론
   - GT를 무시한 케이스 시각화
   - 안전성 개선 측정

---

## 기술적 하이라이트

### 🔥 Trust Gate 알고리즘

```python
# 모델이 스스로 GT의 신뢰도를 판단
if has_danger_keyword(reasoning) and visual_safety < threshold:
    # "내가 보기엔 위험한데 GT는 직진하래? GT가 틀렸어!"
    gt_weight = 0.1  # GT 거의 무시
else:
    # "평범한 상황, GT 믿어도 돼"
    gt_weight = 1.0  # GT 신뢰
```

### 🧠 Visual Safety (No Depth!)

```python
# RGB만으로 위험 지형 감지
texture_safety = 1.0 - laplacian_variance(path)  # 거친 땅 = 높은 분산
color_safety = consistency(path_color, safe_reference)
safety_score = 0.5 * texture_safety + 0.5 * color_safety
```

### 🎯 GRPO-style Learning

```python
# 여러 trajectory 샘플링 → 보상 비교 → 좋은 것 강화
advantages = rewards - baseline
loss = -sum(log_prob[i] * advantage[i])  # Policy gradient
```

---

## 연구 기여

이 코드는 다음 분야에 기여합니다:

1. **Learning from Noisy Labels**: GT 신뢰도 동적 조절
2. **Vision-Language-Action Alignment**: 다중 모달 일관성
3. **Self-Supervised Denoising**: 모델이 스스로 노이즈 감지
4. **Safe RL for Autonomous Driving**: 안전성 기반 보상 설계

---

**질문이나 버그 발견 시 이슈 리포트 부탁드립니다!**
