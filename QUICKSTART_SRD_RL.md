# 🚀 SRD-RL 빠른 시작 가이드

## 30초 요약

**기존 문제**: SFT는 GT가 "진흙으로 직진"이라고 하면 그대로 배움 → 위험!

**SRD-RL 해결책**: 모델이 시각적 증거로 GT가 틀렸다고 판단하면 무시함 → 안전!

---

## 즉시 실행

```bash
cd /home/byounggun/alpamayo/src

# 기본 학습 (권장)
torchrun --nproc_per_node=2 -m alpamayo_r1.alignment.finetune_consistency \
    --data_path /home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl \
    --output_dir /home/byounggun/alpamayo/outputs/alpamayo_srd_rl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --learning_rate 5e-6
```

---

## 핵심 파라미터 (필요시 조정)

### 데이터가 노이즈가 많다면 (GT 신뢰도 낮음)

```bash
--safety_reward_weight 2.0      # 시각적 안전성을 GT보다 우선
--gt_reward_weight 0.3          # GT 영향력 낮춤
--rl_loss_weight 0.7            # RL 비중 높임
```

### 데이터가 깨끗하다면 (GT 신뢰도 높음)

```bash
--safety_reward_weight 0.5      # 시각적 안전성 보조적 역할
--gt_reward_weight 1.0          # GT를 더 신뢰
--rl_loss_weight 0.2            # SFT 위주로
```

---

## 학습 중 확인할 메트릭

```
loss: 2.345          # 전체 손실
rl_loss: 0.234       # RL 손실 (안정적이어야 함)
safety: 0.678        # 시각적 안전성 (↑ good)
gt_sim: 0.823        # GT 유사도 (낮아도 safety 높으면 OK!)
reasoning: 0.712     # 언어-행동 일관성
reward: 2.456        # 총 보상 (↑ good)
```

**핵심**: `safety` 높고 `gt_sim` 낮으면 → 모델이 GT를 의심하고 더 안전한 경로 선택 중!

---

## 문제 해결

| 증상 | 해결책 |
|------|--------|
| RL loss 폭발 | `--rl_loss_weight 0.3` 또는 `--learning_rate 2e-6` |
| 모델이 GT 무시 | `--gt_reward_weight 0.8` 또는 `--safety_reward_weight 1.0` |
| 모델이 GT 맹신 | `--safety_reward_weight 2.0` 또는 `--gt_trust_min 0.05` |
| OOM (메모리 부족) | `--per_device_train_batch_size 1` + `--gradient_accumulation_steps 8` |

---

## 체크포인트 위치

학습 완료 후:

```
/home/byounggun/alpamayo/outputs/alpamayo_srd_rl/
├── checkpoint-1000/
│   ├── vlm_lora/              # VLM LoRA 가중치
│   └── expert_diffusion.pt    # Expert + Diffusion 가중치
├── checkpoint-2000/
└── final/
```

---

## 다음 단계

1. **학습 시작**: 위 명령어 실행
2. **로그 모니터링**: `safety`, `gt_sim`, `reward` 확인
3. **평가**: 학습된 모델로 추론 실행
4. **시각화**: 모델이 GT를 무시한 케이스 분석

---

자세한 내용은 [SRD_RL_README.md](SRD_RL_README.md) 참고!
