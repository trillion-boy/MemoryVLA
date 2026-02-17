# MemoryVLA per_attn 조사 보고서

## Cross-Domain Spatial Grounding을 위한 per_attn 분석 및 발견 사항

**작성일:** 2026-02-17
**브랜치:** `claude/test-libero-maniskill-generalization-nKmWd`

---

## 목차

1. [배경 및 동기](#1-배경-및-동기)
2. [1단계: Spatial Prior (L) 주입 지점 탐색 및 초기 이상 발견](#2-1단계-spatial-prior-l-주입-지점-탐색-및-초기-이상-발견)
3. [2단계: per_attn 작동 불능 확인 (LIBERO ckpt + ManiSkill env)](#3-2단계-per_attn-작동-불능-확인-libero-ckpt--maniskill-env)
4. [3단계: Domain Gap 여부 검증 (LIBERO ckpt + LIBERO env)](#4-3단계-domain-gap-여부-검증-libero-ckpt--libero-env)
5. [4단계: 근본 원인 규명 — Zero Initialization Trap](#5-4단계-근본-원인-규명--zero-initialization-trap)
6. [5단계: 우회 전략 수립 및 구현](#6-5단계-우회-전략-수립-및-구현)
7. [핵심 결론 및 시사점](#7-핵심-결론-및-시사점)

---

## 1. 배경 및 동기

### 1.1 목표

LIBERO에서 학습된 MemoryVLA checkpoint를 ManiSkill2 환경에서 사용할 때, **spatial grounding** (특정 물체 위치에 action을 집중시키는 능력)을 부여하려 했다. ControlMLLM 논문의 접근법을 참고하여, per_token (시각적 패치 특징)에 공간 prior L을 주입하면 per_attn (cross-attention)이 해당 영역에 집중할 것으로 기대했다.

### 1.2 아키텍처 개요

```
Vision Backbone (DINO V2 + SigLIP)
        │
        ▼  [B, 256, vision_dim]
   SE Bottleneck (per_compr)         ← (1) Spatial Prior L 주입 후보 지점
        │
        ▼  [B, 256, 256]
   Memory Bank (per_mem_bank)
        │
        ▼  [B, 256, 256]
   PerTokenEmbedder                  ← Linear(256 → hidden_size)
        │
        ▼  [B, 256, hidden_size]
   DiT Blocks × N
   ┌────────────────────────┐
   │ self-attn (cog_attn)   │  ← Xavier init → 정상 학습됨
   │ per_attn (cross-attn)  │  ← Zero init → ★ 문제 지점 ★
   │ MLP                    │
   └────────────────────────┘
        │
        ▼
   Action Output (noise prediction)
```

---

## 2. 1단계: Spatial Prior (L) 주입 지점 탐색 및 초기 이상 발견

### 2.1 L 주입 지점 결정

Spatial prior L (예: SAM 마스크 → 16×16 패치 그리드)을 어디에 넣을지 탐색했다. 후보는 두 곳이었다:

| 후보 | 위치 | 장단점 |
|------|------|--------|
| A. SE Bottleneck 이후 | `per_compr(vision_feats)` 직후 | vision feature에 직접 영향, 다운스트림 전체에 전파 |
| B. DiT 내부 per_attn | DiTBlock 내 cross-attention | 타겟이 명확하지만 구조 변경 필요 |

**결정: A안 (SE Bottleneck 이후)**을 선택했다. 이유:

- per_token이 이미 256개 패치 × 256차원으로 압축되어 있어 패치 단위 조작에 적합
- Memory Bank, DiT block 등 전체 다운스트림에 자연스럽게 전파
- 기존 아키텍처 변경 없이 `_apply_per_token_prior()` 메서드로 깔끔하게 구현 가능

**구현 (`memory_vla.py:545-570`):**

```python
def _apply_per_token_prior(self, per_tokens, prior, strength, mode):
    prior_map = self._normalize_prior_map(prior)  # [B, N, 1]
    if mode == "add":
        token_scale = per_tokens.detach().std().clamp(min=1e-6)
        return per_tokens + strength * prior_map * token_scale
    return per_tokens * (1.0 + strength * prior_map)
```

### 2.2 초기 이상 징후 발견

간단한 eval script를 돌려 L 주입 효과를 확인하려 했다. **그런데 per_token에 prior를 넣든 안 넣든 action output이 거의 동일했다.** 이것이 이상하다고 느낀 출발점이다.

- prior 없이 실행 → action A
- prior 넣고 실행 → action B
- **|A - B| ≈ 0** (사실상 무반응)

이 시점에서 "L이 들어갔는데 왜 출력이 안 변하지?"라는 의문이 생겼다. per_token이 DiT에서 소비되는 유일한 경로인 **per_attn이 제대로 작동하는지** 의심하기 시작했다.

---

## 3. 2단계: per_attn 작동 불능 확인 (LIBERO ckpt + ManiSkill env)

per_attn이 정말 문제인지 확인하기 위해 **3단계 진단 프레임워크**를 구축했다.

### 3.1 Level 1: Weight Magnitude 검사 (이미지 불필요)

**방법:** per_attn의 `in_proj_weight` (Q/K/V 프로젝션) 절대값 통계를 측정

**코드 (`colab_inference_with_diagnostics.py:65-141`):**

```python
w_in = block.per_attn.in_proj_weight.data
in_mean = w_in.abs().mean()   # ← 이 값이 핵심
in_max  = w_in.abs().max()
```

**판정 기준:**

| in_proj_mean | in_proj_max | 판정 |
|---|---|---|
| < 1e-4 | < 1e-3 | **HIGH_RISK_DEAD** |
| < 0.005 | - | **LIKELY_DEAD** |
| ≥ 0.005 | - | POSSIBLY_ALIVE |

**결과:** 전 블록에서 `in_proj_mean ≈ 0.0003~0.0006`, `in_proj_max ≈ 0.001` 이하 → **HIGH_RISK_DEAD**

**해석:** in_proj가 0에 가까우면 Q = x @ W_q ≈ 0, K = per_token @ W_k ≈ 0, V = per_token @ W_v ≈ 0 → cross-attention output = 0. out_proj가 어떤 값이든 의미 없음 (input이 0이면 output도 0).

### 3.2 Level 2: Activation Ratio 측정 (dummy input 사용)

**방법:** DiT forward pass 중 각 블록에서 hook을 걸어 `||x_c|| / ||x||` 비율 측정

**코드 (`colab_inference_with_diagnostics.py:147-257`):**

```python
# hook 내부
x_after_sa = x_input + module.attn(module.norm1(x_input))
x_c, _ = module.per_attn(module.norm3(x_after_sa), per_tok, per_tok)

ratio = x_c.norm() / (x_after_sa.norm() + 1e-10)
```

**판정 기준:**

| ratio | 판정 |
|---|---|
| < 1e-4 | **DEAD** (per_attn 기여 = 0) |
| < 0.01 | WEAK |
| ≥ 0.01 | ACTIVE |

**결과:** 전 블록에서 `ratio < 1e-4` → **DEAD**

**해석:** self-attention 출력 대비 per_attn 출력이 0.01%도 안 됨. `x = x + x_c`에서 `x_c ≈ 0`이므로 per_attn은 존재하지 않는 것과 동일.

### 3.3 Level 3: Paired-Seed Output Sensitivity 테스트 (가장 결정적)

**방법:** 동일한 diffusion seed로 3가지 조건을 비교

- **A (Baseline):** prior 없음, cond_scale = 0.0
- **B (Prior ON):** random prior, strength=0.3, cond_scale = 0.0 → per_attn 경로 테스트
- **C (Bypass ON):** random prior, strength=0.3, cond_scale = 0.2 → condition 경로 테스트

동일한 seed면 동일한 diffusion noise → 차이는 순수하게 per_attn/bypass 효과

**핵심 코드 (`colab_inference_with_diagnostics.py:263-402`):**

```python
d_ab = |act_a - act_b|.mean()  # per_attn 경로 신호
d_ac = |act_a - act_c|.mean()  # bypass 경로 신호

# Null baseline: 서로 다른 seed의 A끼리 비교 → 고유 noise variance
null = |A(seed_i) - A(seed_j)|.mean()

# 판정: signal이 noise의 2배를 넘어야 유의미
per_attn_effective = (d_ab > null * 2) and (d_ab > 0.001)
```

**판정 기준:**

| 조건 | 판정 |
|---|---|
| d_ab > null × 2 | **PER_ATTN_ALIVE** |
| d_ab ≤ null × 2, d_ac > null × 2 | **PER_ATTN_DEAD_BYPASS_WORKS** |
| 둘 다 ≤ null × 2 | **BOTH_DEAD** |

**결과 (ManiSkill env):** `PER_ATTN_DEAD_BYPASS_WORKS`

- d_ab ≈ noise floor (per_attn 경로: 신호 없음)
- d_ac > noise × 2 (bypass 경로: 작동함)

→ **per_attn이 완전히 dead이고, per_token_cond_scale bypass만 신호를 전달할 수 있음**

---

## 4. 3단계: Domain Gap 여부 검증 (LIBERO ckpt + LIBERO env)

### 4.1 의문

ManiSkill에서 per_attn이 안 되는 게 **도메인 차이 (LIBERO→ManiSkill)** 때문인지, 아니면 **원래부터 dead**인지 확인해야 했다. 만약 LIBERO 환경에서는 per_attn이 살아있다면 domain gap 문제이고, LIBERO에서도 dead이면 아키텍처/학습 문제다.

### 4.2 LIBERO In-Domain 진단 구현

**전용 스크립트 (`diagnose_libero_per_attn.py`)** 를 작성:

1. LIBERO 환경 초기화 (libero_spatial / libero_object / libero_goal)
2. 초기 관측 캡처 (180° 회전 + JPEG encode/decode + lanczos3 resize → 학습 시 전처리와 동일)
3. 물체 안정화 대기 (10 step)
4. 동일한 3-Level 진단 실행

### 4.3 결과: LIBERO에서도 동일하게 Dead

| Level | ManiSkill 결과 | LIBERO 결과 | 일치 여부 |
|-------|---------------|-------------|---------|
| L1: Weight | HIGH_RISK_DEAD | HIGH_RISK_DEAD | 일치 |
| L2: Activation | DEAD (ratio < 1e-4) | DEAD (ratio < 1e-4) | 일치 |
| L3: Sensitivity | PER_ATTN_DEAD | PER_ATTN_DEAD | 일치 |

### 4.4 해석

**Domain gap이 원인이 아니다.** per_attn은 LIBERO 학습 환경에서도 이미 dead이었다. 즉:

- per_attn은 학습 과정에서 한 번도 유의미하게 활성화되지 않았음
- LIBERO의 96.5% 성능은 **per_attn 없이** 달성된 것
- cog_attn (self-attention on action tokens, conditioned by cog_tokens=LLM output)만으로 충분했음
- per_token (spatial vision features)은 DiT까지 도달하지 못하고 무시됨

---

## 5. 4단계: 근본 원인 규명 — Zero Initialization Trap

### 5.1 핵심: 초기화 방식의 차이

**DiT 전체 초기화 (`models.py:257-264`):**

```python
def initialize_weights(self):
    def _basic_init(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)  # ← 표준 초기화
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    self.apply(_basic_init)  # 모든 Linear에 적용
```

이 함수가 먼저 실행되어 DiT 내 **모든 nn.Linear를 Xavier uniform**으로 초기화한다. self-attention (cog_attn), MLP, embedder 등이 여기 해당.

**그런데 per_attn만 이후에 zero로 덮어씀 (`models.py:145-149`):**

```python
# zero initialization trick
nn.init.constant_(self.per_attn.in_proj_weight, 0.)
nn.init.constant_(self.per_attn.in_proj_bias, 0.)
nn.init.constant_(self.per_attn.out_proj.weight, 0.)
nn.init.constant_(self.per_attn.out_proj.bias, 0.)
```

### 5.2 왜 Zero Init이 학습을 방해했는가

Zero-init "trick"의 **원래 의도**는 잔차 연결에서 새 branch를 0으로 시작해 기존 모델을 방해하지 않으면서 점진적으로 학습하는 것이다 (ControlNet, Zero-Init Residual 등에서 사용). **그러나 per_attn에서는 이것이 역효과**를 냈다:

```
x = x + per_attn(norm3(x), per_token, per_token)
         ↓
Q = norm3(x) @ W_q = norm3(x) @ 0 = 0
K = per_token @ W_k = per_token @ 0 = 0
V = per_token @ W_v = per_token @ 0 = 0

attention = softmax(Q @ K^T / sqrt(d)) = softmax(0) = uniform(1/N)
output = uniform_attn @ V = uniform @ 0 = 0

∂L/∂W_q = ∂L/∂output × ∂output/∂Q × ∂Q/∂W_q
         = ∂L/∂output × (small_from_uniform_attn) × norm3(x)
         ≈ 매우 작은 gradient
```

**비교: self-attention (cog_attn)은 Xavier init:**

```
Q = x @ W_q (W_q ≈ Xavier ~ N(0, 1/d))
→ Q가 유의미한 값
→ attention pattern이 분화됨
→ gradient가 충분히 흐름
→ 학습 정상 진행
```

### 5.3 per_attn이 왜 학습 중 복구되지 않았는가

| 요인 | 설명 |
|------|------|
| Gradient 소실 | Q/K/V ≈ 0이면 attention이 uniform → gradient가 극도로 작음 |
| 잔차 구조 | `x = x + 0`이므로 loss가 per_attn 없이도 감소 → optimizer 입장에서 per_attn 학습 동기 없음 |
| cog_attn 충분 | self-attention + condition(t+z)만으로 action 예측에 충분 → per_attn 학습 필요성 자체가 약함 |
| Learning rate | 전체 모델에 동일 lr 적용 시, 0에서 시작하는 weight가 의미있는 값에 도달하기까지 매우 오래 걸림 |

결과적으로 **checkpoint에서 in_proj_weight ≈ 0.0003~0.0006** — 초기 0에서 거의 움직이지 않은 것.

---

## 6. 5단계: 우회 전략 수립 및 구현

per_attn이 dead라는 사실을 확인한 후, spatial grounding을 달성하기 위한 우회 전략을 마련했다.

### 6.1 시도 1: ControlDiT (실패)

**개념:** ControlMLLM 논문을 따라 per_token에 학습 가능한 perturbation pv를 더해, per_attn의 attention이 타겟 영역에 집중하도록 최적화

```python
pv = torch.zeros_like(per_tokens, requires_grad=True)
for step in range(T):
    modified_per = per_tokens + pv
    _, attn_stack = dit.forward(..., per_token=modified_per, return_attn_weights=True)
    score = (attn_avg * target_grid).sum() / attn_avg.sum()
    loss = (1 - score) ** 2
    pv = pv - alpha * pv.grad
```

**실패 이유:** per_attn의 in_proj ≈ 0이므로 per_token을 아무리 수정해도 Q/K/V ≈ 0 → attention pattern이 변하지 않음 → pv 최적화가 loss를 줄이지 못함.

**커밋 기록:** `6d74aa5` (구현) → `2c8a81d` (dead 발견으로 폐기)

### 6.2 시도 2: SpatialGatingControl — 성공 (Method A)

**개념:** dead per_attn을 통째로 교체. SAM 마스크를 직접 attention weight으로 사용.

**구현 (`spatial_control.py:77-139`):**

```python
@contextmanager
def spatial_attn_override(self, target_grid):
    attn_weights = self.compute_spatial_weights(target_grid)  # 마스크 → 정규화된 가중치

    for block in dit.blocks:
        if block.use_per_attn:
            # per_attn.forward()를 수동 spatial attention으로 교체
            block.per_attn.forward = _make_manual_attn(attn_weights, scale)

    try:
        yield  # 이 안에서 diffusion sampling 실행
    finally:
        # 원래 per_attn.forward() 복원
        restore_original()
```

**수동 attention 계산:**

```python
def manual_attn(query, key, value):
    # SAM 마스크 기반 가중합 (query 무시, 학습된 Q/K/V 무시)
    context = einsum('bn,bnd->bd', attn_weights, value)  # 타겟 영역 per_token의 가중합
    output = context.unsqueeze(1).expand(B, T, D) * scale  # 모든 action token에 broadcast
    return output, None
```

**장점:** 최적화 루프 불필요, 즉시 작동, dead weight에 의존하지 않음

### 6.3 시도 3: per_token_cond_scale Bypass (Method B)

**개념:** per_token 정보를 per_attn을 거치지 않고 condition c = t + z에 직접 주입

**구현 (`models.py:308-312`):**

```python
if per_token_cond_scale > 0.0 and per_token is not None:
    per_summary = per_token.mean(dim=1, keepdim=True)  # [B, 1, D]
    c = c + per_token_cond_scale * per_summary           # condition에 합산
```

**한계:** per_token.mean()은 공간 정보를 평균내어 버리므로, 특정 패치에 집중하는 spatial grounding 효과가 약함. 하지만 per_token 정보를 DiT에 전달하는 것 자체는 작동 확인됨 (Level 3 진단에서 bypass_effective = True).

### 6.4 시도 4: Soft Spatial Prior Injection (보조)

**개념:** SE Bottleneck 이후 per_token에 spatial prior를 additive/multiplicative로 주입

**구현 (`memory_vla.py:545-570`):** `_apply_per_token_prior()`

**한계:** L을 넣어도 per_attn이 dead이면 DiT까지 도달하지 못함. **Method A (SpatialGatingControl)와 결합해야 의미가 있음.**

---

## 7. 핵심 결론 및 시사점

### 7.1 발견 사항 요약

| 번호 | 발견 | 근거 |
|------|------|------|
| 1 | per_attn은 zero-init 후 학습되지 않았음 | in_proj ≈ 0.0003~0.0006 (24블록 전체) |
| 2 | cog_attn (self-attention)은 Xavier init → 정상 학습됨 | 초기화 차이가 핵심 원인 |
| 3 | LIBERO 96.5% 성능은 per_attn 없이 달성 | cog_tokens (LLM output) + self-attention만으로 충분 |
| 4 | Domain gap이 아님 | LIBERO env에서도 동일하게 dead |
| 5 | per_token (spatial feature)은 DiT에서 소비되지 않음 | 256개 패치의 공간 정보가 전혀 활용되지 않은 것 |

### 7.2 현재 질문: per_attn weight를 재초기화해야 하는가?

**현재 상황:**

```
vision_feats → SE Bottleneck → L 주입 → per_mem_bank → per_attn(DEAD) → 무시됨
```

L을 아무리 잘 넣어도 per_attn이 0이면 **출력에 반영되지 않는다.** 선택지:

| 방안 | 접근 | 리스크 |
|------|------|--------|
| A. SpatialGatingControl 유지 | per_attn 통째로 교체 (현재 구현) | scale 튜닝 필요, 학습된 weight 아님 |
| B. per_attn Xavier 재초기화 | inference time에 weight를 Xavier로 교체 | random weight → action 품질 저하 가능 |
| C. per_attn identity-like 초기화 | V만 identity, out_proj 작은 scale | per_token이 그대로 통과, 가장 안전 |
| D. per_attn fine-tuning | 소규모 데이터로 per_attn만 재학습 | 데이터 필요, 시간 소요 |

### 7.3 개발 타임라인 (커밋 기록)

```
Feb 10  4a5eeab  ControlDiT 초기 구현 시도 (depth init)
        ↓ (여러 시도: depth mask, saturation, SPSA 등)
Feb 13  6d74aa5  ControlDiT 정식 구현 (Grounding DINO + SAM → pv 최적화)
        c33c571  CFG uncondition 로직 수정
        2c8a81d  ★ per_attn dead 발견 → SpatialGatingControl (Method A) 구현
        18fc2e7  noise_seed 고정으로 비교 가능하게
        b965c5f  soft spatial prior + per_token_cond_scale bypass 추가
Feb 16  fb1b1c9  confidence return + SPSA latent optimization
Feb 17  9bee809  per_attn 3-level 진단 프레임워크 구축
        5525d6a  3-level 진단 구현 (weight → activation → sensitivity)
        1a30427  in_proj 기준 threshold 수정 (bottleneck 확인)
        f671c39  paired-seed Level 3 (noise confound 제거)
        180a284  LIBERO in-domain 진단 → domain gap 아님 확인
```

---

*End of Report*
