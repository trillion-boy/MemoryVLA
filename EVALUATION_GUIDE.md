# MemoryVLA 올바른 평가 가이드

## 🚨 기존 코드의 문제점

당신이 작성한 "핵" 코드들은 다음과 같은 치명적 문제가 있습니다:

### 1. **통계 데이터 돌려막기** (가장 심각)
```python
model.norm_stats["libero_spatial_no_noops"] = model.norm_stats["bridge_orig"]
```

**문제:**
- `bridge_orig`: 실제 로봇 팔 데이터 (액션 범위 ~0.05m)
- `libero_spatial`: 시뮬레이션 데이터 (액션 범위가 다름)
- **결과:** 모델이 0.5m 움직이라고 해도 un-normalization 후 0.005m가 되어 **사실상 정지**

**비유:**
- 한국 돈(원) 통계로 미국 돈(달러)을 계산하는 격
- 1000원이라고 하면 $1000로 계산되어버림

### 2. **회전 정보 강제 삭제**
```python
mw_action = np.concatenate([action[:3], [action[-1]]])  # roll, pitch, yaw 버림
```

**문제:**
- MetaWorld는 4-DoF만 받지만, 모델은 7-DoF로 학습됨
- **Pick-Place는 회전 없이 거의 불가능**

**비유:**
- 3D 게임 캐릭터에게 "앞/뒤/좌/우"만 알려주고 "위/아래/회전"은 숨김
- 계단 올라가라고 해도 못 올라감

### 3. **환경 불일치** (근본적 문제)
- MemoryVLA는 **LIBERO 환경**으로 학습됨
- MetaWorld는 **완전히 다른 물리 엔진, 로봇, 카메라**
- Zero-shot transfer가 될 수도 있지만... 확률 낮음

**비유:**
- 한국에서 운전 배운 사람을 미국에 데려다 놓고 "왜 운전 못해?"라고 하는 격
- (좌핸들 vs 우핸들, 도로 규칙 다름)

---

## ✅ 올바른 평가 방법

### **옵션 A: LIBERO 환경 (강력 추천)**

모델이 실제로 학습한 환경이므로 **가장 정확합니다**.

```bash
# 1. LIBERO 설치 (아직 안 했다면)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

# 2. 평가 실행
cd /home/user/MemoryVLA
python eval_libero_proper.py
```

**예상 결과:**
- 성공률: 60-80% (정상)
- 액션이 자연스럽게 움직임
- 회전도 제대로 적용됨

---

### **옵션 B: MetaWorld (Zero-shot, 실험적)**

LIBERO와 다른 환경이므로 **성공률이 낮을 수 있습니다**.

```bash
# 1. 진단 먼저 실행
python diagnose_model.py

# 출력 예시:
#   XYZ Magnitude: 0.0001  ← 너무 작으면 문제!
#   XYZ Magnitude: 0.05    ← 적절
#   XYZ Magnitude: 1.5     ← 너무 크면 문제!

# 2. 스케일 조정
# eval_metaworld_fixed.py 파일 열어서 수정:
ACTION_SCALE = 10.0  # 0.0001이면 10.0으로 증폭

# 3. 평가 실행
python eval_metaworld_fixed.py
```

**예상 결과:**
- 성공률: 0-30% (Zero-shot이므로 낮음)
- 액션 스케일 맞추기 어려움
- 회전 정보 부족으로 실패 가능

---

## 🔍 문제 진단 흐름도

```
1. diagnose_model.py 실행
   ↓
2. "XYZ Magnitude" 확인
   ↓
3-1. < 0.001  → ACTION_SCALE = 100.0
3-2. 0.01-0.1 → ACTION_SCALE = 1.0 (적절)
3-3. > 1.0    → ACTION_SCALE = 0.1
   ↓
4. eval_metaworld_fixed.py 실행
   ↓
5. 여전히 안 되면?
   → LIBERO 환경에서 테스트 (eval_libero_proper.py)
```

---

## 📊 각 스크립트 설명

| 파일 | 용도 | 추천도 |
|------|------|--------|
| `eval_libero_proper.py` | LIBERO 환경 평가 (정석) | ⭐⭐⭐⭐⭐ |
| `eval_metaworld_fixed.py` | MetaWorld 평가 (개선판) | ⭐⭐⭐ |
| `diagnose_model.py` | 모델 출력 진단 | ⭐⭐⭐⭐ |
| `eval_metaworld_nuclear_v2.py` (기존) | 응급 처방 (비추천) | ⭐ |

---

## 💡 자주 묻는 질문

### Q1: "왜 로봇이 안 움직이나요?"
**A:** 통계 데이터 불일치로 액션 스케일이 0.0001처럼 극도로 작아졌기 때문입니다.

**해결:**
1. `diagnose_model.py`로 확인
2. `ACTION_SCALE` 조정
3. 또는 LIBERO 환경 사용

### Q2: "MetaWorld에서 꼭 테스트해야 하나요?"
**A:** 아니요. MemoryVLA는 LIBERO로 학습되었으므로 LIBERO에서 먼저 테스트하세요.

### Q3: "bridge_orig 통계를 쓰면 안 되나요?"
**A:** 안 됩니다. 데이터셋마다 액션 분포가 다릅니다.

**비유:**
- Bridge Dataset: 실제 로봇 (작은 움직임)
- LIBERO: 시뮬레이션 (큰 움직임)
- 통계 뒤섞으면 스케일이 틀어짐

### Q4: "회전 정보를 꼭 써야 하나요?"
**A:** MetaWorld는 4-DoF만 받지만, 모델은 7-DoF로 학습되었습니다.
- 회전 정보를 버리면 성능 저하 불가피
- LIBERO 환경은 7-DoF를 그대로 사용 가능

### Q5: "ACTION_SCALE은 어떻게 정하나요?"
**A:** 실험적으로 결정해야 합니다.

1. `diagnose_model.py`로 XYZ Magnitude 확인
2. MetaWorld 액션 범위: [-1, 1] (각 축당 ~0.05m)
3. 적절한 크기: 0.01-0.1

---

## 🎯 권장 워크플로우

```bash
# 1단계: 진단
python diagnose_model.py
# → 모델이 정상 작동하는지, 통계 데이터가 있는지 확인

# 2단계: LIBERO 평가 (정석)
python eval_libero_proper.py
# → 모델이 학습한 환경에서 테스트
# → 60%+ 성공률 예상

# 3단계: MetaWorld 평가 (선택)
# ACTION_SCALE 조정 후
python eval_metaworld_fixed.py
# → Zero-shot이므로 낮은 성공률 예상 (0-30%)
```

---

## ⚠️ 주의사항

1. **절대 하지 말 것:**
   - `bridge_orig` 통계를 `libero_spatial`에 복사 ❌
   - 회전 정보 무작정 삭제 ❌
   - ACTION_SCALE 없이 MetaWorld 평가 ❌

2. **반드시 할 것:**
   - LIBERO 환경 먼저 테스트 ✅
   - `diagnose_model.py`로 진단 ✅
   - 액션 크기 모니터링 ✅

3. **현실적인 기대치:**
   - LIBERO: 60-80% 성공률
   - MetaWorld: 0-30% 성공률 (Zero-shot)
   - 회전 없이는 복잡한 조작 불가능

---

## 📚 추가 자료

- [LIBERO 논문](https://arxiv.org/abs/2306.03310)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [MetaWorld 문서](https://meta-world.github.io/)

---

**마지막 조언:**

> "기존의 'nuclear' 방식은 **모델이 로드되는지만 확인**하는 용도였습니다.
> 실제 성능을 보려면 **올바른 환경(LIBERO)**에서 테스트하세요."

**핵심 요약:**
1. 통계 돌려막기 → 액션 스케일 망가짐
2. 회전 삭제 → 조작 능력 저하
3. 환경 불일치 → Zero-shot 실패 가능

**해결책:**
→ **LIBERO 환경에서 eval_libero_proper.py 실행!**
