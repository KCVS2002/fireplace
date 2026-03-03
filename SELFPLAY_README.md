# Hearthstone AI - Self-Play 강화학습

하스스톤 카드 게임을 플레이하는 강화학습 AI입니다. Self-Play 방식으로 Agent가 과거 버전의 자신과 대결하며 학습합니다.

## 📋 요구사항

```bash
conda activate fireplace_rl
# 이미 설치됨: gymnasium, numpy, torch, stable-baselines3
```

## 🚀 사용 방법

### 1. Self-Play 학습 시작

```bash
python train_selfplay.py
```

**학습 파라미터 (파일 수정 가능):**
- `total_timesteps`: 총 학습 스텝 (기본: 100,000)
- `update_interval`: Opponent 업데이트 주기 (기본: 5,000 스텝)
- 5,000 스텝마다 현재 Agent가 Opponent로 복사됨

**학습 중 출력 예시:**
```
Episodes: 100
Recent 100 episodes:
  Avg Reward: 0.45
  Avg Length: 42.3 steps
  Win Rate: 55.0% (55W/40L/5D)

🔄 [Step 5000] Updating opponent policy...
✅ Opponent updated with current agent policy!
```

### 2. 학습된 모델 평가

```bash
# 10 에피소드 평가 (기본)
python test_trained_model.py models/final_model

# 20 에피소드 평가
python test_trained_model.py models/final_model 20

# 대화형 모드 (한 게임씩 관전)
python test_trained_model.py models/final_model interactive
```

### 3. 환경 테스트 (개발용)

```bash
python test_rl_env.py
```

## 📁 파일 구조

```
fireplace/
├── rl_env/
│   ├── __init__.py
│   └── hearthstone_env.py    # Gymnasium 환경 (Self-Play 지원)
├── train_selfplay.py          # Self-Play 학습 스크립트
├── test_trained_model.py      # 모델 평가 스크립트
├── test_rl_env.py             # 환경 테스트
├── models/                    # 학습된 모델 저장
│   └── final_model.zip
├── logs/                      # TensorBoard 로그
└── game_logs/                 # 상세 게임 로그
    └── game_20260205_HHMMSS.log
```

## 🎮 Self-Play 작동 방식

1. **초기**: Agent vs 랜덤 AI
2. **5,000 스텝 후**: Agent vs 과거 Agent (5,000스텝 시점)
3. **10,000 스텝 후**: Agent vs 과거 Agent (10,000스텝 시점)
4. **계속 반복**: Agent가 점점 강해지면 Opponent도 강해짐

## 📊 게임 로그

### 터미널 출력 (간결)
```
=== Turn 5 (Step 23) ===
Agent:    HP 28 | Mana 5/5 | Hand 6 | Field 3
Opponent: HP 25 | Mana 5/5 | Hand 7 | Field 2
```

### 로그 파일 (상세)
`game_logs/game_YYYYMMDD_HHMMSS.log`:
```
[Turn 3, Step 4] Agent: 카드 플레이 - Masked Contender (코스트: 3)
[Turn 3, Step 5] Agent: 공격 - Arcane Servant[2/3] -> Rexxar

--- Game State (Turn 3, Step 5) ---
Agent: Jaina Proudmoore HP: 30, Mana: 0/3
  Hand (6): Emerald Reaver, Faerie Dragon, The Coin, ...
  Field: Arcane Servant[2/3], Masked Contender[2/4]
Opponent: Rexxar HP: 28, Mana: 0/3
  Field: Fel Orc Soulfiend[3/3]
```

## 🧠 알고리즘: PPO (Proximal Policy Optimization)

- **Policy Network**: MlpPolicy (Multi-Layer Perceptron)
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Gamma**: 0.99 (할인율)
- **GAE Lambda**: 0.95

## 🎯 보상 체계

- **승리**: +1.0
- **패배**: -1.0
- **진행 중**: 0.0

## 💡 팁

### 학습 속도 향상
- `total_timesteps` 증가 (예: 500,000)
- `update_interval` 조정 (너무 자주 업데이트하면 불안정)

### TensorBoard로 학습 모니터링
```bash
tensorboard --logdir=logs
```

### 중단된 학습 재개
학습 중 Ctrl+C로 중단하면 `models/interrupted_model.zip`이 저장됩니다.
이를 로드하여 계속 학습 가능합니다.

## 🔧 커스터마이징

### 다른 영웅 클래스 사용
`train_selfplay.py`에서:
```python
env = HearthstoneEnv(
    player_class=CardClass.WARRIOR,  # 변경
    opponent_class=CardClass.PRIEST,  # 변경
    ...
)
```

### 게임 길이 조정
```python
env = HearthstoneEnv(
    max_turns=30,  # 기본 50
    ...
)
```

## 📈 예상 학습 시간

- **CPU**: 약 2-3시간 (100,000 스텝)
- **GPU**: 약 1-2시간 (100,000 스텝)

학습 시간은 하드웨어에 따라 다릅니다.
