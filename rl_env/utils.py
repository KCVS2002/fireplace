"""
RL 환경 유틸리티 함수
"""

from stable_baselines3 import PPO

try:
    from sb3_contrib import MaskablePPO
except Exception:
    MaskablePPO = None


def load_model(model_path):
    """MaskablePPO 또는 PPO 모델을 자동으로 로드합니다."""
    if MaskablePPO is not None:
        try:
            return MaskablePPO.load(model_path)
        except Exception:
            pass
    return PPO.load(model_path)


def predict_with_mask(model, obs, env, deterministic=True):
    """액션 마스크를 적용하여 모델 예측을 수행합니다."""
    action_mask = env.get_action_mask()
    try:
        return model.predict(obs, deterministic=deterministic, action_masks=action_mask)
    except TypeError:
        return model.predict(obs, deterministic=deterministic)
