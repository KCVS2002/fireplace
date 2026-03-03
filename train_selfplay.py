"""
Self-Play 강화학습 훈련 스크립트

Agent와 Opponent가 모두 학습하면서 서로 대결하는 방식으로 훈련합니다.
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO
_maskable_import_error = None
try:
    from sb3_contrib import MaskablePPO
except Exception as e:
    MaskablePPO = None
    _maskable_import_error = repr(e)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
_masker_import_error = None
try:
    from sb3_contrib.common.wrappers import ActionMasker
except Exception as e:
    ActionMasker = None
    _masker_import_error = repr(e)
from rl_env import HearthstoneEnv
from hearthstone.enums import CardClass
import logging
import copy
import multiprocessing as mp
from collections import deque
import time


class SelfPlayCallback(BaseCallback):
    """
    Self-play를 위한 커스텀 콜백
    일정 주기마다 현재 학습 중인 모델을 opponent로 복사합니다.
    """
    def __init__(self, update_interval=5000, n_envs=1, verbose=0):
        super().__init__(verbose)
        self.update_interval = update_interval  # opponent 업데이트 주기
        self.n_envs = n_envs  # 병렬 환경 개수
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_episodes = 0

    def _load_policy(self, path: str):
        if MaskablePPO is not None:
            return MaskablePPO.load(path)
        return PPO.load(path)
        
    def _on_step(self) -> bool:
        # 에피소드 진행 추적
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # 에피소드 종료 확인
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # 승패 기록
            info = self.locals.get('infos', [{}])[0]
            if 'winner' in info:
                if info['winner'] == 'agent':
                    self.wins += 1
                elif info['winner'] == 'opponent':
                    self.losses += 1
                else:
                    self.draws += 1
            
            self.total_episodes += 1
            
            # 100 에피소드마다 통계 출력
            if self.total_episodes % 100 == 0:
                recent_rewards = self.episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                recent_lengths = self.episode_lengths[-100:]
                avg_length = np.mean(recent_lengths) if recent_lengths else 0
                
                win_rate = self.wins / self.total_episodes * 100 if self.total_episodes > 0 else 0
                
                print(f"\n{'='*60}")
                print(f"Episodes: {self.total_episodes}")
                print(f"Recent 100 episodes:")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f} steps")
                print(f"  Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L/{self.draws}D)")
                print(f"{'='*60}\n")
            
            # 리셋
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Opponent 모델 업데이트
        if self.num_timesteps % self.update_interval == 0 and self.num_timesteps > 0:
            print(f"\n🔄 [Step {self.num_timesteps}] Updating opponent policy...")
            update_start = time.time()
            
            # 병렬 환경인 경우 환경을 재생성해야 함
            if self.n_envs > 1:
                # SubprocVecEnv는 직접 접근 불가 - 환경 재생성 필요
                # 모델을 저장하고 새 환경에서 로드
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip', delete=False) as f:
                    temp_path = f.name
                
                try:
                    self.model.save(temp_path)
                    opponent_policy = self._load_policy(temp_path)
                    
                    # 환경 재생성 (새 opponent_policy 사용)
                    from stable_baselines3.common.vec_env import SubprocVecEnv
                    old_env = self.training_env
                    new_env = SubprocVecEnv([
                        make_env(opponent_policy=opponent_policy) for _ in range(self.n_envs)
                    ])
                    
                    # 모델에 새 환경 설정
                    self.model.set_env(new_env)
                    
                    # 이전 환경 정리
                    old_env.close()
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                # 단일 환경인 경우 기존 방식
                env = self.training_env.envs[0]
                if hasattr(env, 'unwrapped'):
                    env = env.unwrapped
                
                # 모델을 파일로 저장 후 다시 로드 (pickle 문제 회피)
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip', delete=False) as f:
                    temp_path = f.name
                
                try:
                    self.model.save(temp_path)
                    env.opponent_policy = self._load_policy(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            print(f"✅ Opponent updated with current agent policy!\n")
            print(f"⏱️  Opponent update time: {time.time() - update_start:.2f}s\n")
            
            # 승률 초기화 (새로운 상대이므로)
            self.wins = 0
            self.losses = 0
            self.draws = 0
            self.total_episodes = 0
        
        return True


def make_env(opponent_policy=None):
    """환경 생성 함수"""
    def _init():
        env = HearthstoneEnv(
            player_class=CardClass.MAGE,
            opponent_class=CardClass.HUNTER,
            max_turns=200,  # 실제 승부가 날 수 있도록 충분히 긴 시간
            render_mode=None,  # 학습 중에는 렌더링 비활성화
            opponent_policy=opponent_policy
        )
        # 학습 중 I/O 병목 방지: 게임 로그 비활성화
        env.enable_logging = False
        # 마스킹 기대 여부 표시 (디버깅용)
        env._masking_expected = MaskablePPO is not None
        if MaskablePPO is not None:
            if ActionMasker is None:
                raise RuntimeError("MaskablePPO 사용을 위해 ActionMasker가 필요합니다.")
            env = ActionMasker(env, lambda e: e.get_action_mask())
            # 래퍼 내부의 실제 환경에도 마스킹 상태 표시
            try:
                env.unwrapped._masking_enabled = True
                if not env.unwrapped._masking_enabled:
                    raise RuntimeError("ActionMasker가 적용되지 않았습니다.")
            except Exception:
                raise RuntimeError("ActionMasker 적용 확인에 실패했습니다.")
        return env
    return _init


def train_selfplay(
    total_timesteps=100000,
    update_interval=5000,
    save_interval=10000,
    model_dir="models",
    log_dir="logs",
    n_envs=4,  # 병렬 환경 개수
    use_gpu=False,  # GPU 강제 사용 옵션
    show_progress=True  # 진행바 표시 여부
):
    """
    Self-play 방식으로 강화학습 훈련
    
    Args:
        total_timesteps: 총 학습 스텝 수
        update_interval: opponent 업데이트 주기
        save_interval: 모델 저장 주기
        model_dir: 모델 저장 디렉토리
        log_dir: 로그 저장 디렉토리
        n_envs: 병렬로 실행할 환경 개수 (1=단일, 4=4배속, 8=8배속)
        use_gpu: True면 GPU 강제 사용 (MlpPolicy는 비권장)
    """
    # 디렉토리 생성
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # fireplace 로그 레벨 설정 (경고만 표시)
    logging.getLogger('fireplace').setLevel(logging.WARNING)
    
    # 장치 선택
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        gpu_name = None
    
    # CPU 코어 수 확인
    cpu_count = mp.cpu_count()
    n_envs = min(n_envs, cpu_count)  # CPU 코어 수를 초과하지 않도록
    
    print("="*60)
    print("🎮 Hearthstone Self-Play 강화학습 시작")
    print("="*60)
    print(f"Script: {__file__}")
    print(f"MaskablePPO: {'ON' if MaskablePPO is not None else 'OFF'} | ActionMasker: {'ON' if ActionMasker is not None else 'OFF'}")
    if MaskablePPO is None and _maskable_import_error:
        print(f"[IMPORT ERROR] MaskablePPO: {_maskable_import_error}")
    if ActionMasker is None and _masker_import_error:
        print(f"[IMPORT ERROR] ActionMasker: {_masker_import_error}")
    print(f"총 학습 스텝: {total_timesteps:,}")
    print(f"Opponent 업데이트 주기: {update_interval:,} steps")
    print(f"모델 저장 주기: {save_interval:,} steps")
    algo_name = "MaskablePPO" if MaskablePPO is not None else "PPO"
    print(f"알고리즘: {algo_name}")
    print(f"ActionMasker: {'사용 가능' if ActionMasker is not None else '사용 불가'}")
    print(f"병렬 환경: {n_envs}개 (CPU 코어: {cpu_count}개)")
    print(f"학습 장치: {device.upper()}", end="")
    if gpu_name:
        print(f" ({gpu_name})")
    else:
        print(" (MlpPolicy는 CPU가 더 효율적)")
    if n_envs > 1:
        print(f"💡 {n_envs}개 환경 병렬 실행으로 약 {n_envs}배 빠른 데이터 수집")
    print("="*60 + "\n")
    
    # 환경 생성 (처음엔 opponent가 랜덤)
    if n_envs > 1:
        # 멀티프로세싱으로 병렬 환경 실행
        env = SubprocVecEnv([make_env(opponent_policy=None) for _ in range(n_envs)])
    else:
        # 단일 환경
        env = DummyVecEnv([make_env(opponent_policy=None)])

    # 마스킹 적용 검증 (단일 환경에서만 확인)
    if n_envs == 1 and MaskablePPO is not None:
        inner_env = env.envs[0]
        if not getattr(inner_env.unwrapped, "_masking_enabled", False):
            raise RuntimeError(
                "Action Masking이 활성화되지 않았습니다. "
                "ActionMasker 적용 여부를 확인하세요."
            )
    
    # PPO 모델 생성 (가능하면 action mask 사용)
    algo = MaskablePPO if MaskablePPO is not None else PPO
    model = algo(
        "MlpPolicy",
        env,
        learning_rate=1e-4,     # 3e-4 → 1e-4 (더 안정적인 학습)
        n_steps=4096,           # 2048 → 4096 (더 많은 경험 수집)
        batch_size=128,         # 64 → 128 (더 안정적인 그래디언트)
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # 탐험 유지
        verbose=1,
        tensorboard_log=log_dir,
        device=device
    )
    
    print("✅ PPO 모델 생성 완료")
    print(f"Algorithm: {algo.__name__}")
    print(f"Policy: MlpPolicy")
    print(f"Learning Rate: 1e-4 (안정적 학습)")
    print(f"N Steps: 4096 (더 많은 경험)")
    print(f"Batch Size: 128 (안정적 그래디언트)\n")
    
    # Self-play 콜백 생성
    callback = SelfPlayCallback(
        update_interval=update_interval,
        n_envs=n_envs,  # 병렬 환경 개수 전달
        verbose=1
    )
    
    print("🚀 학습 시작...\n")
    
    # 학습 실행
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=show_progress
        )
        
        # 최종 모델 저장
        final_model_path = os.path.join(model_dir, "final_model")
        save_start = time.time()
        model.save(final_model_path)
        print(f"⏱️  Final model save time: {time.time() - save_start:.2f}s")
        print(f"\n✅ 최종 모델 저장: {final_model_path}.zip")
        
        # 통계 출력
        print("\n" + "="*60)
        print("🎉 학습 완료!")
        print("="*60)
        print(f"총 에피소드: {callback.total_episodes}")
        print(f"최종 승률: {callback.wins / callback.total_episodes * 100:.1f}%" if callback.total_episodes > 0 else "N/A")
        recent_rewards = list(callback.episode_rewards)[-100:]
        recent_lengths = list(callback.episode_lengths)[-100:]
        print(f"평균 보상: {np.mean(recent_rewards):.2f}" if recent_rewards else "평균 보상: N/A")
        print(f"평균 에피소드 길이: {np.mean(recent_lengths):.1f} steps" if recent_lengths else "평균 에피소드 길이: N/A")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 학습 중단됨 (Ctrl+C)")
        
        # 중간 모델 저장
        interrupt_model_path = os.path.join(model_dir, "interrupted_model")
        save_start = time.time()
        model.save(interrupt_model_path)
        print(f"⏱️  Interrupt model save time: {time.time() - save_start:.2f}s")
        print(f"💾 중간 모델 저장: {interrupt_model_path}.zip")
    
    finally:
        env.close()


if __name__ == "__main__":
    # 기본 설정으로 학습 시작
    train_selfplay(
        total_timesteps=200000,    # 10만 → 20만 (더 긴 학습)
        update_interval=10000,     # 5천 → 1만 (더 안정적인 Self-Play)
        save_interval=20000,       # 2만 스텝마다 모델 저장
        model_dir="models",
        log_dir="logs",
        n_envs=1                   # 단일 환경이 더 빠름
    )
