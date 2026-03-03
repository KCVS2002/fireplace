"""
Self-Play 시스템 간단 테스트 (짧은 학습)
"""

import os
import logging
from train_selfplay import train_selfplay

if __name__ == '__main__':
    # fireplace 로그 비활성화
    logging.getLogger('fireplace').setLevel(logging.ERROR)

    print("="*60)
    print("Self-Play 시스템 테스트")
    print("짧은 학습으로 시스템이 정상 작동하는지 확인합니다")
    print("="*60 + "\n")

    # 짧은 학습 (5,000 스텝만)
    train_selfplay(
        total_timesteps=5000,      # 5천 스텝만 테스트
        update_interval=2000,      # 2천 스텝마다 opponent 업데이트
        save_interval=5000,        
        model_dir="models/test",
        log_dir="logs/test",
        n_envs=1                   # 테스트는 단일 환경으로
    )

    print("\n✅ 테스트 완료!")
    print("실제 학습을 시작하려면: python train_selfplay.py")
