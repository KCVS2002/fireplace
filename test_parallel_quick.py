"""
간단한 병렬 환경 테스트 - 병렬만 테스트
"""

import os
import logging
import time
from train_selfplay import train_selfplay

if __name__ == '__main__':
    # fireplace 로그 비활성화
    logging.getLogger('fireplace').setLevel(logging.ERROR)

    print("="*60)
    print("병렬 환경 테스트 (n_envs=4)")
    print("="*60 + "\n")

    start = time.time()
    train_selfplay(
        total_timesteps=5000,
        update_interval=2000,
        save_interval=5000,
        model_dir="models/test_parallel",
        log_dir="logs/test_parallel",
        n_envs=4
    )
    elapsed = time.time() - start
    
    print(f"\n✅ 병렬 환경 (n_envs=4) 완료: {elapsed:.1f}초")
    print(f"📊 단일 환경 대비: ~37.6초 → {elapsed:.1f}초")
    print(f"⚡ 속도 향상: {37.6/elapsed:.2f}배")
