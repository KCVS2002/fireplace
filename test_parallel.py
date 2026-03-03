"""
병렬 환경 속도 테스트
단일 환경 vs 병렬 환경 속도 비교
"""

import os
import logging
import time
from train_selfplay import train_selfplay

if __name__ == '__main__':
    # fireplace 로그 비활성화
    logging.getLogger('fireplace').setLevel(logging.ERROR)

    print("="*60)
    print("병렬 환경 속도 테스트")
    print("="*60 + "\n")

    # 테스트 설정
    test_timesteps = 5000
    test_interval = 2000

    print("1️⃣  단일 환경 테스트 (n_envs=1)...")
    print("-" * 60)
    start = time.time()
    train_selfplay(
        total_timesteps=test_timesteps,
        update_interval=test_interval,
        save_interval=test_timesteps,
        model_dir="models/test_single",
        log_dir="logs/test_single",
        n_envs=1
    )
    single_time = time.time() - start
    print(f"✅ 단일 환경 완료: {single_time:.1f}초\n")

    print("2️⃣  병렬 환경 테스트 (n_envs=4)...")
    print("-" * 60)
    start = time.time()
    train_selfplay(
        total_timesteps=test_timesteps,
        update_interval=test_interval,
        save_interval=test_timesteps,
        model_dir="models/test_parallel",
        log_dir="logs/test_parallel",
        n_envs=4
    )
    parallel_time = time.time() - start
    print(f"✅ 병렬 환경 완료: {parallel_time:.1f}초\n")

    # 결과 비교
    print("="*60)
    print("📊 속도 비교 결과")
    print("="*60)
    print(f"단일 환경 (n_envs=1): {single_time:.1f}초")
    print(f"병렬 환경 (n_envs=4): {parallel_time:.1f}초")
    print(f"속도 향상: {single_time/parallel_time:.2f}배 빠름")
    print(f"효율성: {(single_time/parallel_time)/4*100:.1f}% (이상적으로는 100%)")
    print("="*60)
