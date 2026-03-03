"""
학습된 모델 벤치마크 테스트

학습된 모델 vs 랜덤 정책으로 여러 게임을 플레이하여 승률을 측정합니다.
"""

import os
import logging
import numpy as np
from rl_env import HearthstoneEnv, load_model, predict_with_mask
from hearthstone.enums import CardClass

# fireplace 로그 비활성화
logging.getLogger('fireplace').setLevel(logging.ERROR)


def benchmark_model(model_path, n_games=100, deterministic=True):
    """
    학습된 모델의 성능을 벤치마크합니다.
    
    Args:
        model_path: 모델 파일 경로 (.zip)
        n_games: 플레이할 게임 수
        deterministic: True면 결정적 행동, False면 확률적 행동
    
    Returns:
        dict: 벤치마크 결과 통계
    """
    print("="*60)
    print("🎯 모델 벤치마크 테스트")
    print("="*60)
    print(f"모델: {model_path}")
    print(f"게임 수: {n_games}")
    print(f"행동 선택: {'결정적 (Deterministic)' if deterministic else '확률적 (Stochastic)'}")
    print("="*60 + "\n")
    
    # 모델 로드
    try:
        model = load_model(model_path)
        print("✅ 모델 로드 완료\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None
    
    # 환경 생성 (opponent는 랜덤)
    env = HearthstoneEnv(
        player_class=CardClass.MAGE,
        opponent_class=CardClass.HUNTER,
        max_turns=200,
        render_mode=None,
        opponent_policy=None  # 랜덤 정책
    )
    
    # 결과 저장
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'total_rewards': [],
        'episode_lengths': [],
        'final_agent_hp': [],
        'final_opponent_hp': []
    }
    
    print("🎮 게임 시작...\n")
    
    for game_num in range(1, n_games + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # 모델로 행동 선택
            action, _ = predict_with_mask(model, obs, env, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        # 결과 기록
        results['total_rewards'].append(episode_reward)
        results['episode_lengths'].append(steps)
        
        if 'winner' in info:
            if info['winner'] == 'agent':
                results['wins'] += 1
                outcome = "승리 🏆"
            elif info['winner'] == 'opponent':
                results['losses'] += 1
                outcome = "패배 ❌"
            else:
                results['draws'] += 1
                outcome = "무승부 ⚖️"
        
        # 최종 체력 기록
        if hasattr(env, 'agent_player') and hasattr(env, 'opponent_player'):
            results['final_agent_hp'].append(env.agent_player.hero.health)
            results['final_opponent_hp'].append(env.opponent_player.hero.health)
        
        # 진행 상황 출력
        if game_num % 10 == 0 or game_num <= 5:
            win_rate = results['wins'] / game_num * 100
            print(f"[{game_num:3d}/{n_games}] {outcome} | "
                  f"보상: {episode_reward:+.2f} | 스텝: {steps:3d} | "
                  f"승률: {win_rate:.1f}%")
    
    env.close()
    
    # 통계 계산
    total_games = results['wins'] + results['losses'] + results['draws']
    win_rate = results['wins'] / total_games * 100 if total_games > 0 else 0
    loss_rate = results['losses'] / total_games * 100 if total_games > 0 else 0
    draw_rate = results['draws'] / total_games * 100 if total_games > 0 else 0
    
    avg_reward = np.mean(results['total_rewards'])
    std_reward = np.std(results['total_rewards'])
    avg_length = np.mean(results['episode_lengths'])
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 벤치마크 결과")
    print("="*60)
    print(f"총 게임: {total_games}")
    print(f"승리: {results['wins']} ({win_rate:.1f}%)")
    print(f"패배: {results['losses']} ({loss_rate:.1f}%)")
    print(f"무승부: {results['draws']} ({draw_rate:.1f}%)")
    print(f"\n평균 보상: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"평균 게임 길이: {avg_length:.1f} 스텝")
    
    if results['final_agent_hp'] and results['final_opponent_hp']:
        avg_agent_hp = np.mean(results['final_agent_hp'])
        avg_opp_hp = np.mean(results['final_opponent_hp'])
        print(f"\n평균 최종 체력:")
        print(f"  Agent: {avg_agent_hp:.1f} HP")
        print(f"  Opponent: {avg_opp_hp:.1f} HP")
    
    print("="*60)
    
    # 성능 평가
    print("\n💡 성능 평가:")
    if win_rate >= 60:
        print("   ✅ 우수: 랜덤 정책 대비 확실히 강합니다!")
    elif win_rate >= 50:
        print("   ⚠️  양호: 랜덤보다는 낫지만 더 학습이 필요합니다.")
    else:
        print("   ❌ 부족: 랜덤 정책보다 약합니다. 더 많은 학습이 필요합니다.")
    
    return results


def compare_models(model_paths, n_games=50):
    """
    여러 모델의 성능을 비교합니다.
    
    Args:
        model_paths: dict - {모델 이름: 모델 경로}
        n_games: 각 모델당 플레이할 게임 수
    """
    print("\n" + "="*60)
    print("🔬 모델 비교 벤치마크")
    print("="*60 + "\n")
    
    all_results = {}
    
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"⚠️  {name}: 파일이 존재하지 않습니다 ({path})\n")
            continue
        
        print(f"\n{'='*60}")
        print(f"테스트 중: {name}")
        print(f"{'='*60}")
        
        results = benchmark_model(path, n_games=n_games, deterministic=True)
        all_results[name] = results
    
    # 비교 결과 출력
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("📈 모델 비교 결과")
        print("="*60)
        print(f"{'모델명':<20} {'승률':>10} {'평균보상':>12} {'평균길이':>10}")
        print("-" * 60)
        
        for name, res in all_results.items():
            total = res['wins'] + res['losses'] + res['draws']
            win_rate = res['wins'] / total * 100 if total > 0 else 0
            avg_reward = np.mean(res['total_rewards'])
            avg_length = np.mean(res['episode_lengths'])
            
            print(f"{name:<20} {win_rate:>9.1f}% {avg_reward:>11.2f} {avg_length:>9.1f}")
        
        print("="*60)


if __name__ == "__main__":
    import sys
    
    # 기본 모델 경로
    default_model = "models/final_model.zip"
    
    if len(sys.argv) > 1:
        # 명령행에서 모델 경로를 지정한 경우
        model_path = sys.argv[1]
        n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        
        if os.path.exists(model_path):
            benchmark_model(model_path, n_games=n_games, deterministic=True)
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
    else:
        # 사용 가능한 모델들을 찾아서 비교
        models_to_test = {}
        
        if os.path.exists(default_model):
            models_to_test["Final Model"] = default_model
        
        # test 폴더의 모델
        test_model = "models/test/final_model.zip"
        if os.path.exists(test_model):
            models_to_test["Test Model"] = test_model
        
        # interrupted 모델
        interrupted_model = "models/interrupted_model.zip"
        if os.path.exists(interrupted_model):
            models_to_test["Interrupted Model"] = interrupted_model
        
        if not models_to_test:
            print("❌ 테스트할 모델을 찾을 수 없습니다.")
            print(f"기본 경로: {default_model}")
            print("\n사용법:")
            print("  python benchmark.py                    # 모든 모델 비교")
            print("  python benchmark.py <모델경로>          # 특정 모델 테스트")
            print("  python benchmark.py <모델경로> <게임수>  # 게임 수 지정")
        elif len(models_to_test) == 1:
            # 모델이 하나만 있으면 그것을 테스트
            name, path = list(models_to_test.items())[0]
            print(f"📦 발견된 모델: {name}")
            benchmark_model(path, n_games=100, deterministic=True)
        else:
            # 여러 모델이 있으면 비교
            compare_models(models_to_test, n_games=50)
