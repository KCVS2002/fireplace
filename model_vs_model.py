"""
모델 간 대결 벤치마크

학습된 모델끼리 직접 대결하여 상대적 강함을 측정합니다.
"""

import os
import logging
import numpy as np
from rl_env import HearthstoneEnv, load_model, predict_with_mask
from hearthstone.enums import CardClass

# fireplace 로그 비활성화
logging.getLogger('fireplace').setLevel(logging.ERROR)


def model_vs_model(model1_path, model2_path, model1_name="Model 1", model2_name="Model 2", n_games=50):
    """
    두 모델을 직접 대결시킵니다.
    
    Args:
        model1_path: 모델 1 경로 (Agent로 플레이)
        model2_path: 모델 2 경로 (Opponent로 플레이)
        model1_name: 모델 1 이름
        model2_name: 모델 2 이름
        n_games: 플레이할 게임 수
    """
    print("="*60)
    print(f"⚔️  {model1_name} vs {model2_name}")
    print("="*60)
    print(f"게임 수: {n_games}")
    print("="*60 + "\n")
    
    # 모델 로드
    try:
        model1 = load_model(model1_path)
        model2 = load_model(model2_path)
        print(f"✅ {model1_name} 로드 완료")
        print(f"✅ {model2_name} 로드 완료\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None
    
    # 환경 생성 (model2를 opponent로 설정)
    env = HearthstoneEnv(
        player_class=CardClass.MAGE,
        opponent_class=CardClass.HUNTER,
        max_turns=200,
        render_mode=None,
        opponent_policy=model2  # Model 2가 opponent
    )
    
    # 결과 저장
    results = {
        'model1_wins': 0,
        'model2_wins': 0,
        'draws': 0,
        'rewards': [],
        'lengths': []
    }
    
    print("🎮 대결 시작...\n")
    
    for game_num in range(1, n_games + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = predict_with_mask(model1, obs, env, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        results['rewards'].append(episode_reward)
        results['lengths'].append(steps)
        
        if 'winner' in info:
            if info['winner'] == 'agent':
                results['model1_wins'] += 1
                outcome = f"{model1_name} 승리 🏆"
            elif info['winner'] == 'opponent':
                results['model2_wins'] += 1
                outcome = f"{model2_name} 승리 🏆"
            else:
                results['draws'] += 1
                outcome = "무승부 ⚖️"
        
        if game_num % 10 == 0 or game_num <= 5:
            m1_wr = results['model1_wins'] / game_num * 100
            print(f"[{game_num:3d}/{n_games}] {outcome} | "
                  f"{model1_name} 승률: {m1_wr:.1f}%")
    
    env.close()
    
    # 결과 출력
    total_games = results['model1_wins'] + results['model2_wins'] + results['draws']
    m1_win_rate = results['model1_wins'] / total_games * 100 if total_games > 0 else 0
    m2_win_rate = results['model2_wins'] / total_games * 100 if total_games > 0 else 0
    draw_rate = results['draws'] / total_games * 100 if total_games > 0 else 0
    
    print("\n" + "="*60)
    print("📊 대결 결과")
    print("="*60)
    print(f"{model1_name}: {results['model1_wins']}승 ({m1_win_rate:.1f}%)")
    print(f"{model2_name}: {results['model2_wins']}승 ({m2_win_rate:.1f}%)")
    print(f"무승부: {results['draws']}회 ({draw_rate:.1f}%)")
    print(f"\n평균 게임 길이: {np.mean(results['lengths']):.1f} 스텝")
    print("="*60)
    
    # 승자 결정
    if results['model1_wins'] > results['model2_wins']:
        print(f"\n🏆 승자: {model1_name}")
    elif results['model2_wins'] > results['model1_wins']:
        print(f"\n🏆 승자: {model2_name}")
    else:
        print("\n⚖️  무승부")
    
    return results


def round_robin_tournament(models, n_games=30):
    """
    여러 모델의 라운드 로빈 토너먼트를 실행합니다.
    
    Args:
        models: dict - {모델 이름: 모델 경로}
        n_games: 각 대결당 게임 수
    """
    print("\n" + "="*60)
    print("🏟️  라운드 로빈 토너먼트")
    print("="*60)
    print(f"참가 모델: {len(models)}개")
    print(f"각 대결: {n_games}게임")
    print("="*60 + "\n")
    
    model_names = list(models.keys())
    scores = {name: 0 for name in model_names}
    
    # 모든 조합으로 대결
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            path1 = models[name1]
            path2 = models[name2]
            
            if not os.path.exists(path1) or not os.path.exists(path2):
                print(f"⚠️  파일 없음: {name1} 또는 {name2}\n")
                continue
            
            # 양방향 대결 (공평성)
            print(f"\n{'='*60}")
            print(f"1라운드: {name1} (선공) vs {name2} (후공)")
            print(f"{'='*60}")
            results1 = model_vs_model(path1, path2, name1, name2, n_games // 2)
            
            print(f"\n{'='*60}")
            print(f"2라운드: {name2} (선공) vs {name1} (후공)")
            print(f"{'='*60}")
            results2 = model_vs_model(path2, path1, name2, name1, n_games // 2)
            
            if results1 and results2:
                # 점수 계산 (승 3점, 무 1점)
                scores[name1] += results1['model1_wins'] * 3 + results1['draws']
                scores[name2] += results1['model2_wins'] * 3 + results1['draws']
                scores[name2] += results2['model1_wins'] * 3 + results2['draws']
                scores[name1] += results2['model2_wins'] * 3 + results2['draws']
    
    # 최종 순위 출력
    print("\n" + "="*60)
    print("🏆 최종 순위표")
    print("="*60)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, score) in enumerate(sorted_scores, 1):
        print(f"{rank}. {name:<30} {score}점")
    
    print("="*60)


if __name__ == "__main__":
    # 테스트할 모델들
    models = {}
    
    if os.path.exists("models/final_model.zip"):
        models["Final Model (100k)"] = "models/final_model.zip"
    
    if os.path.exists("models/test/final_model.zip"):
        models["Test Model (5k)"] = "models/test/final_model.zip"
    
    if len(models) < 2:
        print("❌ 최소 2개의 모델이 필요합니다.")
        print("사용 가능한 모델:")
        for name, path in models.items():
            print(f"  - {name}: {path}")
    else:
        # 라운드 로빈 토너먼트 실행
        round_robin_tournament(models, n_games=50)
