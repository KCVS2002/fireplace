"""
학습된 모델로 게임 플레이 및 평가
"""

from rl_env import HearthstoneEnv, load_model, predict_with_mask
from hearthstone.enums import CardClass
import numpy as np
import logging


def evaluate_model(model_path, num_episodes=10, render=True, opponent_policy=None):
    """
    학습된 모델을 평가합니다.
    
    Args:
        model_path: 모델 파일 경로 (.zip 제외)
        num_episodes: 평가할 에피소드 수
        render: 게임 화면 출력 여부
        opponent_policy: 상대 정책 (None이면 랜덤)
    """
    # fireplace 로그 레벨 설정
    logging.getLogger('fireplace').setLevel(logging.WARNING)
    
    print("="*60)
    print("🎮 학습된 모델 평가")
    print("="*60)
    print(f"모델: {model_path}")
    print(f"에피소드 수: {num_episodes}")
    print(f"상대: {'학습된 모델' if opponent_policy else '랜덤 AI'}")
    print("="*60 + "\n")
    
    # 모델 로드
    model = load_model(model_path)
    print("✅ 모델 로드 완료\n")

    # 환경 생성
    env = HearthstoneEnv(
        player_class=CardClass.MAGE,
        opponent_class=CardClass.HUNTER,
        max_turns=50,
        render_mode='human' if render else None,
        opponent_policy=opponent_policy
    )

    # 통계
    wins = 0
    losses = 0
    draws = 0
    total_rewards = []
    total_steps = []

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"에피소드 {episode + 1}/{num_episodes}")
        print(f"{'='*60}\n")

        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        if render:
            env.render()

        while not (done or truncated):
            # 모델로 액션 예측 (deterministic=True: 가장 확률 높은 액션 선택)
            action, _states = predict_with_mask(model, obs, env, deterministic=True)
            
            # 액션 실행
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if render and step_count % 5 == 0:  # 5스텝마다 렌더링
                env.render()
        
        # 최종 상태 렌더링
        if render:
            env.render()
        
        # 결과 기록
        total_rewards.append(episode_reward)
        total_steps.append(step_count)
        
        if 'winner' in info:
            if info['winner'] == 'agent':
                wins += 1
                result = "🏆 승리!"
            elif info['winner'] == 'opponent':
                losses += 1
                result = "💀 패배"
            else:
                draws += 1
                result = "🤝 무승부"
        else:
            result = "❓ 결과 없음"
        
        print(f"\n{'='*60}")
        print(f"에피소드 {episode + 1} 결과: {result}")
        print(f"보상: {episode_reward:.2f}, 스텝: {step_count}")
        print(f"{'='*60}\n")
    
    env.close()
    
    # 최종 통계
    win_rate = wins / num_episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    
    print("\n" + "="*60)
    print("📊 평가 결과")
    print("="*60)
    print(f"총 에피소드: {num_episodes}")
    print(f"승리: {wins}, 패배: {losses}, 무승부: {draws}")
    print(f"승률: {win_rate:.1f}%")
    print(f"평균 보상: {avg_reward:.2f}")
    print(f"평균 스텝: {avg_steps:.1f}")
    print("="*60 + "\n")
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps
    }


def play_interactive(model_path):
    """
    학습된 모델과 대화형으로 플레이 (한 게임씩 실행)
    """
    logging.getLogger('fireplace').setLevel(logging.WARNING)

    model = load_model(model_path)
    print(f"✅ 모델 로드 완료: {model_path}\n")
    
    while True:
        print("\n" + "="*60)
        print("새 게임 시작 (Ctrl+C로 종료)")
        print("="*60 + "\n")
        
        env = HearthstoneEnv(
            player_class=CardClass.MAGE,
            opponent_class=CardClass.HUNTER,
            max_turns=50,
            render_mode='human'
        )
        
        obs, info = env.reset()
        done = False
        truncated = False
        step_count = 0
        
        env.render()
        
        while not (done or truncated):
            # 액션 예측
            action, _states = predict_with_mask(model, obs, env, deterministic=True)

            # 스텝 실행
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # 렌더링
            env.render()
            
            # 게임 종료 확인
            if done or truncated:
                if 'winner' in info:
                    if info['winner'] == 'agent':
                        print("\n🏆 승리!")
                    elif info['winner'] == 'opponent':
                        print("\n💀 패배")
                    else:
                        print("\n🤝 무승부")
                
                print(f"총 {step_count} 스텝")
                break
        
        env.close()
        
        # 계속 여부 확인
        continue_play = input("\n다른 게임 하시겠습니까? (y/n): ")
        if continue_play.lower() != 'y':
            break
    
    print("\n게임 종료!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  평가: python test_trained_model.py <모델경로> [에피소드수]")
        print("  예시: python test_trained_model.py models/final_model 20")
        print("  대화형: python test_trained_model.py models/final_model interactive")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if len(sys.argv) > 2 and sys.argv[2] == 'interactive':
        # 대화형 모드
        play_interactive(model_path)
    else:
        # 평가 모드
        num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        evaluate_model(model_path, num_episodes=num_episodes, render=True)
