"""
게임 플레이 시각화

학습된 모델의 실제 게임 플레이를 상세히 출력합니다.
"""

import os
import logging
from stable_baselines3 import PPO
from rl_env import HearthstoneEnv
from hearthstone.enums import CardClass

# fireplace 로그 비활성화
logging.getLogger('fireplace').setLevel(logging.ERROR)


def watch_game(agent_model_path, opponent_model_path=None, n_games=3):
    """
    게임 플레이를 관찰합니다.
    
    Args:
        agent_model_path: Agent 모델 경로
        opponent_model_path: Opponent 모델 경로 (None이면 랜덤)
        n_games: 관찰할 게임 수
    """
    print("="*60)
    print("🎮 게임 플레이 관찰")
    print("="*60)
    
    # Agent 모델 로드
    if agent_model_path:
        agent_model = PPO.load(agent_model_path)
        print(f"✅ Agent: {agent_model_path}")
    else:
        agent_model = None
        print("✅ Agent: 랜덤")
    
    # Opponent 모델 로드
    if opponent_model_path:
        opponent_model = PPO.load(opponent_model_path)
        print(f"✅ Opponent: {opponent_model_path}")
    else:
        opponent_model = None
        print("✅ Opponent: 랜덤")
    
    print("="*60 + "\n")
    
    # 환경 생성 (로깅 활성화)
    env = HearthstoneEnv(
        player_class=CardClass.MAGE,
        opponent_class=CardClass.HUNTER,
        max_turns=200,
        render_mode=None,
        opponent_policy=opponent_model
    )
    
    for game_num in range(1, n_games + 1):
        print("\n" + "="*60)
        print(f"🎲 게임 {game_num}/{n_games}")
        print("="*60 + "\n")
        
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        print(f"초기 상태:")
        print(f"  Agent HP: {env.agent_player.hero.health}")
        print(f"  Opponent HP: {env.opponent_player.hero.health}")
        print()
        
        while not done and step < 100:  # 최대 100 스텝
            step += 1
            
            # 현재 턴 정보
            current_player = "Agent" if env.game.current_player == env.agent_player else "Opponent"
            print(f"\n--- 스텝 {step} (Turn {env.game_turn}, {current_player}) ---")
            
            # Agent 행동
            if agent_model:
                action, _ = agent_model.predict(obs, deterministic=True)
            else:
                # 랜덤 행동
                valid_actions = env.get_valid_actions()
                action = 0 if not valid_actions else min(len(valid_actions) - 1, action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # 상태 출력
            print(f"Agent HP: {env.agent_player.hero.health} | "
                  f"Opponent HP: {env.opponent_player.hero.health}")
            
            if reward != 0:
                print(f"💰 보상: {reward:+.1f}")
        
        # 게임 결과
        print("\n" + "="*60)
        print("🏁 게임 종료")
        print("="*60)
        print(f"총 스텝: {step}")
        print(f"총 보상: {total_reward:+.2f}")
        
        if 'winner' in info:
            winner = info['winner']
            if winner == 'agent':
                print("🏆 승자: Agent")
            elif winner == 'opponent':
                print("🏆 승자: Opponent")
            else:
                print("⚖️  무승부")
        
        print(f"\n최종 체력:")
        print(f"  Agent: {env.agent_player.hero.health} HP")
        print(f"  Opponent: {env.opponent_player.hero.health} HP")
        
        # 게임이 너무 짧게 끝났는지 확인
        if step < 20:
            print("\n⚠️  경고: 게임이 비정상적으로 짧게 끝났습니다!")
        
        print("="*60)
    
    env.close()


def analyze_first_turn_advantage():
    """
    선공 이점을 분석합니다.
    """
    print("\n" + "="*60)
    print("📊 선공 이점 분석")
    print("="*60)
    print("랜덤 vs 랜덤 100게임으로 선공 승률 측정")
    print("="*60 + "\n")
    
    env = HearthstoneEnv(
        player_class=CardClass.MAGE,
        opponent_class=CardClass.HUNTER,
        max_turns=200,
        render_mode=None,
        opponent_policy=None
    )
    
    first_player_wins = 0
    second_player_wins = 0
    draws = 0
    game_lengths = []
    
    for game_num in range(100):
        obs, info = env.reset()
        done = False
        steps = 0
        
        # 선공 확인 (플레이어 순서는 env.game.players[0]이 선공)
        first_player_is_agent = (env.game.players[0] == env.agent_player)
        
        while not done:
            # 랜덤 행동
            valid_actions = env.get_valid_actions()
            action = 0 if len(valid_actions) == 0 else env.action_space.sample() % len(valid_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        
        game_lengths.append(steps)
        
        if 'winner' in info:
            if info['winner'] == 'agent':
                if first_player_is_agent:
                    first_player_wins += 1
                else:
                    second_player_wins += 1
            elif info['winner'] == 'opponent':
                if not first_player_is_agent:
                    first_player_wins += 1
                else:
                    second_player_wins += 1
            else:
                draws += 1
        
        if (game_num + 1) % 20 == 0:
            print(f"진행: {game_num + 1}/100")
    
    env.close()
    
    # 결과
    total = first_player_wins + second_player_wins + draws
    first_rate = first_player_wins / total * 100 if total > 0 else 0
    second_rate = second_player_wins / total * 100 if total > 0 else 0
    
    print("\n" + "="*60)
    print("📊 분석 결과")
    print("="*60)
    print(f"선공 승률: {first_rate:.1f}% ({first_player_wins}승)")
    print(f"후공 승률: {second_rate:.1f}% ({second_player_wins}승)")
    print(f"무승부: {draws}회")
    print(f"평균 게임 길이: {sum(game_lengths)/len(game_lengths):.1f} 스텝")
    print("="*60)
    
    if first_rate > 70:
        print("\n⚠️  심각: 선공 이점이 너무 큽니다! (>70%)")
    elif first_rate > 60:
        print("\n⚠️  주의: 선공 이점이 큽니다. (60-70%)")
    elif first_rate > 55:
        print("\n✅ 정상: 적절한 선공 이점입니다. (55-60%)")
    else:
        print("\n✅ 정상: 균형잡힌 게임입니다. (<55%)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 명령행 인자로 모델 지정
        agent_path = sys.argv[1]
        opponent_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        watch_game(agent_path, opponent_path, n_games=3)
    else:
        # 선공 이점 분석
        analyze_first_turn_advantage()
        
        # Final 모델 관찰
        if os.path.exists("models/final_model.zip"):
            print("\n" + "="*60)
            print("Final Model 게임 관찰")
            print("="*60)
            watch_game("models/final_model.zip", None, n_games=2)
