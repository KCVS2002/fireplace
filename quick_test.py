"""
간단한 선공 이점 테스트 (10게임)
"""

import logging
from rl_env import HearthstoneEnv
from hearthstone.enums import CardClass

logging.getLogger('fireplace').setLevel(logging.ERROR)

env = HearthstoneEnv(
    player_class=CardClass.MAGE,
    opponent_class=CardClass.HUNTER,
    max_turns=200,
    render_mode=None,
    opponent_policy=None
)

print("="*60)
print("⚡ 빠른 선공 이점 테스트 (10게임)")
print("="*60 + "\n")

first_player_wins = 0
second_player_wins = 0
draws = 0
game_lengths = []

for game_num in range(10):
    obs, info = env.reset()
    done = False
    steps = 0
    
    # 선공 확인
    first_player_is_agent = (env.game.players[0] == env.agent_player)
    
    while not done and steps < 500:
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
                result = "선공 승리"
            else:
                second_player_wins += 1
                result = "후공 승리"
        elif info['winner'] == 'opponent':
            if not first_player_is_agent:
                first_player_wins += 1
                result = "선공 승리"
            else:
                second_player_wins += 1
                result = "후공 승리"
        else:
            draws += 1
            result = "무승부"
    
    first_turn = "Agent" if first_player_is_agent else "Opponent"
    print(f"게임 {game_num+1}: 선공={first_turn}, 결과={result}, 스텝={steps}")

env.close()

total = first_player_wins + second_player_wins + draws
first_rate = first_player_wins / total * 100 if total > 0 else 0
second_rate = second_player_wins / total * 100 if total > 0 else 0

print("\n" + "="*60)
print("📊 결과")
print("="*60)
print(f"선공 승률: {first_rate:.1f}% ({first_player_wins}승)")
print(f"후공 승률: {second_rate:.1f}% ({second_player_wins}승)")
print(f"무승부: {draws}회")
print(f"평균 게임 길이: {sum(game_lengths)/len(game_lengths):.1f} 스텝")
print("="*60)

if first_rate >= 80:
    print("\n⚠️ 심각: 선공이 압도적으로 유리합니다! (80%+)")
    print("   → 게임 밸런스에 문제가 있습니다.")
elif first_rate >= 65:
    print("\n⚠️ 주의: 선공 이점이 매우 큽니다. (65-80%)")
    print("   → AI 학습이 제대로 안 될 수 있습니다.")
elif first_rate >= 55:
    print("\n✅ 정상: 적절한 선공 이점입니다. (55-65%)")
else:
    print("\n✅ 균형: 게임이 균형잡혀 있습니다. (<55%)")
