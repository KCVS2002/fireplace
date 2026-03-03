"""
강화학습 환경 테스트 스크립트
"""

from rl_env import HearthstoneEnv
from hearthstone.enums import CardClass
import sys
import logging


def test_env():
    """환경이 제대로 작동하는지 테스트합니다."""
    
    # fireplace 로그를 파일로 저장
    log_file = open('game_log.txt', 'w', encoding='utf-8')
    
    # fireplace 로거 설정
    fireplace_logger = logging.getLogger('fireplace')
    fireplace_logger.setLevel(logging.WARNING)  # WARNING 이상만 출력
    
    print("=== Hearthstone RL 환경 테스트 ===\n")
    
    # 환경 생성
    env = HearthstoneEnv(
        player_class=CardClass.MAGE,
        opponent_class=CardClass.HUNTER,
        max_turns=20,
        render_mode='human'
    )
    
    print("환경 생성 완료!")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}\n")
    
    # 에피소드 실행
    observation, info = env.reset()
    print("게임 초기화 완료!\n")
    
    total_reward = 0
    step_count = 0
    actions_taken = []
    last_game_turn = 0
    
    # 초기 상태 출력
    env.render()
    
    for step in range(100):  # 최대 100 스텝
        # 유효한 액션 가져오기
        valid_actions = env.get_valid_actions()
        
        # 랜덤 액션 선택
        action_idx = env.action_space.sample() % len(valid_actions)
        selected_action = valid_actions[action_idx]
        
        # 액션 기록
        if selected_action.type != 'end_turn':
            actions_taken.append(str(selected_action))
        
        observation, reward, terminated, truncated, info = env.step(action_idx)
        total_reward += reward
        step_count += 1
        
        # 턴이 바뀌었는지 체크 (매 턴마다 출력)
        current_game_turn = info['game_turn']
        if current_game_turn != last_game_turn:
            print(f"\n{'='*60}")
            env.render()
            print(f"{'='*60}")
            last_game_turn = current_game_turn
        
        if terminated or truncated:
            # 최종 상태 출력
            print("\n" + "="*50)
            env.render()
            print("="*50)
            print(f"\n게임 종료!")
            print(f"이유: {'승부 결정' if terminated else '최대 턴 도달'}")
            print(f"총 스텝: {step_count}")
            print(f"총 보상: {total_reward}")
            print(f"최종 정보: {info}")
            
            # 주요 액션들 출력
            print(f"\n주요 액션 수 (턴 종료 제외): {len(actions_taken)}")
            if actions_taken:
                print("처음 10개 액션:")
                for i, action in enumerate(actions_taken[:10]):
                    print(f"  {i+1}. {action}")
            
            break
    
    log_file.close()
    env.close()
    
    # 로그 파일 분석
    print("\n" + "="*50)
    print("로그 파일 분석 (game_log.txt)")
    print("="*50)
    analyze_log()
    
    print("\n테스트 완료!")


def analyze_log():
    """로그 파일을 분석하여 요약 정보를 출력합니다."""
    try:
        with open('game_log.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if lines:
            print(f"총 로그 라인 수: {len(lines)}")
            
            # 주요 이벤트 카운트
            card_plays = sum(1 for line in lines if 'play' in line.lower())
            attacks = sum(1 for line in lines if 'attack' in line.lower())
            deaths = sum(1 for line in lines if 'dead' in line.lower() or 'death' in line.lower())
            
            print(f"카드 플레이: ~{card_plays} 번")
            print(f"공격: ~{attacks} 번")
            print(f"사망 이벤트: ~{deaths} 번")
        else:
            print("로그가 비어있습니다 (정상 - WARNING 레벨만 기록)")
    except FileNotFoundError:
        print("로그 파일을 찾을 수 없습니다.")


if __name__ == "__main__":
    test_env()
