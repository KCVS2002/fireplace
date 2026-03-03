from fireplace import cards
from fireplace.game import Game, BaseGame, CoinRules
from fireplace.player import Player
from fireplace.utils import random_draft
from hearthstone.enums import CardClass 
import random


def auto_mulligan(game, strategy='random'):
    """
    자동으로 멀리건을 처리하는 함수
    
    Args:
        game: 게임 객체
        strategy: 멀리건 전략 ('random', 'keep_all', 'replace_all', 'smart')
    """
    for player in game.players:
        if player.choice:
            if strategy == 'keep_all':
                # 모든 카드 유지
                player.choice.choose()
            elif strategy == 'replace_all':
                # 모든 카드 교체
                player.choice.choose(*player.choice.cards)
            elif strategy == 'random':
                # 랜덤하게 일부 카드 교체
                num_to_replace = random.randint(0, len(player.choice.cards))
                cards_to_replace = random.sample(player.choice.cards, num_to_replace)
                player.choice.choose(*cards_to_replace)
            elif strategy == 'smart':
                # 간단한 휴리스틱: 3코스트 이상 카드는 교체
                cards_to_replace = [c for c in player.choice.cards if c.cost >= 3]
                player.choice.choose(*cards_to_replace)


# 멀리건 없이 바로 게임을 시작하는 커스텀 게임 클래스
class GameWithoutMulligan(CoinRules, BaseGame):
    """멀리건 단계를 건너뛰고 바로 게임을 시작하는 클래스"""
    pass


def run_test_game():
    # 카드 데이터베이스 초기화
    cards.db.initialize()
    
    # 랜덤 덱 생성
    deck1 = random_draft(card_class=CardClass.MAGE)
    deck2 = random_draft(card_class=CardClass.HUNTER)
    
    # 플레이어 생성 - default_hero를 사용
    p1 = Player("Player1", deck1, CardClass.MAGE.default_hero)
    p2 = Player("Player2", deck2, CardClass.HUNTER.default_hero)

    # 일반 게임 시작 (멀리건 포함)
    game = Game(players=(p1, p2))
    game.start()
    
    # 멀리건 자동 처리 (랜덤 전략 사용)
    # 나중에 강화학습 에이전트가 이 부분을 결정하게 됨
    auto_mulligan(game, strategy='random')

    print(f"게임 시작! P1: {p1.hero} vs P2: {p2.hero}")
    
    for i in range(10):
        current_player = game.current_player
        print(f"Turn {game.turn}: {current_player}의 차례 (마나: {current_player.mana})")
        
        game.end_turn()

        if game.ended:
            print("게임 종료!")
            break

if __name__ == "__main__":
    run_test_game()