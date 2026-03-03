"""
Hearthstone Gymnasium 환경

강화학습을 위한 하스스톤 게임 환경을 제공합니다.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import datetime
import os

from fireplace import cards
from fireplace.game import Game
from fireplace.player import Player
from fireplace.utils import random_draft
from fireplace.exceptions import GameOver, InvalidAction
from hearthstone.enums import CardClass, CardType, Zone
import random


class Action:
    """게임 액션을 표현하는 클래스"""
    def __init__(self, action_type: str, source=None, target=None, index=None):
        self.type = action_type  # 'end_turn', 'play_card', 'attack', 'hero_power'
        self.source = source
        self.target = target
        self.index = index
    
    def __repr__(self):
        if self.type == 'end_turn':
            return "Action(END_TURN)"
        elif self.type == 'play_card':
            return f"Action(PLAY {self.source})"
        elif self.type == 'attack':
            return f"Action(ATTACK {self.source} -> {self.target})"
        elif self.type == 'hero_power':
            return f"Action(HERO_POWER {self.target if self.target else ''})"
        return f"Action({self.type})"


class HearthstoneEnv(gym.Env):
    """
    하스스톤 강화학습 환경
    
    Gymnasium API를 따르는 하스스톤 게임 환경입니다.
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self, 
        player_class: CardClass = CardClass.MAGE,
        opponent_class: CardClass = CardClass.HUNTER,
        max_turns: int = 50,
        render_mode: Optional[str] = None,
        opponent_policy = None  # Self-play를 위한 상대 정책
    ):
        super().__init__()
        
        self.player_class = player_class
        self.opponent_class = opponent_class
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.opponent_policy = opponent_policy  # 학습된 모델 또는 None(랜덤)
        
        # 카드 데이터베이스 초기화 (한 번만)
        if not hasattr(cards.db, '_initialized'):
            cards.db.initialize()
            cards.db._initialized = True
        
        # 액션 스페이스
        self.action_space = spaces.Discrete(100)
        self._action_list = []
        
        # 상태 스페이스
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(50,), dtype=np.float32
        )
        
        self.game = None
        self.agent_player = None
        self.opponent_player = None
        self.step_count = 0
        self.game_turn = 0
        self._masking_enabled = False
        
        # 로그 파일 설정
        self.log_file = None
        self.enable_logging = True
        self._setup_log_file()

    def _safe_health(self, entity) -> int:
        """체력을 로그/관찰용으로 안전하게 보정합니다."""
        try:
            return max(0, int(entity.health))
        except Exception:
            return 0

    def _select_valid_action(self, valid_actions, action_index: int, player_name: str):
        """항상 유효한 액션을 선택하도록 보정합니다."""
        if not valid_actions:
            return None

        if action_index >= len(valid_actions):
            remapped_index = action_index % len(valid_actions)
            if self._masking_enabled:
                self._log(
                    f"[Turn {self.game_turn}, Step {self.step_count}] {player_name}: "
                    f"마스킹 활성화 상태인데 유효하지 않은 액션 인덱스({action_index}) 발생 → {remapped_index}로 보정"
                )
            else:
                self._log(
                    f"[Turn {self.game_turn}, Step {self.step_count}] {player_name}: "
                    f"유효하지 않은 액션 인덱스({action_index}) → {remapped_index}로 보정"
                )
            action_index = remapped_index

        return valid_actions[action_index]
    
    def __getstate__(self):
        """객체를 pickle할 때 로그 파일을 제외합니다."""
        state = self.__dict__.copy()
        # 로그 파일은 pickle할 수 없으므로 제외
        state['log_file'] = None
        return state
    
    def __setstate__(self, state):
        """pickle에서 복원할 때 로그 파일을 다시 생성합니다."""
        self.__dict__.update(state)
        if self.enable_logging:
            self._setup_log_file()
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """환경을 초기화하고 게임을 시작합니다."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 새 게임 생성
        deck1 = random_draft(card_class=self.player_class)
        deck2 = random_draft(card_class=self.opponent_class)
        
        self.agent_player = Player("Agent", deck1, self.player_class.default_hero)
        self.opponent_player = Player("Opponent", deck2, self.opponent_class.default_hero)
        
        self.game = Game(players=(self.agent_player, self.opponent_player))
        self.game.start()
        
        # 멀리건 자동 처리
        self._auto_mulligan()
        
        # 로그 파일 새로 생성
        self._setup_log_file()
        self._log("=" * 60)
        self._log("NEW GAME STARTED")
        self._log(f"Agent: {self.agent_player.hero.id} vs Opponent: {self.opponent_player.hero.id}")
        self._log(f"Action Masking: {'ON' if self._masking_enabled else 'OFF'}")
        self._log(f"Masking Expected: {'ON' if getattr(self, '_masking_expected', False) else 'OFF'}")
        if getattr(self, "_masking_expected", False) and not self._masking_enabled:
            self._log("[WARN] Masking expected but not enabled. Check ActionMasker 적용 여부.")
        self._log("=" * 60)
        
        self.step_count = 0
        
        # 초기 턴 카운트 계산 (양쪽 플레이어 중 최대 마나 기준)
        # 현재 진행중인 라운드 번호를 나타냄 (선공/후공 모두 같은 라운드로 표시)
        if self.game and not self.game.ended:
            agent_turn = self.agent_player.max_mana
            opp_turn = self.opponent_player.max_mana
            self.game_turn = max(agent_turn, opp_turn)
        else:
            self.game_turn = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """액션을 수행하고 환경을 한 스텝 진행합니다."""
        try:
            if self.game.current_player == self.agent_player:
                # 선택 카드 처리 (Discover 등)
                if self.agent_player.choice:
                    choice = self.agent_player.choice
                    if choice.cards:
                        chosen_card = random.choice(choice.cards)
                        self._log(f"[Turn {self.game_turn}, Step {self.step_count}] Agent: 선택 - {chosen_card}")
                        choice.choose(chosen_card)
                        # 선택 후 다시 액션 수행
                        return self.step(action)
                
                # 유효한 액션 목록 가져오기
                valid_actions = self.get_valid_actions()
                
                # 액션 실행 (유효하지 않은 인덱스는 보정)
                action_obj = self._select_valid_action(valid_actions, action, "Agent")
                if action_obj is not None:
                    self._log_action("Agent", action_obj)
                    self._execute_action(action_obj)
                else:
                    # 가능한 액션이 없으면 턴 종료
                    if not self.agent_player.choice and not self.game.ended:
                        self._log(f"[Turn {self.game_turn}, Step {self.step_count}] Agent: 턴 종료 (가능한 액션 없음)")
                        self.game.end_turn()
            
            # 상대 턴 처리 (랜덤 봇)
            if not self.game.ended and self.game.current_player == self.opponent_player:
                self._opponent_turn()
                
        except GameOver:
            # 게임 종료는 정상적인 흐름
            pass
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            # Fireplace 엔진의 버그나 예상치 못한 게임 상태 발생
            # 로그를 남기고 게임을 강제 종료
            self._log(f"[ERROR] Fireplace 엔진 오류 발생: {type(e).__name__}: {e}")
            self._log(f"[ERROR] 게임을 강제 종료합니다. (Turn {self.game_turn}, Step {self.step_count})")
            
            # 게임을 무승부로 강제 종료
            if not self.game.ended:
                self.game.ended = True
                # 무승부로 처리하기 위해 아무도 승리하지 않은 상태로 설정
        
        # 턴 카운트 업데이트 (현재 진행중인 라운드 번호)
        # 양쪽 중 최대 마나를 기준으로 현재 라운드를 계산
        if self.game and not self.game.ended:
            agent_turn = self.agent_player.max_mana
            opp_turn = self.opponent_player.max_mana
            self.game_turn = max(agent_turn, opp_turn)
        
        self.step_count += 1
        
        # 스텝 종료 시 게임 상태 로깅
        if self.step_count % 5 == 0:  # 5스텝마다 상세 게임 상태 기록
            self._log_game_state()
        
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self.game.ended
        truncated = self.game_turn >= self.max_turns
        info = self._get_info()
        
        # truncated 상태에서는 무승부로 처리 (체력 기반 판정 제거)
        # 하스스톤은 체력이 낮아도 역전 가능하므로 실제 승부만 인정
        if truncated and not terminated:
            info['winner'] = 'draw'
        
        if terminated or truncated:
            self._log_game_state()
            self._log(f"\n{'='*60}")
            self._log(f"게임 종료 - Terminated: {terminated}, Truncated: {truncated}")
            if terminated:
                winner = "Agent" if self.opponent_player.hero.dead else "Opponent"
                self._log(f"승자: {winner}")
            self._log(f"{'='*60}\n")
            if self.log_file:
                self.log_file.flush()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """게임 상태를 터미널에 요약하여 출력합니다. (상세 로그는 파일에 기록됨)"""
        if self.render_mode == 'human' or self.render_mode == 'ansi':
            print(f"\n=== Turn {self.game_turn} (Step {self.step_count}) ===")
            
            # Agent 상태
            agent_minions = len(self.agent_player.field)
            agent_cards = len(self.agent_player.hand)
            print(f"Agent:    HP {self.agent_player.hero.health:2d} | Mana {self.agent_player.mana}/{self.agent_player.max_mana} | Hand {agent_cards} | Field {agent_minions}")
            
            # Opponent 상태
            opp_minions = len(self.opponent_player.field)
            opp_cards = len(self.opponent_player.hand)
            print(f"Opponent: HP {self.opponent_player.hero.health:2d} | Mana {self.opponent_player.mana}/{self.opponent_player.max_mana} | Hand {opp_cards} | Field {opp_minions}")
    
    def close(self):
        """환경을 정리합니다."""
        if self.log_file:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None
    
    def _build_observation(self, my_player, their_player) -> np.ndarray:
        """
        플레이어 관점의 관찰값을 구축합니다.

        50차원 벡터 구성:
        [0]     내 영웅 HP / 30
        [1]     내 영웅 방어도 / 30
        [2]     내 마나 / 10
        [3]     내 최대 마나 / 10
        [4]     내 손패 수 / 10
        [5]     내 필드 미니언 수 / 7
        [6]     내 덱 잔여 카드 수 / 30
        [7-13]  내 필드 미니언 공격력 (7슬롯, /15)
        [14-20] 내 필드 미니언 체력 (7슬롯, /15)
        [21]    상대 영웅 HP / 30
        [22]    상대 영웅 방어도 / 30
        [23]    상대 마나 / 10
        [24]    상대 최대 마나 / 10
        [25]    상대 손패 수 / 10
        [26]    상대 필드 미니언 수 / 7
        [27]    상대 덱 잔여 카드 수 / 30
        [28-34] 상대 필드 미니언 공격력 (7슬롯, /15)
        [35-41] 상대 필드 미니언 체력 (7슬롯, /15)
        [42]    내 무기 공격력 / 10
        [43]    내 무기 내구도 / 5
        [44]    내 영웅 능력 사용 가능 여부 (0/1)
        [45]    내 영웅 공격 가능 여부 (0/1)
        [46]    게임 턴 / 50
        [47]    플레이 가능한 카드 수 / 10
        [48]    내 필드 총 공격력 / 30
        [49]    상대 필드 총 공격력 / 30
        """
        obs = np.zeros(50, dtype=np.float32)

        # --- 내 정보 ---
        obs[0] = self._safe_health(my_player.hero) / 30
        obs[1] = getattr(my_player.hero, 'armor', 0) / 30
        obs[2] = my_player.mana / 10
        obs[3] = my_player.max_mana / 10
        obs[4] = len(my_player.hand) / 10
        obs[5] = len(my_player.field) / 7
        obs[6] = len(my_player.deck) / 30

        # 내 필드 미니언 상세
        for i, minion in enumerate(my_player.field[:7]):
            obs[7 + i] = minion.atk / 15
            obs[14 + i] = self._safe_health(minion) / 15

        # --- 상대 정보 ---
        obs[21] = self._safe_health(their_player.hero) / 30
        obs[22] = getattr(their_player.hero, 'armor', 0) / 30
        obs[23] = their_player.mana / 10
        obs[24] = their_player.max_mana / 10
        obs[25] = len(their_player.hand) / 10
        obs[26] = len(their_player.field) / 7
        obs[27] = len(their_player.deck) / 30

        # 상대 필드 미니언 상세
        for i, minion in enumerate(their_player.field[:7]):
            obs[28 + i] = minion.atk / 15
            obs[35 + i] = self._safe_health(minion) / 15

        # --- 추가 정보 ---
        weapon = my_player.weapon
        if weapon:
            obs[42] = weapon.atk / 10
            obs[43] = weapon.durability / 5

        try:
            obs[44] = 1.0 if my_player.hero.power.is_usable() else 0.0
        except Exception:
            obs[44] = 0.0

        try:
            obs[45] = 1.0 if my_player.hero.can_attack() else 0.0
        except Exception:
            obs[45] = 0.0

        obs[46] = self.game_turn / 50

        # 플레이 가능한 카드 수
        try:
            playable = sum(1 for c in my_player.hand if c.is_playable())
            obs[47] = playable / 10
        except Exception:
            obs[47] = 0.0

        # 필드 총 공격력
        my_total_atk = sum(m.atk for m in my_player.field)
        their_total_atk = sum(m.atk for m in their_player.field)
        obs[48] = my_total_atk / 30
        obs[49] = their_total_atk / 30

        return obs

    def _get_observation(self) -> np.ndarray:
        """현재 게임 상태를 관찰값으로 변환합니다."""
        if not self.game or not self.agent_player.hero:
            return np.zeros(50, dtype=np.float32)
        return self._build_observation(self.agent_player, self.opponent_player)

    def _get_observation_for_opponent(self) -> np.ndarray:
        """상대 플레이어 관점의 관찰값을 반환합니다 (Self-play용)."""
        if not self.game or not self.opponent_player.hero:
            return np.zeros(50, dtype=np.float32)
        return self._build_observation(self.opponent_player, self.agent_player)
    
    def _get_info(self) -> Dict[str, Any]:
        """추가 정보를 반환합니다."""
        info = {
            'game_turn': self.game_turn,
            'step_count': self.step_count,
        }
        
        if self.game and self.game.ended:
            if self.agent_player.hero.dead:
                info['winner'] = 'opponent'
            elif self.opponent_player.hero.dead:
                info['winner'] = 'agent'
            else:
                info['winner'] = 'draw'
        
        return info
    
    def _calculate_reward(self) -> float:
        """
        보상을 계산합니다.
        
        개선된 보상 체계:
        1. 승리/패배: ±1.0 (최종 목표)
        2. 체력 차이: ±0.01 per HP (중간 목표)
        3. 필드 우위: ±0.005 per 공격력 차이 (전략적 목표)
        4. 카드 어드밴티지: ±0.002 per 카드 (자원 관리)
        """
        reward = 0.0
        
        # 1. 게임 종료 보상 (가장 중요)
        if self.game.ended:
            if self.agent_player.hero.dead:
                return -1.0
            elif self.opponent_player.hero.dead:
                return 1.0
            else:
                return 0.0
        
        # 2. 최대 턴 도달 (무승부)
        if self.game_turn >= self.max_turns:
            return 0.0
        
        # 3. 중간 보상: 체력 차이 (생존력)
        agent_hp = self.agent_player.hero.health
        opponent_hp = self.opponent_player.hero.health
        hp_diff = agent_hp - opponent_hp
        reward += hp_diff * 0.001  # -30 ~ +30 HP 차이 → -0.03 ~ +0.03
        
        # 4. 필드 우위 보상 (공격력 합계)
        agent_attack = sum(minion.atk for minion in self.agent_player.field)
        opponent_attack = sum(minion.atk for minion in self.opponent_player.field)
        attack_diff = agent_attack - opponent_attack
        reward += attack_diff * 0.002  # 공격력 차이에 비례
        
        # 5. 카드 어드밴티지 (핸드 개수)
        hand_diff = len(self.agent_player.hand) - len(self.opponent_player.hand)
        reward += hand_diff * 0.001  # 카드 우위
        
        # 중간 보상은 작게 유지 (최종 승패가 가장 중요)
        # 최대 약 ±0.05 정도의 중간 보상
        reward = np.clip(reward, -0.1, 0.1)
        
        return reward
    
    def get_valid_actions(self):
        """현재 유효한 액션 목록을 반환합니다."""
        actions = []
        player = self.game.current_player
        
        if not self.game or self.game.ended:
            return actions
        
        # 턴 종료는 항상 가능
        actions.append(Action('end_turn'))
        
        # 카드 플레이
        for i, card in enumerate(player.hand):
            if card.is_playable():
                if card.requires_target():
                    # 타겟이 필요한 카드
                    for target in card.targets:
                        actions.append(Action('play_card', source=card, target=target, index=i))
                else:
                    # 타겟이 필요 없는 카드
                    actions.append(Action('play_card', source=card, target=None, index=i))
        
        # 영웅 파워
        if player.hero.power.is_usable():
            if player.hero.power.requires_target():
                for target in player.hero.power.targets:
                    actions.append(Action('hero_power', source=player.hero.power, target=target))
            else:
                actions.append(Action('hero_power', source=player.hero.power, target=None))
        
        # 공격
        for minion in player.field:
            if minion.can_attack():
                for target in minion.attack_targets:
                    actions.append(Action('attack', source=minion, target=target))
        
        # 영웅 공격 (무기가 있을 때)
        if player.hero.can_attack():
            for target in player.hero.attack_targets:
                actions.append(Action('attack', source=player.hero, target=target))
        
        return actions

    def get_action_mask(self) -> np.ndarray:
        """현재 유효한 액션 마스크를 반환합니다 (MaskablePPO용)."""
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        valid_actions = self.get_valid_actions()
        valid_count = min(len(valid_actions), self.action_space.n)
        if valid_count > 0:
            mask[:valid_count] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO가 요구하는 마스크 함수."""
        return self.get_action_mask()
    
    def _execute_action(self, action: Action):
        """액션을 실행합니다."""
        try:
            if self.game.ended:
                return  # 게임이 이미 종료되었으면 아무것도 하지 않음
                
            if action.type == 'end_turn':
                self.game.end_turn()
            elif action.type == 'play_card':
                if action.target:
                    action.source.play(target=action.target)
                else:
                    action.source.play()
            elif action.type == 'attack':
                action.source.attack(action.target)
            elif action.type == 'hero_power':
                if action.target:
                    action.source.use(target=action.target)
                else:
                    action.source.use()
        except GameOver:
            # 게임 종료는 정상적인 흐름
            pass
        except InvalidAction as e:
            # 유효하지 않은 액션은 무시
            pass
        except Exception as e:
            # 기타 예외는 로그만 남기고 계속 진행
            pass
    
    def _opponent_turn(self):
        """상대의 턴을 처리합니다 (랜덤 또는 학습된 정책)."""
        self._log(f"\n[Turn {self.game_turn}] === Opponent's Turn ===")
        
        try:
            while self.game.current_player == self.opponent_player and not self.game.ended:
                # 선택 카드 처리 (제프리스 등)
                if self.opponent_player.choice:
                    choice = self.opponent_player.choice
                    if choice.cards:
                        # 랜덤으로 선택
                        chosen_card = random.choice(choice.cards)
                        self._log(f"[Turn {self.game_turn}, Step {self.step_count}] Opponent: 선택 - {chosen_card}")
                        choice.choose(chosen_card)
                        continue
                
                valid_actions = self.get_valid_actions()
                if not valid_actions:
                    break
                
                # 학습된 정책이 있으면 사용, 없으면 랜덤
                if self.opponent_policy is not None:
                    # Self-play: 학습된 모델로 행동 선택
                    obs = self._get_observation_for_opponent()
                    action_mask = self.get_action_mask()
                    try:
                        action_idx, _ = self.opponent_policy.predict(
                            obs, deterministic=False, action_masks=action_mask
                        )
                    except TypeError:
                        action_idx, _ = self.opponent_policy.predict(obs, deterministic=False)
                    
                    action = self._select_valid_action(valid_actions, action_idx, "Opponent")
                    if action is not None:
                        self._log_action("Opponent", action)
                        self._execute_action(action)
                    else:
                        # 가능한 액션이 없으면 턴 종료
                        if not self.opponent_player.choice and not self.game.ended:
                            self._log(f"[Turn {self.game_turn}, Step {self.step_count}] Opponent: 턴 종료 (가능한 액션 없음)")
                            self.game.end_turn()
                        break
                else:
                    # 랜덤 AI
                    if random.random() < 0.3:
                        if not self.opponent_player.choice and not self.game.ended:
                            self._log(f"[Turn {self.game_turn}, Step {self.step_count}] Opponent: 턴 종료")
                            self.game.end_turn()
                            break
                    
                    non_end_turn_actions = [a for a in valid_actions if a.type != 'end_turn']
                    if non_end_turn_actions:
                        action = random.choice(non_end_turn_actions)
                        self._log_action("Opponent", action)
                        self._execute_action(action)
                    else:
                        if not self.opponent_player.choice and not self.game.ended:
                            self._log(f"[Turn {self.game_turn}, Step {self.step_count}] Opponent: 턴 종료")
                            self.game.end_turn()
                            break
        except GameOver:
            # 게임이 종료되면 정상적으로 루프 종료
            pass
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            # Fireplace 엔진의 버그나 예상치 못한 게임 상태
            self._log(f"[ERROR] Opponent turn 오류: {type(e).__name__}: {e}")
            self._log(f"[ERROR] Opponent 턴을 강제 종료합니다.")
            
            # 게임을 무승부로 강제 종료
            if not self.game.ended:
                self.game.ended = True
    
    def _auto_mulligan(self):
        """자동으로 멀리건을 처리합니다."""
        for player in self.game.players:
            if player.choice:
                # 랜덤으로 멀리건 (3코스트 이상 카드 교체)
                to_mulligan = [
                    card for card in player.choice.cards
                    if card.cost >= 3
                ]
                player.choice.choose(*to_mulligan)
    
    def _setup_log_file(self):
        """로그 파일을 생성합니다."""
        if not self.enable_logging:
            return
        
        # 로그 디렉토리 생성
        log_dir = "game_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 로그 파일 이름 (타임스탬프 포함)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"game_{timestamp}.log")
        
        # 기존 파일이 있으면 닫기
        if self.log_file:
            self.log_file.close()
        
        # 새 로그 파일 열기
        self.log_file = open(log_filename, 'w', encoding='utf-8')
        
    def _log(self, message: str):
        """로그 파일에 메시지를 기록합니다."""
        if self.enable_logging and self.log_file:
            self.log_file.write(message + '\n')
    
    def _log_action(self, player_name: str, action: Action):
        """액션을 상세하게 로그에 기록합니다."""
        timestamp = f"[Turn {self.game_turn}, Step {self.step_count}]"
        
        if action.type == 'end_turn':
            self._log(f"{timestamp} {player_name}: 턴 종료")
        elif action.type == 'play_card':
            card = action.source
            target_str = f" -> {action.target}" if action.target else ""
            self._log(f"{timestamp} {player_name}: 카드 플레이 - {card}{target_str} (코스트: {card.cost})")
        elif action.type == 'attack':
            self._log(f"{timestamp} {player_name}: 공격 - {action.source}[{action.source.atk}/{action.source.health}] -> {action.target}")
        elif action.type == 'hero_power':
            target_str = f" -> {action.target}" if action.target else ""
            self._log(f"{timestamp} {player_name}: 영웅 능력 사용{target_str}")
    
    def _log_game_state(self):
        """현재 게임 상태를 로그에 기록합니다."""
        self._log(f"\n--- Game State (Turn {self.game_turn}, Step {self.step_count}) ---")
        self._log(
            f"Agent: {self.agent_player.hero} HP: {self._safe_health(self.agent_player.hero)}, "
            f"Mana: {self.agent_player.mana}/{self.agent_player.max_mana}"
        )
        self._log(f"  Hand ({len(self.agent_player.hand)}): {', '.join([str(c) for c in self.agent_player.hand])}")
        if self.agent_player.field:
            self._log(f"  Field: {', '.join([f'{str(m)}[{m.atk}/{m.health}]' for m in self.agent_player.field])}")
        
        self._log(
            f"Opponent: {self.opponent_player.hero} HP: {self._safe_health(self.opponent_player.hero)}, "
            f"Mana: {self.opponent_player.mana}/{self.opponent_player.max_mana}"
        )
        self._log(f"  Hand ({len(self.opponent_player.hand)})")
        if self.opponent_player.field:
            self._log(f"  Field: {', '.join([f'{str(m)}[{m.atk}/{m.health}]' for m in self.opponent_player.field])}")
