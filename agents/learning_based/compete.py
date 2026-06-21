"""
Comprehensive Dhumbal (Jhyap) Card Game Evaluation with PPO vs DQN Agents
======================================================================

This script evaluates pre-trained PPO and DQN agents in the Dhumbal card game over 2000 rounds.
The state encoding produces a 117-dimensional vector, matching the trained models' input size.
Game logic is corrected to ensure proper Jhyap calls, scoring, round progression, and robust action handling.

Game Rules:
- 2 players, each dealt 5 cards from a standard 52-card deck
- Goal: Achieve lowest hand value (≤ 10 points) to call "Jhyap"
- Card values: A=1, 2-10=face value, J=11, Q=12, K=13
- Valid discards: Single cards, same-rank sets (2+ cards), consecutive same-suit sequences (3+ cards)
- Turn: Optional Jhyap call, then discard, then pick from discard pile or deck
- Scoring: 
  - Successful Jhyap: Winner gains coins equal to opponents' hand values (capped at 100)
  - Failed Jhyap: Caller pays sum of all hand values (capped at 100); lowest non-caller wins
  - No Jhyap: Round ends on deck exhaustion or max turns (100); lowest hand value wins
- Round ends: On Jhyap call, deck exhaustion, or max turns (100)
- Tie handling: Caller wins only if uniquely lowest; otherwise, lowest non-caller wins

Agent Features:
- PPO: Actor-critic with architecture (117-128-64-128) for policy and (117-128-64-1) for value
- DQN: Q-network with target network (117-128-64-128) for action values
- State encoding: 117 dimensions (hand: 52, discard top: 52, player one-hot: 2, features: 7, phase: 3, padding: 1)
- Action space: 128 discrete actions (padded)

Evaluation:
- 2000 rounds with metrics: win rates, coin changes, Jhyap success, decision times
- Statistical analysis: Cohen's d, t-tests with Bonferroni correction
- Output: CSV and JSON files with detailed results

Implementation Details:
- Python 3.12+ with TensorFlow, NumPy, SciPy, tqdm
- Fixed random seed (42) for reproducibility
- GPU acceleration with TensorFlow
- Robust error handling and logging
- Corrected game logic for Jhyap calls, scoring, round progression, and action validation
"""

import sys
import os
import tensorflow as tf
import numpy as np
import random
import itertools
import json
import logging
import csv
import time
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from tensorflow.keras import models, layers
from scipy.stats import ttest_ind
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from agents.rule_based.rule_based_agent import RuleBasedAI, AIStyle as RuleBasedAIStyle

# Configure TensorFlow for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Game constants
NUM_PLAYERS = 4
MAX_PLAYERS = 5
MIN_PLAYERS = 2
HAND_SIZE = 5
JHYAP_THRESHOLD = 10
STARTING_COINS = 10000
MAX_TURNS = 100
MIN_DISCARD_PILE_SIZE = 2
MAX_PAYMENT = 100
NUM_ROUNDS = 1024
MAX_ACTION_SIZE = 128
STATE_SIZE = 128

class AIStyle(Enum):
    PPO = "ppo"
    DQN = "dqn"

class RuleBasedAdapter:
    def __init__(self, rule_based_agent):
        self.agent = rule_based_agent
        self.name = rule_based_agent.name
        self.model_type = 'rule_based'
        self.decision_times = []
        self.player_id = rule_based_agent.player_id
        
    def act(self, state: np.ndarray, env: 'DhumbalEnv') -> int:
        start_time = time.time()
        hand = env.state.hands[env.state.current_player]
        phase = env.state.phase
        
        if phase == 'call':
            decision = self.agent.should_call_jhyap(hand, env.state, env.game)
            action = decision
        elif phase == 'discard':
            discard = self.agent.choose_discard(hand, env.state, env.game)
            if not env.game.validate_discard(discard) and hand:
                discard = [max(hand, key=lambda x: x.value)] if hand else []
            action = discard
        elif phase == 'pick':
            top_discard = env.state.discard_pile[-1] if env.state.discard_pile else None
            should_pick, _ = self.agent.should_pick_from_discard([top_discard] if top_discard else [], hand, env.state)
            if should_pick and top_discard:
                action = 'discard'
            else:
                action = 'deck'
                
        self.decision_times.append(time.time() - start_time)
        return env.action_to_index(action)

class RandomAdapter:
    def __init__(self, player_id):
        self.player_id = player_id
        self.name = f"Random_{player_id}"
        self.model_type = 'random'
        self.decision_times = []
        
    def act(self, state: np.ndarray, env: 'DhumbalEnv') -> int:
        start_time = time.time()
        valid_actions = env.get_actions()
        action = random.choice(valid_actions) if valid_actions else None
        
        if not action and env.state.phase == 'call':
            action = False
        elif not action and env.state.phase == 'discard':
            hand = env.state.hands[env.state.current_player]
            action = [hand[0]] if hand else []
        elif not action:
            action = 'deck'
            
        self.decision_times.append(time.time() - start_time)
        return env.action_to_index(action)

@dataclass
class GameState:
    round_number: int
    current_player: int
    hands: List[List['Card']]
    discard_pile: List['Card']
    deck_size: int
    player_coins: List[int]
    turn_count: int
    phase: str
    done: bool = False
    winner: Optional[int] = None

    def copy(self) -> 'GameState':
        return GameState(
            round_number=self.round_number,
            current_player=self.current_player,
            hands=[hand[:] for hand in self.hands],
            discard_pile=self.discard_pile[:],
            deck_size=self.deck_size,
            player_coins=self.player_coins.copy(),
            turn_count=self.turn_count,
            phase=self.phase,
            done=self.done,
            winner=self.winner
        )

    def to_dict(self) -> dict:
        return {
            'round_number': self.round_number,
            'current_player': self.current_player,
            'hands': [[str(card) for card in hand] for hand in self.hands],
            'discard_pile': [str(card) for card in self.discard_pile],
            'deck_size': self.deck_size,
            'player_coins': self.player_coins,
            'turn_count': self.turn_count,
            'phase': self.phase,
            'done': self.done,
            'winner': self.winner
        }

@dataclass
class RoundResult:
    round_number: int
    caller: int
    winner: int
    hand_values: List[int]
    coin_changes: List[int]
    final_coins: List[int]
    turns_played: int
    successful_call: bool
    hands: List[List['Card']]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'round_number': self.round_number,
            'caller': self.caller,
            'winner': self.winner,
            'hand_values': self.hand_values,
            'coin_changes': self.coin_changes,
            'final_coins': self.final_coins,
            'turns_played': self.turns_played,
            'successful_call': self.successful_call,
            'hands': [[str(card) for card in hand] for hand in self.hands]
        }

class Card:
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.value = self._calculate_value()

    def _calculate_value(self) -> int:
        if self.rank == 'A': return 1
        elif self.rank == 'J': return 11
        elif self.rank == 'Q': return 12
        elif self.rank == 'K': return 13
        return int(self.rank)

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Card): return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self) -> int:
        return hash((self.suit, self.rank))

class DhumbalGame:
    def __init__(self, num_players: int = NUM_PLAYERS):
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(f"Dhumbal requires {MIN_PLAYERS}-{MAX_PLAYERS} players")
        self.num_players = num_players
        self.player_coins = [STARTING_COINS] * num_players
        self.round_number = 0
        self.game_history: List[RoundResult] = []
        self.SUITS = ['♠', '♥', '♦', '♣']
        self.RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.RANK_ORDER = {rank: i for i, rank in enumerate(self.RANKS)}

    def create_full_deck(self) -> List[Card]:
        return [Card(suit, rank) for suit in self.SUITS for rank in self.RANKS]

    def create_deck(self) -> List[Card]:
        deck = self.create_full_deck()
        random.shuffle(deck)
        return deck

    def deal_cards(self) -> Tuple[List[List[Card]], List[Card]]:
        deck = self.create_deck()
        hands = [[] for _ in range(self.num_players)]
        for _ in range(HAND_SIZE):
            for player in range(self.num_players):
                if deck:
                    hands[player].append(deck.pop())
        return hands, deck

    def calculate_hand_value(self, hand: List[Card]) -> int:
        return sum(card.value for card in hand)

    def can_call_jhyap(self, hand: List[Card]) -> bool:
        return self.calculate_hand_value(hand) <= JHYAP_THRESHOLD

    def validate_same_rank_set(self, cards: List[Card]) -> bool:
        if len(cards) < 2: return False
        return all(card.rank == cards[0].rank for card in cards)

    def validate_sequence(self, cards: List[Card]) -> bool:
        if len(cards) < 3: return False
        if not all(card.suit == cards[0].suit for card in cards): return False
        positions = sorted(self.RANK_ORDER[card.rank] for card in cards)
        return all(positions[i] == positions[i-1] + 1 for i in range(1, len(positions)))

    def validate_discard(self, cards: List[Card]) -> bool:
        if not cards: return False
        if len(cards) == 1: return True
        return self.validate_same_rank_set(cards) or self.validate_sequence(cards)

    def get_active_players(self) -> List[int]:
        return [i for i, coins in enumerate(self.player_coins) if coins > 0]

    def is_game_over(self) -> bool:
        return len(self.get_active_players()) < MIN_PLAYERS

class DhumbalEnv:
    def __init__(self, game: DhumbalGame):
        self.game = game
        self.hands, self.deck = game.deal_cards()
        self.discard_pile = [self.deck.pop()] if self.deck else []
        self.state = GameState(
            round_number=game.round_number + 1,
            current_player=0,
            hands=[hand[:] for hand in self.hands],
            discard_pile=self.discard_pile[:],
            deck_size=len(self.deck),
            player_coins=game.player_coins.copy(),
            turn_count=0,
            phase='call'
        )
        self.action_cache: Dict[int, List[Any]] = {}

    def reset(self) -> np.ndarray:
        self.hands, self.deck = self.game.deal_cards()
        self.discard_pile = [self.deck.pop()] if self.deck else []
        self.state = GameState(
            round_number=self.game.round_number + 1,
            current_player=0,
            hands=[hand[:] for hand in self.hands],
            discard_pile=self.discard_pile[:],
            deck_size=len(self.deck),
            player_coins=game.player_coins.copy(),
            turn_count=0,
            phase='call'
        )
        self.action_cache.clear()
        return self.encode_state()

    def set_state(self, new_state: GameState):
        self.state = new_state.copy()
        self.hands = [hand[:] for hand in new_state.hands]
        self.discard_pile = new_state.discard_pile[:]
        current_deck_size = len(self.deck)
        target_deck_size = new_state.deck_size
        if current_deck_size < target_deck_size:
            self.deck.extend([Card(random.choice(self.game.SUITS), random.choice(self.game.RANKS))
                             for _ in range(target_deck_size - current_deck_size)])
        elif current_deck_size > target_deck_size:
            self.deck = self.deck[:target_deck_size]

    def encode_state(self) -> np.ndarray:
        hand = self.state.hands[self.state.current_player]
        hand_encoding = np.zeros(52)
        for card in hand:
            suit_idx = self.game.SUITS.index(card.suit)
            rank_idx = self.game.RANKS.index(card.rank)
            card_idx = suit_idx * 13 + rank_idx
            hand_encoding[card_idx] = 1
        discard_encoding = np.zeros(52)
        if self.state.discard_pile:
            top_card = self.state.discard_pile[-1]
            suit_idx = self.game.SUITS.index(top_card.suit)
            rank_idx = self.game.RANKS.index(top_card.rank)
            discard_encoding[suit_idx * 13 + rank_idx] = 1
            
        player_one_hot = np.zeros(self.game.num_players)
        player_one_hot[self.state.current_player] = 1
        
        hand_sizes_norm = [len(self.state.hands[i]) / HAND_SIZE for i in range(self.game.num_players)]
        # Held constant to match the training-time encoding (see ppo.py / dqn.py):
        # coins and the cross-round counter are invariant within a round, so feeding
        # their varying evaluation-time values would push the policy out of distribution.
        coins_norm = [1.0 for _ in range(self.game.num_players)]

        hand_value = self.game.calculate_hand_value(hand) / (13 * HAND_SIZE)
        turn_norm = self.state.turn_count / MAX_TURNS
        discard_pile_size = len(self.state.discard_pile) / 52
        game_progress = 0.0
        
        phase_one_hot = np.zeros(3)
        phase_map = {'call': 0, 'discard': 1, 'pick': 2}
        if self.state.phase in phase_map:
            phase_one_hot[phase_map[self.state.phase]] = 1
            
        padding = np.zeros(5)
        
        state = np.concatenate([
            hand_encoding,       # 52
            discard_encoding,    # 52
            player_one_hot,      # 4
            hand_sizes_norm,     # 4
            coins_norm,          # 4
            [hand_value, turn_norm, discard_pile_size, game_progress], # 4
            phase_one_hot,       # 3
            padding              # 5
        ])
        assert len(state) == STATE_SIZE, f"State size is {len(state)}, expected {STATE_SIZE}"
        return state

    def get_actions(self) -> List[Any]:
        # IMPORTANT: do NOT cache on id(self.state). GameState objects are short-lived
        # (a fresh copy is created every step), and CPython reuses the ids of freed
        # objects, so an id()-keyed cache returns the legal actions of a *different*
        # prior state ~90% of the time. That corrupts discard decoding (valid discards
        # map to cards no longer in hand -> rejected -> skipped), so hands never shrink
        # and Jhyap is essentially never callable. Recomputing is cheap and correct.
        actions = []
        if self.state.phase == 'call':
            actions = [True, False] if self.game.can_call_jhyap(self.state.hands[self.state.current_player]) else [False]
        elif self.state.phase == 'discard':
            hand = self.state.hands[self.state.current_player]
            if not hand:
                actions = []
            else:
                actions = [[card] for card in hand]
                rank_groups = defaultdict(list)
                for card in hand:
                    rank_groups[card.rank].append(card)
                for cards in rank_groups.values():
                    if len(cards) >= 2:
                        for size in range(2, len(cards) + 1):
                            actions.extend(list(combo) for combo in itertools.combinations(cards, size))
                suit_groups = defaultdict(list)
                for card in hand:
                    suit_groups[card.suit].append(card)
                for suit, cards in suit_groups.items():
                    if len(cards) >= 3:
                        cards_sorted = sorted(cards, key=lambda x: self.game.RANK_ORDER[x.rank])
                        for size in range(3, len(cards_sorted) + 1):
                            for start in range(len(cards_sorted) - size + 1):
                                combo_list = cards_sorted[start:start + size]
                                if self.game.validate_sequence(combo_list):
                                    actions.append(combo_list)
        elif self.state.phase == 'pick':
            actions = ['deck']
            if self.state.discard_pile:
                actions.append('discard')
        return actions

    def action_to_index(self, action) -> int:
        actions = self.get_actions()
        if not actions:
            return 0
        def action_key(act):
            if isinstance(act, bool):
                return str(act)
            elif isinstance(act, str):
                return act
            else:
                return tuple(sorted(str(c) for c in act))
        sorted_actions = sorted(actions, key=action_key)
        key = action_key(action)
        for idx, act in enumerate(sorted_actions):
            if action_key(act) == key:
                return idx
        return 0

    def index_to_action(self, index: int):
        actions = self.get_actions()
        if not actions:
            if self.state.phase == 'call':
                return False
            elif self.state.phase == 'discard':
                hand = self.state.hands[self.state.current_player]
                return [hand[0]] if hand else []
            else:
                return 'deck'
        # Must mirror action_to_index() (and the training-time index_to_action in
        # ppo.py / dqn.py): sort legal actions by key, then map the raw network index
        # modulo the number of legal actions. Sorting + modulo here is what keeps the
        # learned index->action mapping identical between training and evaluation.
        def action_key(act):
            if isinstance(act, bool):
                return str(act)
            elif isinstance(act, str):
                return act
            else:
                return tuple(sorted(str(c) for c in act))
        sorted_actions = sorted(actions, key=action_key)
        return sorted_actions[index % len(sorted_actions)]

    def get_action_space_size(self) -> int:
        return MAX_ACTION_SIZE

    def step(self, action, player: int) -> Tuple[np.ndarray, float, bool, dict]:
        new_state = self.state.copy()
        reward = 0.0
        done = False
        log_entry = {
            'turn': new_state.turn_count,
            'player': player,
            'phase': new_state.phase,
            'hand_size': len(new_state.hands[player]),
            'hand_value': self.game.calculate_hand_value(new_state.hands[player]),
            'action': str(action) if isinstance(action, list) else action,
            'state_before': new_state.to_dict()
        }

        old_coins = new_state.player_coins.copy()

        if new_state.phase == 'call':
            if isinstance(action, bool):
                if action and self.game.can_call_jhyap(new_state.hands[player]):
                    logger.debug(f"Player {player} called Jhyap with hand value {self.game.calculate_hand_value(new_state.hands[player])}")
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    caller = player
                    min_value = min(hand_values)
                    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                    successful_call = len(min_value_players) == 1 and min_value_players[0] == caller
                    if successful_call:
                        winner = caller
                        total = 0
                        for i in range(self.game.num_players):
                            if i != winner:
                                payment = min(hand_values[i], MAX_PAYMENT)
                                new_state.player_coins[i] -= payment
                                total += payment
                        new_state.player_coins[winner] += total
                        reward = total if player == winner else -min(hand_values[player], MAX_PAYMENT)
                        logger.debug(f"Successful Jhyap by Player {winner}: Gained {total} coins")
                    else:
                        non_caller_min = [i for i in min_value_players if i != caller]
                        winner = non_caller_min[0] if non_caller_min else min_value_players[0]
                        total = sum(min(v, MAX_PAYMENT) for v in hand_values)
                        new_state.player_coins[caller] -= total
                        new_state.player_coins[winner] += total
                        reward = -total if player == caller else (total if player == winner else -min(hand_values[player], MAX_PAYMENT))
                        logger.debug(f"Failed Jhyap by Player {caller}: Paid {total} coins; Winner: Player {winner}")
                    new_state.done = True
                    new_state.winner = winner
                else:
                    new_state.phase = 'discard'
                    reward = 0.0
                    logger.debug(f"Player {player} chose not to call Jhyap or invalid call")
            else:
                reward = -10.0
                logger.debug(f"Invalid action in call phase by Player {player}: Expected bool, got {type(action)}")
                new_state.phase = 'discard'

        elif new_state.phase == 'discard':
            valid = False
            hand = new_state.hands[player]
            if isinstance(action, list) and all(isinstance(card, Card) for card in action):
                if all(card in hand for card in action) and self.game.validate_discard(action):
                    valid = True
                    for card in action:
                        new_state.hands[player].remove(card)
                    new_state.discard_pile.extend(action)
                    new_state.phase = 'pick'
                    reward = 1.0
                    logger.debug(f"Player {player} discarded {action}")
            if not valid:
                reward = -10.0
                logger.debug(f"Invalid discard by Player {player}: {action}, Hand: {[str(c) for c in hand]}")
                valid_actions = self.get_actions()
                if valid_actions and hand:
                    action = random.choice(valid_actions)
                    if isinstance(action, list) and all(card in hand for card in action):
                        for card in action:
                            new_state.hands[player].remove(card)
                        new_state.discard_pile.extend(action)
                        new_state.phase = 'pick'
                        reward += 1.0
                        logger.debug(f"Corrected to valid discard: {action}")
                    else:
                        logger.debug(f"No valid discard action, skipping to pick")
                        new_state.phase = 'pick'
                else:
                    logger.debug(f"No valid discards available for Player {player}, skipping to pick")
                    new_state.phase = 'pick'

        elif new_state.phase == 'pick':
            if len(new_state.hands[player]) >= HAND_SIZE:
                reward = -10.0
                logger.debug(f"Player {player} cannot pick: Hand full")
                new_state.phase = 'call'
                new_state.current_player = (new_state.current_player + 1) % self.game.num_players
                new_state.turn_count += 1
            elif isinstance(action, str) and action == 'discard' and new_state.discard_pile:
                card = new_state.discard_pile.pop()
                new_state.hands[player].append(card)
                new_state.deck_size = len(self.deck)
                reward = 1.0
                logger.debug(f"Player {player} picked {card} from discard pile")
                new_state.phase = 'call'
                new_state.current_player = (new_state.current_player + 1) % self.game.num_players
                new_state.turn_count += 1
            elif isinstance(action, str) and action == 'deck':
                if not self.deck and len(new_state.discard_pile) >= MIN_DISCARD_PILE_SIZE:
                    top = new_state.discard_pile.pop() if new_state.discard_pile else None
                    random.shuffle(new_state.discard_pile)
                    self.deck.extend(new_state.discard_pile)
                    new_state.discard_pile = [top] if top else []
                    new_state.deck_size = len(self.deck)
                if self.deck:
                    card = self.deck.pop()
                    new_state.hands[player].append(card)
                    new_state.deck_size = len(self.deck)
                    reward = 1.0
                    logger.debug(f"Player {player} picked {card} from deck")
                    new_state.phase = 'call'
                    new_state.current_player = (new_state.current_player + 1) % self.game.num_players
                    new_state.turn_count += 1
                else:
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    min_value = min(hand_values)
                    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                    winner = min_value_players[0]
                    new_state.winner = winner
                    total = 0
                    for i in range(self.game.num_players):
                        if i != winner:
                            payment = min(hand_values[i], MAX_PAYMENT)
                            new_state.player_coins[i] -= payment
                            total += payment
                    new_state.player_coins[winner] += total
                    reward = total if player == winner else -min(hand_values[player], MAX_PAYMENT)
                    new_state.done = True
                    logger.debug(f"Deck empty: Winner Player {winner} gained {total} coins")
            else:
                reward = -10.0
                logger.debug(f"Invalid pick by Player {player}: {action}, Hand: {[str(c) for c in new_state.hands[player]]}")
                valid_actions = self.get_actions()
                if valid_actions:
                    action = random.choice(valid_actions)
                    if action == 'discard' and new_state.discard_pile:
                        card = new_state.discard_pile.pop()
                        new_state.hands[player].append(card)
                        new_state.deck_size = len(self.deck)
                        reward += 1.0
                        logger.debug(f"Corrected to valid pick from discard: {card}")
                    elif action == 'deck' and self.deck:
                        card = self.deck.pop()
                        new_state.hands[player].append(card)
                        new_state.deck_size = len(self.deck)
                        reward += 1.0
                        logger.debug(f"Corrected to valid pick from deck: {card}")
                    new_state.phase = 'call'
                    new_state.current_player = (new_state.current_player + 1) % self.game.num_players
                    new_state.turn_count += 1
                else:
                    new_state.phase = 'call'
                    new_state.current_player = (new_state.current_player + 1) % self.game.num_players
                    new_state.turn_count += 1
                    logger.debug(f"No valid picks available for Player {player}, skipping to next player")

        if new_state.turn_count >= MAX_TURNS and not new_state.done:
            hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
            min_value = min(hand_values)
            min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
            winner = min_value_players[0]
            new_state.winner = winner
            total = 0
            for i in range(self.game.num_players):
                if i != winner:
                    payment = min(hand_values[i], MAX_PAYMENT)
                    new_state.player_coins[i] -= payment
                    total += payment
            new_state.player_coins[winner] += total
            reward = total if player == winner else -min(hand_values[player], MAX_PAYMENT)
            new_state.done = True
            logger.debug(f"Max turns reached: Winner Player {winner} gained {total} coins")

        self.set_state(new_state)
        # BUGFIX: return new_state.done (the local `done` was never updated from the
        # branches that set new_state.done=True, so step() always returned done=False).
        # That made simulate_round skip its `if done:` block, leaving successful_call
        # permanently False (Jhyap success = 0%) even when the caller won.
        done = new_state.done
        log_entry['reward'] = reward
        log_entry['done'] = done
        log_entry['state_after'] = new_state.to_dict()
        coin_changes = [new - old for new, old in zip(new_state.player_coins, old_coins)]
        log_entry['coin_changes'] = coin_changes
        return self.encode_state(), reward, done, log_entry

class PPO:
    def __init__(self, state_size: int, max_action_size: int):
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        logger.debug(f"PPO actor input shape: {self.actor.input_shape}, output shape: {self.actor.output_shape}")
        logger.debug(f"PPO critic input shape: {self.critic.input_shape}, output shape: {self.critic.output_shape}")

    def build_actor(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.max_action_size, activation='softmax')(x)
        return models.Model(inputs, outputs)

    def build_critic(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        return models.Model(inputs, outputs)

    def act(self, state: np.ndarray, env: DhumbalEnv) -> Tuple[int, float]:
        # Sample a raw index over the full action head (128), exactly as in training
        # (ppo.py PPO.act). index_to_action() folds the raw index back onto the legal
        # actions via sorting + modulo, so the policy is decoded identically here.
        action_space_size = env.get_action_space_size()
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        with tf.device('/GPU:0'):
            probs = self.actor(state, training=False)[0].numpy()
        probs = probs[:action_space_size]
        probs /= np.sum(probs + 1e-10)
        action = np.random.choice(action_space_size, p=probs)
        return action, probs[action]

class DQN:
    def __init__(self, state_size: int, max_action_size: int):
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.model = self.build_model()
        self.target_model = self.build_model()
        logger.debug(f"DQN model input shape: {self.model.input_shape}, output shape: {self.model.output_shape}")
        logger.debug(f"DQN target model input shape: {self.target_model.input_shape}, output shape: {self.target_model.output_shape}")

    def build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.max_action_size, activation='linear')(x)
        return models.Model(inputs, outputs)

    def act(self, state: np.ndarray, env: DhumbalEnv) -> int:
        action_space_size = len(env.get_actions())
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        with tf.device('/GPU:0'):
            q_values = self.model(state, training=False)[0].numpy()
        q_values = q_values[:action_space_size]
        action = np.argmax(q_values)
        return action

class LearningBasedAI:
    def __init__(self, player_id: int, state_size: int, max_action_size: int, model_type: str):
        self.player_id = player_id
        self.name = f"AI_{model_type}_{player_id}"
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.model_type = model_type
        if model_type == 'ppo':
            self.agent = PPO(state_size, max_action_size)
            self.load_ppo_models()
        elif model_type == 'dqn':
            self.agent = DQN(state_size, max_action_size)
            self.load_dqn_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.decision_times: List[float] = []

    def load_ppo_models(self):
        base_dir = os.path.dirname(__file__)
        ppo_actor_path = os.path.join(base_dir, 'ppo', 'ppo_actor_final.weights.h5')
        ppo_critic_path = os.path.join(base_dir, 'ppo', 'ppo_critic_final.weights.h5')
        try:
            self.agent.actor.load_weights(ppo_actor_path)
            self.agent.critic.load_weights(ppo_critic_path)
            logger.info(f"Loaded PPO models: {ppo_actor_path}, {ppo_critic_path}")
        except Exception as e:
            logger.error(f"Failed to load PPO models: {e}")
            raise

    def load_dqn_model(self):
        base_dir = os.path.dirname(__file__)
        dqn_model_path = os.path.join(base_dir, 'dqn', 'dqn_model_ep_final.weights.h5')
        dqn_target_path = os.path.join(base_dir, 'dqn', 'dqn_target_model_ep_final.weights.h5')
        try:
            self.agent.model.load_weights(dqn_model_path)
            self.agent.target_model.load_weights(dqn_target_path)
            logger.info(f"Loaded DQN models: {dqn_model_path}, {dqn_target_path}")
        except Exception as e:
            logger.error(f"Failed to load DQN models: {e}")
            raise

def simulate_round(game: DhumbalGame, players: List[Any], env: DhumbalEnv, verbose: bool = False, debug: bool = False) -> RoundResult:
    if debug:
        logger.setLevel(logging.DEBUG)
    
    seating_order = list(range(game.num_players))
    random.shuffle(seating_order)
    seated_players = [players[i] for i in seating_order]
    
    original_coins = game.player_coins.copy()
    game.player_coins = [original_coins[i] for i in seating_order]
    
    game.round_number += 1
    env.reset()
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {game.round_number}")
        logger.info(f"{'='*60}")
        logger.info(f"Initial discard: {env.discard_pile[0] if env.discard_pile else 'None'}")
        for i in range(game.num_players):
            hand_str = [str(card) for card in env.hands[i]]
            value = game.calculate_hand_value(env.hands[i])
            logger.info(f"Seat {i} (Player {seating_order[i]}: {seated_players[i].name}): {hand_str} (value: {value})")
            
    caller = -1
    successful_call = False
    while env.state.turn_count < MAX_TURNS and not env.state.done:
        current_seat = env.state.current_player
        player = seated_players[current_seat]
        start_time = time.time()
        state = env.encode_state()
        valid_actions = env.get_actions()
        if not valid_actions:
            logger.debug(f"No valid actions for Seat {current_seat} in phase {env.state.phase}")
            if env.state.phase == 'call':
                action = False
            elif env.state.phase == 'discard':
                action = [env.state.hands[current_seat][0]] if env.state.hands[current_seat] else []
            else:
                action = 'deck'
        else:
            if player.model_type == 'ppo':
                action_idx, _ = player.agent.act(state, env)
            elif player.model_type == 'dqn':
                action_idx = player.agent.act(state, env)
            else:
                action_idx = player.act(state, env)
            action = env.index_to_action(action_idx)
        _, reward, done, log = env.step(action, current_seat)
        player.decision_times.append(time.time() - start_time)
        if verbose:
            logger.info(f"Turn {env.state.turn_count}, Seat {current_seat} ({player.name}), Phase {env.state.phase}, Action {action}, Reward {reward}")
        if env.state.phase == 'call' and isinstance(action, bool) and action:
            caller = current_seat
        if done:
            successful_call = env.state.winner == caller if caller != -1 else False
            break
            
    if not env.state.done:
        hand_values = [game.calculate_hand_value(hand) for hand in env.state.hands]
        min_value = min(hand_values)
        min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
        winner = min_value_players[0]
        env.state.winner = winner
        total = 0
        for i in range(game.num_players):
            if i != winner:
                payment = min(hand_values[i], MAX_PAYMENT)
                env.state.player_coins[i] -= payment
                total += payment
        env.state.player_coins[winner] += total
        env.state.done = True
        logger.debug(f"Forced end: Winner Seat {winner} gained {total} coins")
        
    seat_hand_values = [game.calculate_hand_value(hand) for hand in env.hands]
    seat_winner = env.state.winner
    seat_caller = caller
    seat_coin_changes = [env.state.player_coins[i] - game.player_coins[i] for i in range(game.num_players)]
    
    inverse_seating = [0] * game.num_players
    for seat, original_id in enumerate(seating_order):
        inverse_seating[original_id] = seat
        
    game.player_coins = [env.state.player_coins[inverse_seating[i]] for i in range(game.num_players)]
    
    mapped_winner = seating_order[seat_winner] if seat_winner != -1 else -1
    mapped_caller = seating_order[seat_caller] if seat_caller != -1 else -1
    mapped_hand_values = [seat_hand_values[inverse_seating[i]] for i in range(game.num_players)]
    mapped_coin_changes = [seat_coin_changes[inverse_seating[i]] for i in range(game.num_players)]
    mapped_hands = [env.hands[inverse_seating[i]][:] for i in range(game.num_players)]
    
    result = RoundResult(
        round_number=game.round_number,
        caller=mapped_caller,
        winner=mapped_winner,
        hand_values=mapped_hand_values,
        coin_changes=mapped_coin_changes,
        final_coins=game.player_coins.copy(),
        turns_played=env.state.turn_count,
        successful_call=successful_call,
        hands=mapped_hands
    )
    game.game_history.append(result)
    if verbose:
        logger.info(f"Round {game.round_number} ended, Winner: {mapped_winner}, Successful Call: {successful_call}")
        logger.info(f"Hand values: {mapped_hand_values}")
        logger.info(f"Coin changes: {mapped_coin_changes}")
    return result

def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    pooled_std = np.sqrt(((len(group1)-1)*np.var(group1) + (len(group2)-1)*np.var(group2)) / (len(group1)+len(group2)-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0.0

def simulate_game(game: DhumbalGame, players: List[LearningBasedAI], max_rounds: int = NUM_ROUNDS, verbose: bool = True, debug: bool = False) -> Dict[str, Any]:
    env = DhumbalEnv(game)
    round_results = []
    for round_idx in tqdm(range(max_rounds), desc="Simulating rounds"):
        if game.is_game_over():
            logger.info(f"Game ended early due to bankruptcy after {round_idx} rounds")
            break
        try:
            result = simulate_round(game, players, env, verbose, debug)
            round_results.append(result)
            bankrupt_players = [i for i, coins in enumerate(game.player_coins) if coins <= 0]
            if bankrupt_players and verbose:
                for player in bankrupt_players:
                    logger.info(f"Player {player} ({players[player].name}) is bankrupt!")
        except Exception as e:
            logger.error(f"Error in round {game.round_number}: {e}")
            return {"error": f"Simulation failed at round {game.round_number}: {str(e)}"}
    return analyze_game_results(game, players, round_results, verbose)

def analyze_game_results(game: DhumbalGame, players: List[Any], round_results: List[RoundResult], verbose: bool = True) -> Dict[str, Any]:
    """Learning-based SELECTION analysis: aggregate per algorithm (e.g. 2xPPO vs 2xDQN).

    Each algorithm occupies several seats; metrics are pooled across that algorithm's
    seats and the two algorithms are compared head-to-head. The algorithm with the
    higher win rate (economic performance breaks ties) is selected for the championship.
    """
    if not round_results:
        return {"error": "No rounds completed"}
    n = len(round_results)

    # Group seats by algorithm, preserving first-seen order (dict is insertion-ordered).
    groups = {}
    for i, p in enumerate(players):
        groups.setdefault(p.model_type, []).append(i)
    group_names = list(groups.keys())

    # Per-round, per-algorithm series (pooled over that algorithm's seats).
    series = {g: {'win': [], 'econ': [], 'jhyap': []} for g in group_names}
    call_hand_values = {g: [] for g in group_names}   # caller's hand value when this group called
    call_success = {g: [] for g in group_names}       # 1/0 success for those calls
    for r in round_results:
        for g, idxs in groups.items():
            called = (r.caller in idxs)
            series[g]['win'].append(1 if r.winner in idxs else 0)
            series[g]['econ'].append(sum(r.coin_changes[i] for i in idxs))
            series[g]['jhyap'].append(1 if (called and r.successful_call) else 0)
            if called:
                call_hand_values[g].append(r.hand_values[r.caller])
                call_success[g].append(1 if r.successful_call else 0)

    perf = {}
    for g in group_names:
        wins = np.array(series[g]['win'], dtype=float)
        econ = np.array(series[g]['econ'], dtype=float)
        calls, succ = call_hand_values[g], call_success[g]
        calls_made, successes = len(calls), int(sum(call_success[g]))
        if len(calls) > 1 and np.std(calls) > 0 and np.std(succ) > 0:
            risk = float(np.corrcoef(calls, succ)[0, 1])
        else:
            risk = None
        dec_times = [t for i in groups[g] for t in players[i].decision_times]
        perf[g] = {
            'seats': groups[g],
            'win_rate': float(np.mean(wins) * 100),
            'win_ci': float(1.96 * np.std(wins) / np.sqrt(n) * 100),
            'economic': float(np.mean(econ)),
            'jhyap_calls': calls_made,
            'jhyap_successes': successes,
            'jhyap_success_rate': float(successes / calls_made * 100) if calls_made > 0 else 0.0,
            'risk_corr': risk,
            'avg_decision_ms': float(np.mean(dec_times) * 1000) if dec_times else 0.0,
        }

    # Head-to-head comparison (Cohen's d + Welch t-test) between the two algorithms.
    comparison = {}
    if len(group_names) == 2:
        ga, gb = group_names
        for metric in ('win', 'econ', 'jhyap'):
            a, b = series[ga][metric], series[gb][metric]
            try:
                _, pval = ttest_ind(a, b, equal_var=False)
                pval = float(pval)
            except Exception:
                pval = None
            comparison[metric] = {'cohens_d': calculate_cohens_d(a, b), 'p_value': pval}

    # Selection: higher win rate, economic performance breaks ties.
    ranked = sorted(group_names, key=lambda g: (perf[g]['win_rate'], perf[g]['economic']), reverse=True)
    selected = ranked[0]
    avg_turns = float(np.mean([r.turns_played for r in round_results]))
    avg_winning_hand = float(np.mean([min(r.hand_values) for r in round_results]))
    total_calls = sum(perf[g]['jhyap_calls'] for g in group_names)
    total_succ = sum(perf[g]['jhyap_successes'] for g in group_names)
    per_seat = {players[i].name: {
        'win_rate': float(np.mean([1 if r.winner == i else 0 for r in round_results]) * 100),
        'economic': float(np.mean([r.coin_changes[i] for r in round_results])),
    } for i in range(game.num_players)}
    results = {
        'tournament': '2xPPO vs 2xDQN (4-player mirror match, seat-randomised)',
        'total_rounds': n,
        'selected_agent': selected.upper(),
        'algorithm_performance': {g.upper(): perf[g] for g in group_names},
        'head_to_head': ({f'{group_names[0].upper()}_vs_{group_names[1].upper()}': comparison}
                         if comparison else {}),
        'per_seat': per_seat,
        'game_statistics': {
            'avg_turns_per_round': round(avg_turns, 1),
            'avg_winning_hand_value': round(avg_winning_hand, 1),
            'total_jhyap_calls': total_calls,
            'total_successful_calls': total_succ,
        },
        'round_details': [r.to_dict() for r in round_results],
    }
    base_dir = os.path.dirname(__file__)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_metrics_path = os.path.join(base_dir, f'learning_selection_metrics_{n}_{timestamp}.csv')
    csv_cohens_path = os.path.join(base_dir, f'learning_selection_cohens_d_{n}_{timestamp}.csv')
    json_path = os.path.join(base_dir, f'learning_selection_results_{n}_{timestamp}.json')
    with open(csv_metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Seats', 'Win Rate (%)', 'Win CI (%)', 'Economic Perf.',
                         'Jhyap Calls', 'Jhyap Success (%)', 'Risk Corr.', 'Avg Dec. Time (ms)'])
        for g in group_names:
            pf = perf[g]
            writer.writerow([
                g.upper(), len(pf['seats']),
                f"{pf['win_rate']:.1f}", f"{pf['win_ci']:.1f}", f"{pf['economic']:.1f}",
                pf['jhyap_calls'], f"{pf['jhyap_success_rate']:.1f}",
                (f"{pf['risk_corr']:.3f}" if pf['risk_corr'] is not None else 'N/A'),
                f"{pf['avg_decision_ms']:.2f}",
            ])
    if comparison:
        with open(csv_cohens_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Comparison', 'Win d', 'Win p', 'Economic d', 'Economic p', 'Jhyap d', 'Jhyap p'])
            row = [f'{group_names[0].upper()} vs {group_names[1].upper()}']
            for metric in ('win', 'econ', 'jhyap'):
                c = comparison[metric]
                row.append(f"{c['cohens_d']:.3f}")
                row.append(f"{c['p_value']:.4f}" if c['p_value'] is not None else 'N/A')
            writer.writerow(row)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    if verbose:
        logger.info(f"\n{'='*64}")
        logger.info(f"LEARNING-BASED SELECTION  ({n} rounds, 2xPPO vs 2xDQN)")
        logger.info(f"{'='*64}")
        for g in group_names:
            pf = perf[g]
            rc_str = f"{pf['risk_corr']:.3f}" if pf['risk_corr'] is not None else 'N/A'
            logger.info(f"  {g.upper()} (seats {pf['seats']}):")
            logger.info(f"    Win Rate        : {pf['win_rate']:.1f}% +/- {pf['win_ci']:.1f}%")
            logger.info(f"    Economic/Round  : {pf['economic']:.2f}")
            logger.info(f"    Jhyap Success   : {pf['jhyap_success_rate']:.1f}%  ({pf['jhyap_successes']}/{pf['jhyap_calls']})")
            logger.info(f"    Risk Corr.      : {rc_str}")
            logger.info(f"    Avg Dec. Time   : {pf['avg_decision_ms']:.2f} ms")
        if comparison:
            logger.info(f"\n  Head-to-head ({group_names[0].upper()} vs {group_names[1].upper()}):")
            for metric in ('win', 'econ', 'jhyap'):
                c = comparison[metric]
                pstr = f"{c['p_value']:.4g}" if c['p_value'] is not None else 'N/A'
                logger.info(f"    {metric:8s}: Cohen's d = {c['cohens_d']:+.3f},  p = {pstr}")
        logger.info(f"\n  Avg turns/round: {avg_turns:.1f}   Avg winning hand: {avg_winning_hand:.1f}")
        logger.info(f"  Jhyap calls overall: {total_succ}/{total_calls} successful "
                    f"({(total_succ / total_calls * 100 if total_calls else 0):.1f}%)")
        logger.info(f"\n  >>> SELECTED for championship: {selected.upper()} <<<")
        logger.info(f"\nFiles saved: {os.path.basename(csv_metrics_path)}, "
                    f"{os.path.basename(csv_cohens_path)}, {os.path.basename(json_path)}")
    return results

if __name__ == "__main__":
    game = DhumbalGame(num_players=NUM_PLAYERS)
    state_size = STATE_SIZE
    max_action_size = MAX_ACTION_SIZE
    try:
        # Learning-based SELECTION stage: PPO vs DQN, decided in the same 4-player
        # setting as the championship. Two instances of each algorithm fill the four
        # seats (a "mirror match"); with per-round seat randomisation and per-algorithm
        # aggregation this is a balanced, rule-based-free head-to-head. The winner
        # advances to the cross-category championship. No rule-based/random baselines
        # are used at the selection stage (they are not part of this comparison).
        players = [
            LearningBasedAI(0, state_size, max_action_size, model_type='ppo'),
            LearningBasedAI(1, state_size, max_action_size, model_type='ppo'),
            LearningBasedAI(2, state_size, max_action_size, model_type='dqn'),
            LearningBasedAI(3, state_size, max_action_size, model_type='dqn'),
        ]
        logger.info("🃏 DHUMBAL LEARNING-BASED SELECTION: 2xPPO vs 2xDQN (4-player mirror match)")
        logger.info("=" * 70)
        logger.info(f"Configuration: {NUM_ROUNDS} rounds, {game.num_players} players, seed=42")
        logger.info(f"State Size: {state_size}, Max Action Size: {max_action_size}")
        logger.info(f"Starting Coins/Player: {STARTING_COINS:,}")
        logger.info("\nAI Agents:")
        for player in players:
            logger.info(f"  • {player.name} ({player.model_type.upper()})")
        logger.info("\nStarting simulation...")
        results = simulate_game(game, players, max_rounds=NUM_ROUNDS, verbose=True, debug=False)
        logger.info("\n" + "=" * 70)
        logger.info("✅ SIMULATION COMPLETE")
        logger.info("=" * 70)
        if "error" not in results:
            logger.info(f"Selected agent for championship: {results['selected_agent']}")
            logger.info(f"Rounds Completed: {results['total_rounds']}")
        else:
            logger.error(f"Simulation failed: {results['error']}")
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        logger.error("Please ensure the model weight files exist in ./ppo/ and ./dqn/ directories.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")