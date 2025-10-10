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
NUM_PLAYERS = 2
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
STATE_SIZE = 117

class AIStyle(Enum):
    PPO = "ppo"
    DQN = "dqn"

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
        player_one_hot = np.zeros(NUM_PLAYERS)
        player_one_hot[self.state.current_player] = 1
        hand_value = self.game.calculate_hand_value(hand) / (13 * HAND_SIZE)
        turn_norm = self.state.turn_count / MAX_TURNS
        opp_hand_size = len(self.state.hands[1 - self.state.current_player]) / HAND_SIZE
        my_coins_norm = self.state.player_coins[self.state.current_player] / STARTING_COINS
        opp_coins_norm = self.state.player_coins[1 - self.state.current_player] / STARTING_COINS
        discard_pile_size = len(self.state.discard_pile) / 52
        game_progress = self.state.round_number / NUM_ROUNDS
        phase_one_hot = np.zeros(3)
        phase_map = {'call': 0, 'discard': 1, 'pick': 2}
        phase_one_hot[phase_map[self.state.phase]] = 1
        state = np.concatenate([
            hand_encoding,          # 52
            discard_encoding,       # 52
            player_one_hot,         # 2
            [hand_value, turn_norm, opp_hand_size, my_coins_norm, opp_coins_norm, discard_pile_size, game_progress],  # 7
            phase_one_hot,          # 3
            [0]                     # 1 (padding to reach 117)
        ])
        logger.debug(f"State components: hand=52, discard=52, "
                     f"player_one_hot=2, features=7, phase=3, padding=1, total={len(state)}")
        assert len(state) == STATE_SIZE, f"State size is {len(state)}, expected {STATE_SIZE}"
        return state

    def get_actions(self) -> List[Any]:
        state_hash = id(self.state)
        if state_hash in self.action_cache:
            return self.action_cache[state_hash]
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
        self.action_cache[state_hash] = actions
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
        return actions[min(index, len(actions) - 1)]

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
        action_space_size = len(env.get_actions())
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
        ppo_actor_path = './ppo/ppo_actor_final.weights.h5'
        ppo_critic_path = './ppo/ppo_critic_final.weights.h5'
        try:
            self.agent.actor.load_weights(ppo_actor_path)
            self.agent.critic.load_weights(ppo_critic_path)
            logger.info(f"Loaded PPO models: {ppo_actor_path}, {ppo_critic_path}")
        except Exception as e:
            logger.error(f"Failed to load PPO models: {e}")
            raise

    def load_dqn_model(self):
        dqn_model_path = './dqn/dqn_model_ep500.weights.h5'
        dqn_target_path = './dqn/dqn_target_model_ep500.weights.h5'
        try:
            self.agent.model.load_weights(dqn_model_path)
            self.agent.target_model.load_weights(dqn_target_path)
            logger.info(f"Loaded DQN models: {dqn_model_path}, {dqn_target_path}")
        except Exception as e:
            logger.error(f"Failed to load DQN models: {e}")
            raise

def simulate_round(game: DhumbalGame, players: List[LearningBasedAI], env: DhumbalEnv, verbose: bool = False, debug: bool = False) -> RoundResult:
    if debug:
        logger.setLevel(logging.DEBUG)
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
            logger.info(f"Player {i} ({players[i].name}): {hand_str} (value: {value})")
    caller = -1
    successful_call = False
    while env.state.turn_count < MAX_TURNS and not env.state.done:
        current_player = env.state.current_player
        player = players[current_player]
        start_time = time.time()
        state = env.encode_state()
        valid_actions = env.get_actions()
        if not valid_actions:
            logger.debug(f"No valid actions for Player {current_player} in phase {env.state.phase}")
            if env.state.phase == 'call':
                action = False
            elif env.state.phase == 'discard':
                action = [env.state.hands[current_player][0]] if env.state.hands[current_player] else []
            else:
                action = 'deck'
        else:
            if player.model_type == 'ppo':
                action_idx, _ = player.agent.act(state, env)
            else:
                action_idx = player.agent.act(state, env)
            action = env.index_to_action(action_idx)
        _, reward, done, log = env.step(action, current_player)
        player.decision_times.append(time.time() - start_time)
        if verbose:
            logger.info(f"Turn {env.state.turn_count}, Player {current_player} ({player.name}), Phase {env.state.phase}, Action {action}, Reward {reward}")
        if env.state.phase == 'call' and isinstance(action, bool) and action:
            caller = current_player
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
        logger.debug(f"Forced end: Winner Player {winner} gained {total} coins")
    hand_values = [game.calculate_hand_value(hand) for hand in env.hands]
    winner = env.state.winner
    coin_changes = [env.state.player_coins[i] - game.player_coins[i] for i in range(game.num_players)]
    game.player_coins = env.state.player_coins.copy()
    result = RoundResult(
        round_number=game.round_number,
        caller=caller,
        winner=winner,
        hand_values=hand_values,
        coin_changes=coin_changes,
        final_coins=game.player_coins.copy(),
        turns_played=env.state.turn_count,
        successful_call=successful_call,
        hands=[hand[:] for hand in env.hands]
    )
    game.game_history.append(result)
    if verbose:
        logger.info(f"Round {game.round_number} ended, Winner: {winner}, Successful Call: {successful_call}")
        logger.info(f"Hand values: {hand_values}")
        logger.info(f"Coin changes: {coin_changes}")
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

def analyze_game_results(game: DhumbalGame, players: List[LearningBasedAI], round_results: List[RoundResult], verbose: bool = True) -> Dict[str, Any]:
    if not round_results:
        return {"error": "No rounds completed"}
    total_rounds = len(round_results)
    final_coins = game.player_coins.copy()
    winner_id = max(range(game.num_players), key=lambda i: final_coins[i])
    winner_counts = Counter(r.winner for r in round_results)
    caller_counts = Counter(r.caller for r in round_results if r.caller != -1)
    successful_calls = [r for r in round_results if r.successful_call and r.caller != -1]
    success_rates = {}
    avg_decision_times = [np.mean(ai.decision_times) * 1000 if ai.decision_times else 0.0 for ai in players]
    win_data = [[] for _ in range(game.num_players)]
    economic_data = [[] for _ in range(game.num_players)]
    jhyap_data = [[] for _ in range(game.num_players)]
    risk_data = [[] for _ in range(game.num_players)]
    jhyap_calls = [[] for _ in range(game.num_players)]
    for r in round_results:
        for i in range(game.num_players):
            win_data[i].append(1 if r.winner == i else 0)
            economic_data[i].append(r.coin_changes[i])
            jhyap_data[i].append(1 if r.caller == i and r.caller != -1 else 0)
            if r.caller == i and r.caller != -1:
                jhyap_calls[i].append(r.hand_values[i])
                risk_data[i].append(1 if r.successful_call else 0)
    for i in range(game.num_players):
        calls_made = caller_counts.get(i, 0)
        successful = len([r for r in successful_calls if r.caller == i])
        success_rates[i] = (successful / calls_made * 100) if calls_made > 0 else 0
    avg_winning_hand = sum(min(r.hand_values) for r in round_results) / total_rounds if total_rounds > 0 else 0
    avg_turns_per_round = sum(r.turns_played for r in round_results) / total_rounds if total_rounds > 0 else 0
    total_coins_transferred = sum(sum(abs(change) for change in r.coin_changes) / 2 for r in round_results)
    win_rates = [winner_counts.get(i, 0) / total_rounds * 100 for i in range(game.num_players)] if total_rounds > 0 else [0] * game.num_players
    win_ci = [1.96 * np.std(win_data[i]) / np.sqrt(total_rounds) * 100 for i in range(game.num_players)] if total_rounds > 0 else [0] * game.num_players
    economic_performance = [sum(economic_data[i]) / total_rounds for i in range(game.num_players)] if total_rounds > 0 else [0] * game.num_players
    jhyap_success_rates = [success_rates[i] for i in range(game.num_players)]
    risk_assessment = []
    for i in range(game.num_players):
        calls = jhyap_calls[i]
        successes = risk_data[i]
        if len(calls) > 1:
            corr = np.corrcoef(calls, successes)[0, 1]
            risk_assessment.append(corr)
        else:
            risk_assessment.append(None)
    cohens_d = {}
    p_values = {}
    metrics = ['win', 'economic', 'jhyap']
    data_lists = [win_data, economic_data, jhyap_data]
    comparisons = [(0, 1)]
    adjusted_alpha = 0.05 / len(comparisons) if comparisons else 0.05
    for m, metric in enumerate(metrics):
        cohens_d[metric] = {}
        p_values[metric] = {}
        for pair in comparisons:
            i, j = pair
            comp_key = f'P{i} vs P{j}'
            cohens_d[metric][comp_key] = calculate_cohens_d(data_lists[m][i], data_lists[m][j])
            if len(data_lists[m][i]) >= 2 and len(data_lists[m][j]) >= 2:
                _, p_val = ttest_ind(data_lists[m][i], data_lists[m][j], equal_var=False)
                p_values[metric][comp_key] = p_val
            else:
                p_values[metric][comp_key] = None
    results = {
        'game_summary': {
            'total_rounds': total_rounds,
            'final_winner': winner_id,
            'winner_name': players[winner_id].name,
            'final_coins': final_coins,
            'starting_coins': STARTING_COINS
        },
        'player_performance': {
            'win_rates': win_rates,
            'win_ci': win_ci,
            'economic_performance': economic_performance,
            'jhyap_success_rates': jhyap_success_rates,
            'risk_assessment': risk_assessment,
            'avg_decision_times_ms': avg_decision_times
        },
        'game_statistics': {
            'avg_winning_hand_value': round(avg_winning_hand, 1),
            'avg_turns_per_round': round(avg_turns_per_round, 1),
            'successful_calls': len(successful_calls),
            'total_coins_transferred': int(total_coins_transferred)
        },
        'statistical_analysis': {
            'cohens_d': cohens_d,
            'p_values': p_values,
            'adjusted_alpha': adjusted_alpha
        },
        'round_details': [r.to_dict() for r in round_results]
    }
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_metrics_path = f'game_metrics_rounds_{total_rounds}_{timestamp}.csv'
    csv_cohens_path = f'game_cohens_d_rounds_{total_rounds}_{timestamp}.csv'
    json_path = f'game_results_rounds_{total_rounds}_{timestamp}.json'
    with open(csv_metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Player', 'Win Rate (%)', 'Win CI (%)', 'Economic Perf.', 'Jhyap Success (%)', 'Risk Corr.', 'Avg Dec. Time (ms)'])
        for i in range(game.num_players):
            risk_val = risk_assessment[i] if risk_assessment[i] is not None else 'N/A'
            writer.writerow([
                players[i].name,
                f"{win_rates[i]:.1f}",
                f"{win_ci[i]:.1f}",
                f"{economic_performance[i]:.1f}",
                f"{jhyap_success_rates[i]:.1f}",
                risk_val,
                f"{avg_decision_times[i]:.1f}"
            ])
    with open(csv_cohens_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Comparison']
        for metric in metrics:
            header.extend([f'{metric.capitalize()} d', f'{metric.capitalize()} p'])
        writer.writerow(header)
        for comp in comparisons:
            comp_key = f'P{comp[0]} vs P{comp[1]}'
            row = [comp_key]
            for metric in metrics:
                d = cohens_d[metric][comp_key]
                p = p_values[metric][comp_key] if p_values[metric][comp_key] is not None else 'N/A'
                row.extend([f"{d:.3f}", f"{p:.4f}" if isinstance(p, float) else p])
            writer.writerow(row)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"GAME ANALYSIS ({total_rounds} Rounds)")
        logger.info(f"{'='*60}")
        logger.info(f"🏆 Winner: {results['game_summary']['winner_name']}")
        logger.info(f"Total coins transferred: {total_coins_transferred:.0f}")
        logger.info("\nFinal Standings:")
        for rank, pid in enumerate(sorted(range(game.num_players), key=lambda i: final_coins[i], reverse=True), 1):
            change = final_coins[pid] - STARTING_COINS
            logger.info(f"  {rank}. {players[pid].name}: {final_coins[pid]:,} coins {f'(+{change:,})' if change > 0 else f'({change:,})'}")
        logger.info("\nPlayer Statistics:")
        for i in range(game.num_players):
            logger.info(f"  {players[i].name}:")
            logger.info(f"    • Win Rate: {win_rates[i]:.1f}% ± {win_ci[i]:.1f}%")
            logger.info(f"    • Avg. Coins/Round: {economic_performance[i]:.1f}")
            logger.info(f"    • Jhyap Success: {jhyap_success_rates[i]:.1f}%")
            risk_val = f"{risk_assessment[i]:.3f}" if risk_assessment[i] is not None else "N/A"
            logger.info(f"    • Risk Correlation: {risk_val}")
            logger.info(f"    • Avg. Decision Time: {avg_decision_times[i]:.1f} ms")
        logger.info("\nGame Statistics:")
        logger.info(f"  • Avg. Winning Hand: {avg_winning_hand:.1f} points")
        logger.info(f"  • Avg. Turns/Round: {avg_turns_per_round:.1f}")
        logger.info(f"  • Successful Calls: {len(successful_calls)} / {sum(caller_counts.values())} ({len(successful_calls)/sum(caller_counts.values())*100:.1f}% if sum(caller_counts.values()) > 0 else 0)")
        logger.info(f"\nFiles saved: {csv_metrics_path}, {csv_cohens_path}, {json_path}")
    return results

if __name__ == "__main__":
    game = DhumbalGame(num_players=NUM_PLAYERS)
    state_size = STATE_SIZE
    max_action_size = MAX_ACTION_SIZE
    try:
        ppo_player = LearningBasedAI(0, state_size, max_action_size, model_type='ppo')
        dqn_player = LearningBasedAI(1, state_size, max_action_size, model_type='dqn')
        players = [ppo_player, dqn_player]
        logger.info("🃏 DHUMBAL (Jhyap) LEARNING-BASED AI SIMULATION (PPO vs DQN)")
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
            logger.info(f"Overall Winner: {results['game_summary']['winner_name']}")
            logger.info(f"Rounds Completed: {results['game_summary']['total_rounds']}")
            logger.info(f"Avg. Winning Hand Value: {results['game_statistics']['avg_winning_hand_value']:.1f} points")
        else:
            logger.error(f"Simulation failed: {results['error']}")
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        logger.error("Please ensure the model weight files exist in ./ppo/ and ./dqn/ directories.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")