"""
Comprehensive Dhumbal (Jhyap) Card Game Evaluation with Hybrid Agents
======================================================================

This script evaluates hybrid agents (PPO_MCTS, DQN_MCTS, PPO_ISMCTS, DQN_ISMCTS) in the Dhumbal card game over 2000 rounds.
It adapts path management to match learning-based compete code, using ./ppo/ and ./dqn/ for models for outputs.
State encoding is fixed at 117 dimensions, and Jhyap success is improved through prioritized actions and logging.

Game Rules:
- 4 players, each dealt 5 cards from a 52-card deck
- Goal: Achieve lowest hand value (≤10) to call "Jhyap"
- Card values: A=1, 2-10=face value, J=11, Q=12, K=13
- Valid discards: Single cards
- Turn: Optional Jhyap call, then discard, then pick from discard pile or deck
- Scoring:
  - Successful Jhyap: Winner gains coins equal to opponents' hand values (capped at 100)
  - Failed Jhyap: Caller pays sum of all hand values (capped at 100); lowest non-caller wins
  - No Jhyap: Round ends on deck exhaustion or max turns (50); lowest hand value wins
- Tie handling: Caller wins only if uniquely lowest; otherwise, lowest non-caller wins

Agent Features:
- PPO_MCTS/PPO_ISMCTS: MCTS/ISMCTS with PPO priors and values
- DQN_MCTS/DQN_ISMCTS: MCTS/ISMCTS with DQN priors and Q-values
- State encoding: 117 dimensions (hand: 52, discard top: 52, player one-hot: 4, features: 5, phase: 3, padding: 1)
- Action space: 128 discrete actions (padded)

Evaluation:
- 2000 rounds with metrics: win rates, coin changes, Jhyap success, decision times
- Statistical analysis: Cohen's d, t-tests with Bonferroni correction
- Output: CSV and JSON files

Implementation Details:
- Python 3.12+ with TensorFlow, NumPy, SciPy, tqdm
- Fixed random seed (42) for reproducibility
- GPU acceleration with TensorFlow
- Robust path management and error handling
- MCTS/ISMCTS with UCB1 and shared models
"""
import time
import tensorflow as tf
import numpy as np
import random
import csv
import os
import json
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from tensorflow.keras import models, layers
from scipy.stats import ttest_ind
from datetime import datetime
import math

# Configure TensorFlow for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./tournament_error.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Game constants
NUM_PLAYERS = 4
MAX_PLAYERS = 4
MIN_PLAYERS = 2
HAND_SIZE = 5
JHYAP_THRESHOLD = 10
STARTING_COINS = 10000
MAX_TURNS = 50
MIN_DISCARD_PILE_SIZE = 2
MAX_PAYMENT = 100
EVALUATION_EPISODES = 2000
MCTS_ITERATIONS = 100
ISMCTS_SAMPLES = 5
UCB_C = 1.4
STATE_SIZE = 117
MAX_ACTION_SIZE = 128
NUM_ROUNDS = EVALUATION_EPISODES

class AIStyle(Enum):
    PPO_MCTS = "ppo_mcts"
    DQN_MCTS = "dqn_mcts"
    PPO_ISMCTS = "ppo_ismcts"
    DQN_ISMCTS = "dqn_ismcts"

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
            'hands': [[{'suit': card.suit, 'rank': card.rank} for card in hand] for hand in self.hands],
            'discard_pile': [{'suit': card.suit, 'rank': card.rank} for card in self.discard_pile],
            'deck_size': self.deck_size,
            'player_coins': self.player_coins,
            'turn_count': self.turn_count,
            'phase': self.phase,
            'done': self.done,
            'winner': self.winner
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GameState':
        hands = [[Card(card['suit'], card['rank']) for card in hand] for hand in data['hands']]
        discard_pile = [Card(card['suit'], card['rank']) for card in data['discard_pile']]
        return cls(
            round_number=data['round_number'],
            current_player=data['current_player'],
            hands=hands,
            discard_pile=discard_pile,
            deck_size=data['deck_size'],
            player_coins=data['player_coins'],
            turn_count=data['turn_count'],
            phase=data['phase'],
            done=data['done'],
            winner=data['winner']
        )

@dataclass
class RoundResult:
    round_number: int
    caller: int
    winner: int
    hand_values: List[int]
    coin_changes: List[int]
    successful_call: bool
    decision_times: List[float]

    def to_dict(self) -> dict:
        return {
            'round_number': self.round_number,
            'caller': self.caller,
            'winner': self.winner,
            'hand_values': self.hand_values,
            'coin_changes': self.coin_changes,
            'successful_call': self.successful_call,
            'decision_times': self.decision_times
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
        self.SUITS = ['♠', '♥', '♦', '♣']
        self.RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.RANK_ORDER = {rank: i for i, rank in enumerate(self.RANKS)}
        self.jhyap_attempts = [0] * num_players
        self.jhyap_successes = [0] * num_players

    def create_deck(self) -> List[Card]:
        deck = [Card(suit, rank) for suit in self.SUITS for rank in self.RANKS]
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

    def validate_discard(self, cards: List[Card]) -> bool:
        return len(cards) == 1

    def get_active_players(self) -> List[int]:
        return [i for i, coins in enumerate(self.player_coins) if coins > 0]

    def is_game_over(self) -> bool:
        return len(self.get_active_players()) < MIN_PLAYERS

class DhumbalEnv:
    def __init__(self, game: DhumbalGame, ai_players: List, current_player: int):
        self.game = game
        self.ai_players = ai_players
        self.current_player = current_player
        self.hands, self.deck = game.deal_cards()
        self.discard_pile = [self.deck.pop()] if self.deck else []
        self.state = GameState(
            round_number=game.round_number + 1,
            current_player=current_player,
            hands=[hand[:] for hand in self.hands],
            discard_pile=self.discard_pile[:],
            deck_size=len(self.deck),
            player_coins=game.player_coins.copy(),
            turn_count=0,
            phase='call'
        )

    def reset(self) -> np.ndarray:
        self.hands, self.deck = self.game.deal_cards()
        self.discard_pile = [self.deck.pop()] if self.deck else []
        self.state = GameState(
            round_number=self.game.round_number + 1,
            current_player=self.current_player,
            hands=[hand[:] for hand in self.hands],
            discard_pile=self.discard_pile[:],
            deck_size=len(self.deck),
            player_coins=self.game.player_coins.copy(),
            turn_count=0,
            phase='call'
        )
        logger.info(f"Reset environment: Round {self.state.round_number}, Deck size {self.state.deck_size}, Initial hands {[self.game.calculate_hand_value(hand) for hand in self.hands]}")
        return self.encode_state()

    def set_state(self, new_state: GameState):
        self.state = new_state.copy()
        self.hands = [hand[:] for hand in new_state.hands]
        self.discard_pile = new_state.discard_pile[:]
        self.game.player_coins = new_state.player_coins.copy()
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
        hand_value = self.game.calculate_hand_value(hand) / (13 * HAND_SIZE)
        turn_norm = self.state.turn_count / MAX_TURNS
        avg_opp_hand_size = np.mean([len(h) / HAND_SIZE for i, h in enumerate(self.state.hands) if i != self.state.current_player]) if self.game.num_players > 1 else 0
        my_coins_norm = self.state.player_coins[self.state.current_player] / STARTING_COINS
        avg_opp_coins = np.mean([c / STARTING_COINS for i, c in enumerate(self.state.player_coins) if i != self.state.current_player]) if self.game.num_players > 1 else 0
        phase_one_hot = np.zeros(3)
        phase_map = {'call': 0, 'discard': 1, 'pick': 2}
        phase_one_hot[phase_map[self.state.phase]] = 1
        state = np.concatenate([
            hand_encoding,          # 52
            discard_encoding,       # 52
            player_one_hot,         # 4
            [hand_value, turn_norm, avg_opp_hand_size, my_coins_norm, avg_opp_coins],  # 5
            phase_one_hot,          # 3
            [0]                     # 1 (padding to 117)
        ])
        assert len(state) == STATE_SIZE, f"State size {len(state)}, expected {STATE_SIZE}"
        return state

    def get_actions(self) -> List:
        actions = []
        if self.state.phase == 'call':
            if self.game.can_call_jhyap(self.state.hands[self.state.current_player]):
                actions = [True, False]
                hand_value = self.game.calculate_hand_value(self.state.hands[self.state.current_player])
                if hand_value <= JHYAP_THRESHOLD / 2:  # Prioritize Jhyap for very low hands
                    actions = [True]
            else:
                actions = [False]
        elif self.state.phase == 'discard':
            hand = self.state.hands[self.state.current_player]
            actions = [[card] for card in hand] if hand else [[]]
        elif self.state.phase == 'pick':
            if len(self.state.hands[self.state.current_player]) < HAND_SIZE:
                actions = ['deck']
                if self.state.discard_pile:
                    actions.append('discard')
        return actions

    def action_to_index(self, action) -> int:
        actions = self.get_actions()
        if not actions:
            return 0
        if self.state.phase == 'call':
            return 0 if action is False else 1
        elif self.state.phase == 'discard':
            hand = self.state.hands[self.state.current_player]
            all_combinations = [[card] for card in hand] if hand else [[]]
            all_combinations.sort(key=lambda x: tuple((c.suit, c.rank) for c in x) if x else ())
            for idx, combo in enumerate(all_combinations):
                if combo == action:
                    return idx
            return 0
        else:
            return 0 if action == 'deck' else 1

    def index_to_action(self, index: int):
        actions = self.get_actions()
        if not actions:
            if self.state.phase == 'call':
                return False
            elif self.state.phase == 'discard':
                return [self.state.hands[self.state.current_player][0]] if self.state.hands[self.state.current_player] else []
            else:
                return 'deck'
        return actions[index % len(actions)]

    def get_action_space_size(self) -> int:
        return max(1, len(self.get_actions()))

    def step(self, action, player_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        new_state = self.state.copy()
        reward = 0.0
        done = False
        coin_change = 0
        player = new_state.current_player
        hand_size = len(new_state.hands[player])
        hand_value = self.game.calculate_hand_value(new_state.hands[player])
        log_entry = {
            'turn': new_state.turn_count + 1,
            'player': player,
            'phase': new_state.phase,
            'hand_size': hand_size,
            'hand_value': hand_value,
            'action': [{'suit': card.suit, 'rank': card.rank} for card in action] if isinstance(action, list) else action,
            'state_before': new_state.to_dict()
        }
        logger.info(f"Turn {new_state.turn_count+1}, Player {player}, Phase {new_state.phase}, Hand size {hand_size}, Hand value {hand_value}, Action {action}")

        if new_state.phase == 'call':
            if action and self.game.can_call_jhyap(new_state.hands[player]):
                self.game.jhyap_attempts[player] += 1
                hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                caller = player
                caller_value = hand_values[caller]
                min_value = min(hand_values)
                min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                if len(min_value_players) == 1 and min_value_players[0] == caller:
                    self.game.jhyap_successes[player] += 1
                    new_state.winner = caller
                    reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != caller)
                    coin_change = reward
                    for i in range(self.game.num_players):
                        if i == caller:
                            new_state.player_coins[i] += coin_change
                        else:
                            new_state.player_coins[i] -= min(hand_values[i], MAX_PAYMENT)
                else:
                    non_caller_min = [i for i in min_value_players if i != caller]
                    new_state.winner = min(non_caller_min) if non_caller_min else min_value_players[0]
                    reward = -sum(min(v, MAX_PAYMENT) for v in hand_values)
                    coin_change = reward
                    for i in range(self.game.num_players):
                        new_state.player_coins[i] -= min(hand_values[i], MAX_PAYMENT)
                    new_state.player_coins[caller] -= coin_change
                new_state.done = True
                logger.info(f"Jhyap call: Hand values {hand_values}, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
            elif action and not self.game.can_call_jhyap(new_state.hands[player]):
                reward = -100.0
                coin_change = -100
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                new_state.player_coins[player] += coin_change
                logger.info(f"Invalid Jhyap call: Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
            else:
                new_state.phase = 'discard'
                logger.info(f"Player {player} chose not to call Jhyap, moving to discard phase")

        elif new_state.phase == 'discard':
            if self.game.validate_discard(action) and all(isinstance(card, Card) and card in new_state.hands[player] for card in action):
                for card in action:
                    new_state.hands[player].remove(card)
                new_state.discard_pile.extend(action)
                new_state.phase = 'pick'
                logger.info(f"Player {player} discarded {action}, moving to pick phase")
            else:
                reward = -100.0
                coin_change = -100
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                new_state.player_coins[player] += coin_change
                logger.info(f"Invalid discard: Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")

        elif new_state.phase == 'pick':
            if len(new_state.hands[player]) >= HAND_SIZE:
                reward = -100.0
                coin_change = -100
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                new_state.player_coins[player] += coin_change
                logger.info(f"Hand size limit exceeded: Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
            elif action == 'discard' and new_state.discard_pile:
                card = new_state.discard_pile.pop()
                new_state.hands[player].append(card)
                new_state.deck_size = len(self.deck)
                logger.info(f"Player {player} picked {card} from discard pile, new hand value {self.game.calculate_hand_value(new_state.hands[player])}")
            elif action == 'deck':
                if not self.deck and len(new_state.discard_pile) >= MIN_DISCARD_PILE_SIZE:
                    top = new_state.discard_pile.pop() if new_state.discard_pile else None
                    random.shuffle(new_state.discard_pile)
                    self.deck.extend(new_state.discard_pile[:])
                    new_state.discard_pile = [top] if top else []
                    new_state.deck_size = len(self.deck)
                    logger.info(f"Shuffled discard pile into deck, new deck size {new_state.deck_size}")
                if self.deck:
                    card = self.deck.pop()
                    new_state.hands[player].append(card)
                    new_state.deck_size = len(self.deck)
                    logger.info(f"Player {player} picked {card} from deck, new hand value {self.game.calculate_hand_value(new_state.hands[player])}")
                else:
                    new_state.done = True
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    new_state.winner = hand_values.index(min(hand_values))
                    reward = -sum(min(v, MAX_PAYMENT) for v in hand_values) if new_state.winner != player else sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != player)
                    coin_change = reward
                    for i in range(self.game.num_players):
                        if i == new_state.winner:
                            new_state.player_coins[i] += coin_change
                        else:
                            new_state.player_coins[i] -= min(hand_values[i], MAX_PAYMENT)
                    logger.info(f"Deck exhausted: Hand values {hand_values}, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
            else:
                reward = -100.0
                coin_change = -100
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                new_state.player_coins[player] += coin_change
                logger.info(f"Invalid pick: Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
            if not new_state.done:
                new_state.turn_count += 1
                new_state.current_player = (new_state.current_player + 1) % self.game.num_players
                new_state.phase = 'call'
                logger.info(f"Advancing to next turn: Turn {new_state.turn_count+1}, Next player {new_state.current_player}")
                if new_state.turn_count >= MAX_TURNS:
                    new_state.done = True
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    new_state.winner = hand_values.index(min(hand_values))
                    reward = -sum(min(v, MAX_PAYMENT) for v in hand_values) if new_state.winner != player else sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != player)
                    coin_change = reward
                    for i in range(self.game.num_players):
                        if i == new_state.winner:
                            new_state.player_coins[i] += coin_change
                        else:
                            new_state.player_coins[i] -= min(hand_values[i], MAX_PAYMENT)
                    logger.info(f"Max turns reached: Hand values {hand_values}, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")

        self.set_state(new_state)
        log_entry['reward'] = reward
        log_entry['coin_change'] = coin_change
        log_entry['done'] = done
        log_entry['state_after'] = new_state.to_dict()
        return self.encode_state(), reward, done, log_entry

class MCTSNode:
    def __init__(self, state: GameState, env: DhumbalEnv, parent: Optional['MCTSNode'] = None, action=None):
        if not isinstance(state, GameState):
            raise TypeError(f"Expected GameState, got {type(state)}")
        self.state = state
        self.env = env
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_value = 0.0
        self.prior = 0.0
        self.untried_actions = env.get_actions()

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def select_child(self) -> 'MCTSNode':
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb_score())

    def ucb_score(self) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.total_value / self.visits
        exploration = UCB_C * math.sqrt(math.log(self.parent.visits + 1) / (self.visits + 1e-6))
        return exploitation + exploration + self.prior

    def expand(self):
        if not self.untried_actions:
            return None
        action = self.untried_actions.pop(0)
        env_copy = DhumbalEnv(self.env.game, self.env.ai_players, self.state.current_player)
        env_copy.set_state(self.state)
        next_state_encoded, _, done, log_entry = env_copy.step(action, self.state.current_player)
        next_state = GameState.from_dict(log_entry['state_after'])
        child = MCTSNode(next_state, env_copy, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, value: float):
        self.visits += 1
        self.total_value += value

class ISMCTSNode:
    def __init__(self, state: GameState, env: DhumbalEnv, parent: Optional['ISMCTSNode'] = None, action=None):
        if not isinstance(state, GameState):
            raise TypeError(f"Expected GameState, got {type(state)}")
        self.state = state
        self.env = env
        self.parent = parent
        self.action = action
        self.children: List['ISMCTSNode'] = []
        self.visits = 0
        self.total_value = 0.0
        self.prior = 0.0
        self.untried_actions = env.get_actions()

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def select_child(self) -> 'ISMCTSNode':
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb_score())

    def ucb_score(self) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.total_value / self.visits
        exploration = UCB_C * math.sqrt(math.log(self.parent.visits + 1) / (self.visits + 1e-6))
        return exploitation + exploration + self.prior

    def expand(self):
        if not self.untried_actions:
            return None
        action = self.untried_actions.pop(0)
        env_copy = DhumbalEnv(self.env.game, self.env.ai_players, self.state.current_player)
        env_copy.set_state(self.state)
        next_state_encoded, _, done, log_entry = env_copy.step(action, self.state.current_player)
        next_state = GameState.from_dict(log_entry['state_after'])
        child = ISMCTSNode(next_state, env_copy, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, value: float):
        self.visits += 1
        self.total_value += value

class HybridAgent:
    _shared_models = {
        'ppo_actor': None,
        'ppo_critic': None,
        'dqn_model': None
    }

    def __init__(self, state_size: int, max_action_size: int, model_type: str):
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.model_type = model_type
        self.ppo_actor_path = './ppo/ppo_actor_final.weights.h5'
        self.ppo_critic_path = './ppo/ppo_critic_final.weights.h5'
        self.dqn_model_path = './dqn/dqn_model_ep500.weights.h5'
        os.makedirs('./ppo', exist_ok=True)
        os.makedirs('./dqn', exist_ok=True)
        if model_type in ['ppo_mcts', 'ppo_ismcts']:
            if HybridAgent._shared_models['ppo_actor'] is None:
                self.actor = self.build_actor()
                self.critic = self.build_critic()
                self.load_ppo_models()
                HybridAgent._shared_models['ppo_actor'] = self.actor
                HybridAgent._shared_models['ppo_critic'] = self.critic
            else:
                self.actor = HybridAgent._shared_models['ppo_actor']
                self.critic = HybridAgent._shared_models['ppo_critic']
        elif model_type in ['dqn_mcts', 'dqn_ismcts']:
            if HybridAgent._shared_models['dqn_model'] is None:
                self.model = self.build_dqn_model()
                self.load_dqn_model()
                HybridAgent._shared_models['dqn_model'] = self.model
            else:
                self.model = HybridAgent._shared_models['dqn_model']
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

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

    def build_dqn_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.max_action_size, activation='linear')(x)
        return models.Model(inputs, outputs)

    def load_ppo_models(self):
        if not (os.path.exists(self.ppo_actor_path) and os.path.exists(self.ppo_critic_path)):
            raise FileNotFoundError(f"PPO model files not found: {self.ppo_actor_path}, {self.ppo_critic_path}")
        self.actor.load_weights(self.ppo_actor_path)
        self.critic.load_weights(self.ppo_critic_path)
        logger.info(f"Loaded PPO models: {self.ppo_actor_path}, {self.ppo_critic_path}")

    def load_dqn_model(self):
        if not os.path.exists(self.dqn_model_path):
            raise FileNotFoundError(f"DQN model file not found: {self.dqn_model_path}")
        self.model.load_weights(self.dqn_model_path)
        logger.info(f"Loaded DQN model: {self.dqn_model_path}")

    def get_policy(self, state: np.ndarray, env: DhumbalEnv) -> np.ndarray:
        action_space_size = env.get_action_space_size()
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        with tf.device('/GPU:0'):
            if self.model_type in ['ppo_mcts', 'ppo_ismcts']:
                probs = self.actor(state, training=False)[0].numpy()
            else:
                q_values = self.model(state, training=False)[0].numpy()
                probs = np.exp(q_values / 0.1) / np.sum(np.exp(q_values) + 1e-10)  # Softmax with temperature
        probs = probs[:action_space_size]
        probs = probs / np.sum(probs + 1e-10)
        return probs

    def get_value(self, state: np.ndarray) -> float:
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        with tf.device('/GPU:0'):
            if self.model_type in ['ppo_mcts', 'ppo_ismcts']:
                value = self.critic(state, training=False).numpy()[0][0]
            else:
                q_values = self.model(state, training=False)[0].numpy()
                value = np.max(q_values)
        return value

    def mcts_search(self, env: DhumbalEnv, root_state: GameState) -> Tuple[int, any]:
        root = MCTSNode(root_state, env) if self.model_type in ['ppo_mcts', 'dqn_mcts'] else ISMCTSNode(root_state, env)
        root_state_encoded = env.encode_state()
        probs = self.get_policy(root_state_encoded, env)
        root.untried_actions = env.get_actions()
        for action_idx, action in enumerate(root.untried_actions):
            if action_idx < len(probs):
                child_env = DhumbalEnv(env.game, env.ai_players, root_state.current_player)
                child_env.set_state(root_state)
                _, _, done, log_entry = child_env.step(action, root_state.current_player)
                next_state = GameState.from_dict(log_entry['state_after'])
                child = MCTSNode(next_state, child_env, parent=root, action=action) if self.model_type in ['ppo_mcts', 'dqn_mcts'] else ISMCTSNode(next_state, child_env, parent=root, action=action)
                child.prior = probs[action_idx]
                root.children.append(child)
        root.untried_actions = []

        iterations = ISMCTS_SAMPLES * MCTS_ITERATIONS if 'ismcts' in self.model_type else MCTS_ITERATIONS
        for _ in range(iterations):
            node = root
            sim_env = DhumbalEnv(env.game, env.ai_players, node.state.current_player)
            sim_env.set_state(node.state)
            while not node.is_leaf() and node.is_fully_expanded():
                node = node.select_child()
            if not node.is_fully_expanded() and not node.state.done:
                node = node.expand()
            if not node.state.done:
                value = self.get_value(sim_env.encode_state())
            else:
                value = 1.0 if node.state.winner == root_state.current_player else -1.0 if node.state.winner is not None else 0.0
                value *= 100
            while node:
                node.update(value)
                node = node.parent

        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        if best_child:
            action_idx = env.action_to_index(best_child.action)
            logger.info(f"{self.model_type.upper()} selected action {action_idx} (visits: {best_child.visits}, value: {best_child.total_value / best_child.visits:.2f})")
            return action_idx, best_child.action
        action_idx = random.randint(0, env.get_action_space_size() - 1)
        action = env.index_to_action(action_idx)
        logger.info(f"{self.model_type.upper()} fallback to random action {action_idx}")
        return action_idx, action

class LearningBasedAI:
    def __init__(self, player_id: int, state_size: int, max_action_size: int, model_type: str):
        self.player_id = player_id
        self.name = f"AI_{model_type}_{player_id}"
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.model_type = model_type
        self.agent = HybridAgent(state_size, max_action_size, model_type)
        self.decision_times = []

    def choose_discard(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> List[Card]:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'discard'
        start_time = time.time()
        _, action = self.agent.mcts_search(env, game_state)
        self.decision_times.append(time.time() - start_time)
        return action if isinstance(action, list) and game.validate_discard(action) else [max(hand, key=lambda x: x.value)] if hand else []

    def should_pick_from_discard(self, discard_pile: List[Card], current_hand: List[Card], game_state: GameState, game: DhumbalGame) -> Tuple[bool, Optional[Card]]:
        if len(current_hand) >= HAND_SIZE: return False, None
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'pick'
        start_time = time.time()
        _, action = self.agent.mcts_search(env, game_state)
        self.decision_times.append(time.time() - start_time)
        return (True, discard_pile[-1]) if action == 'discard' and discard_pile else (False, None)

    def should_call_jhyap(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> bool:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'call'
        start_time = time.time()
        _, action = self.agent.mcts_search(env, game_state)
        self.decision_times.append(time.time() - start_time)
        return action

def calculate_cohens_d(data1: List[float], data2: List[float]) -> float:
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

def evaluate_hybrid_agents(game: DhumbalGame):
    state_size = STATE_SIZE
    max_action_size = MAX_ACTION_SIZE
    hybrid_agents = [
        LearningBasedAI(0, state_size, max_action_size, model_type='ppo_mcts'),
        LearningBasedAI(1, state_size, max_action_size, model_type='dqn_mcts'),
        LearningBasedAI(2, state_size, max_action_size, model_type='ppo_ismcts'),
        LearningBasedAI(3, state_size, max_action_size, model_type='dqn_ismcts')
    ]
    results = {agent.name: {'wins': [], 'rewards': [], 'coin_changes': [], 'jhyap_attempts': [], 'jhyap_successes': [], 'decision_times': [], 'hand_values': []} for agent in hybrid_agents}
    csv_path = './hybrid_tournament_results.csv'
    json_path = './hybrid_tournament_logs.json'
    json_log = []

    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Agent', 'Player_ID', 'Win', 'Reward', 'Coin_Change', 'Jhyap_Attempt', 'Jhyap_Success', 'Decision_Time', 'Hand_Value'])
            csvfile.flush()

            logger.info("Evaluating Hybrid Agents in Tournament...")
            env = DhumbalEnv(game, hybrid_agents, 0)
            for episode in range(EVALUATION_EPISODES):
                state = env.reset()
                episode_rewards = {agent.name: 0 for agent in hybrid_agents}
                episode_coin_changes = {agent.name: 0 for agent in hybrid_agents}
                episode_jhyap_attempts = {agent.name: 0 for agent in hybrid_agents}
                episode_jhyap_successes = {agent.name: 0 for agent in hybrid_agents}
                turns = 0
                caller = -1
                episode_log = {
                    'episode': episode + 1,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'turns': [],
                    'winner': None,
                    'hand_values': [],
                    'caller': -1,
                    'successful_call': False
                }
                logger.info(f"Starting Episode {episode+1}/{EVALUATION_EPISODES}")

                while not env.state.done:
                    current_player = env.state.current_player
                    if not (0 <= current_player < NUM_PLAYERS):
                        raise ValueError(f"Invalid current_player index: {current_player}")
                    player = hybrid_agents[current_player]
                    start_time = time.time()
                    _, action = player.agent.mcts_search(env, env.state)
                    decision_time = time.time() - start_time
                    player.decision_times.append(decision_time)
                    jhyap_attempt = 1 if env.state.phase == 'call' and action else 0
                    episode_jhyap_attempts[player.name] += jhyap_attempt
                    next_state, reward, done, log_entry = env.step(action, current_player)
                    episode_rewards[player.name] += reward
                    episode_coin_changes[player.name] += log_entry['coin_change']
                    if env.state.phase == 'call' and action and env.game.can_call_jhyap(env.state.hands[current_player]):
                        caller = current_player
                        episode_jhyap_successes[player.name] += 1 if env.state.winner == current_player else 0
                    turns += 1
                    episode_log['turns'].append(log_entry)
                    state = next_state
                    if done:
                        break

                episode_log['winner'] = env.state.winner
                episode_log['hand_values'] = [game.calculate_hand_value(h) for h in env.state.hands]
                episode_log['caller'] = caller
                episode_log['successful_call'] = caller != -1 and env.state.winner == caller
                json_log.append(episode_log)
                for agent in hybrid_agents:
                    win = 1 if env.state.winner == agent.player_id else 0
                    final_hand_value = game.calculate_hand_value(env.state.hands[agent.player_id])
                    results[agent.name]['wins'].append(win)
                    results[agent.name]['rewards'].append(episode_rewards[agent.name])
                    results[agent.name]['coin_changes'].append(episode_coin_changes[agent.name])
                    results[agent.name]['jhyap_attempts'].append(episode_jhyap_attempts[agent.name])
                    results[agent.name]['jhyap_successes'].append(episode_jhyap_successes[agent.name])
                    results[agent.name]['decision_times'].append(np.mean(agent.decision_times[-turns:]) if agent.decision_times else 0)
                    results[agent.name]['hand_values'].append(final_hand_value)
                    writer.writerow([
                        episode + 1,
                        agent.model_type,
                        agent.player_id,
                        win,
                        episode_rewards[agent.name],
                        episode_coin_changes[agent.name],
                        episode_jhyap_attempts[agent.name],
                        episode_jhyap_successes[agent.name],
                        results[agent.name]['decision_times'][-1],
                        final_hand_value
                    ])
                csvfile.flush()
                logger.info(f"Episode {episode+1} ended after {turns} turns, Winner: Player {env.state.winner} ({hybrid_agents[env.state.winner].model_type.upper()})")

                if episode % 100 == 0 or episode == EVALUATION_EPISODES - 1:
                    with open(json_path, 'w') as f:
                        json.dump(json_log, f, indent=2)
                    if os.path.exists(json_path):
                        file_size = os.path.getsize(json_path)
                        logger.info(f"JSON log saved to {json_path}, File size: {file_size} bytes")

                if episode % 100 == 0:
                    for agent in hybrid_agents:
                        win_rate = np.mean(results[agent.name]['wins'][-100:])
                        avg_reward = np.mean(results[agent.name]['rewards'][-100:])
                        avg_coin_change = np.mean(results[agent.name]['coin_changes'][-100:])
                        jhyap_success_rate = np.mean(results[agent.name]['jhyap_successes'][-100:]) / max(1, np.mean(results[agent.name]['jhyap_attempts'][-100:]))
                        avg_decision_time = np.mean(results[agent.name]['decision_times'][-100:])
                        avg_hand_value = np.mean(results[agent.name]['hand_values'][-100:])
                        logger.info(
                            f"Episode {episode+1}/{EVALUATION_EPISODES}, "
                            f"{agent.model_type.upper()} (Player {agent.player_id}) "
                            f"Win Rate: {win_rate:.3f}, "
                            f"Avg Reward: {avg_reward:.1f}, "
                            f"Avg Coin Change: {avg_coin_change:.1f}, "
                            f"Jhyap Success: {jhyap_success_rate:.3f}, "
                            f"Avg Decision Time: {avg_decision_time:.4f}s, "
                            f"Avg Hand Value: {avg_hand_value:.1f}"
                        )

        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            logger.info(f"Tournament results saved to {csv_path}, File size: {file_size} bytes")
            with open(csv_path, 'r') as f:
                content = f.read()
                logger.info(f"CSV content preview: {content[:200]}...")

        # Statistical analysis
        comparisons = []
        agent_names = [agent.name for agent in hybrid_agents]
        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                name1, name2 = agent_names[i], agent_names[j]
                wins1, wins2 = results[name1]['wins'], results[name2]['wins']
                coins1, coins2 = results[name1]['coin_changes'], results[name2]['coin_changes']
                jhyaps1, jhyaps2 = results[name1]['jhyap_successes'], results[name2]['jhyap_successes']
                d_wins = calculate_cohens_d(wins1, wins2)
                d_coins = calculate_cohens_d(coins1, coins2)
                d_jhyaps = calculate_cohens_d(jhyaps1, jhyaps2)
                p_wins = ttest_ind(wins1, wins2, equal_var=False).pvalue
                p_coins = ttest_ind(coins1, coins2, equal_var=False).pvalue
                p_jhyaps = ttest_ind(jhyaps1, jhyaps2, equal_var=False).pvalue
                comparisons.append({
                    'comparison': f"{name1} vs {name2}",
                    'win_d': d_wins,
                    'win_p': p_wins,
                    'economic_d': d_coins,
                    'economic_p': p_coins,
                    'jhyap_d': d_jhyaps,
                    'jhyap_p': p_jhyaps
                })

        # Summarize results
        summary = []
        for agent in hybrid_agents:
            win_rate = np.mean(results[agent.name]['wins'])
            avg_reward = np.mean(results[agent.name]['rewards'])
            avg_coin_change = np.mean(results[agent.name]['coin_changes'])
            jhyap_success_rate = np.sum(results[agent.name]['jhyap_successes']) / max(1, np.sum(results[agent.name]['jhyap_attempts']))
            avg_decision_time = np.mean(results[agent.name]['decision_times'])
            avg_hand_value = np.mean(results[agent.name]['hand_values'])
            summary.append({
                'agent_type': agent.model_type,
                'player_id': agent.player_id,
                'win_rate': win_rate,
                'avg_reward': avg_reward,
                'avg_coin_change': avg_coin_change,
                'jhyap_success_rate': jhyap_success_rate,
                'avg_decision_time': avg_decision_time,
                'avg_hand_value': avg_hand_value
            })
            logger.info(f"\n{agent.model_type.upper()} (Player {agent.player_id}) Results:")
            logger.info(f"Win Rate: {win_rate:.3f}")
            logger.info(f"Average Reward: {avg_reward:.1f}")
            logger.info(f"Average Coin Change: {avg_coin_change:.1f}")
            logger.info(f"Jhyap Success Rate: {jhyap_success_rate:.3f}")
            logger.info(f"Average Decision Time: {avg_decision_time:.4f}s")
            logger.info(f"Average Hand Value: {avg_hand_value:.1f}")

        # Statistical comparisons
        for comp in comparisons:
            logger.info(f"\nComparison {comp['comparison']}:")
            logger.info(f"Win: Cohen's d = {comp['win_d']:.3f}, p-value = {comp['win_p']:.4f}")
            logger.info(f"Economic: Cohen's d = {comp['economic_d']:.3f}, p-value = {comp['economic_p']:.4f}")
            logger.info(f"Jhyap: Cohen's d = {comp['jhyap_d']:.3f}, p-value = {comp['jhyap_p']:.4f}")

        best_agent = max(summary, key=lambda x: (x['win_rate'], x['avg_coin_change']))
        logger.info(f"\nBest Hybrid Agent: {best_agent['agent_type'].upper()} (Player {best_agent['player_id']})")
        logger.info(f"Win Rate: {best_agent['win_rate']:.3f}")
        logger.info(f"Average Reward: {best_agent['avg_reward']:.1f}")
        logger.info(f"Average Coin Change: {best_agent['avg_coin_change']:.1f}")
        logger.info(f"Jhyap Success Rate: {best_agent['jhyap_success_rate']:.3f}")
        logger.info(f"Average Decision Time: {best_agent['avg_decision_time']:.4f}s")
        logger.info(f"Average Hand Value: {best_agent['avg_hand_value']:.1f}")

        return summary, comparisons

    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        with open(json_path, 'w') as f:
            json.dump(json_log, f, indent=2)
        raise
    except Exception as e:
        logger.error(f"Tournament failed: {e}")
        with open(json_path, 'w') as f:
            json.dump(json_log, f, indent=2)
        raise

if __name__ == "__main__":
    game = DhumbalGame(num_players=NUM_PLAYERS)
    try:
        logger.info("Starting Hybrid Agents Tournament...")
        summary, comparisons = evaluate_hybrid_agents(game)
        logger.info("\nTournament Complete.")
        # Save summary and comparisons
        with open('./hybrid_tournament_summary.json', 'w') as f:
            json.dump({'summary': summary, 'comparisons': comparisons}, f, indent=2)
        logger.info("Tournament summary saved to ./hybrid_tournament_summary.json")
    except Exception as e:
        logger.error(f"Tournament failed: {e}")