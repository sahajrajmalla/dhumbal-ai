import tensorflow as tf
import numpy as np
import random
import csv
import os
import json
from collections import deque, defaultdict
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from tensorflow.keras import models, layers, optimizers
from datetime import datetime
import itertools

# Configure TensorFlow for GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
TRAINING_EPISODES = 5000
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PPO_CLIP = 0.2
PPO_EPOCHS = 5
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5
MAX_MEMORY = 2000
# --- Early stopping (statistically grounded; applied to both PPO and DQN) ---
# Tuned for a ~5000-episode budget on limited (Colab) compute. Parameters live in
# each training file so PPO and DQN can be tuned independently if needed.
ES_WARMUP_EPISODES = 1000    # no early stopping before this (covers exploration / epsilon anneal)
ES_WINDOW = 500              # rolling window for the win-rate and reward signals
ES_CHECK_FREQ = 100          # evaluate the stop criterion every N episodes
ES_PATIENCE = 15             # recent checks inspected for a sustained plateau (slope-significance baseline)
ES_MIN_DELTA = 0.02          # floor on a "meaningful" win-rate change (~1 std-error at p~0.1-0.25)
ES_Z = 1.5                   # noise multiplier: a trend/volatility must clear z * standard-error to count

class AIStyle(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    OPPORTUNISTIC = "opportunistic"
    BALANCED = "balanced"
    PPO = "ppo"

@dataclass
class GameState:
    round_number: int
    current_player: int
    hands: List[List['Card']]
    discard_pile: List['Card']
    deck: List['Card']
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
            deck=self.deck[:],
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
            'deck': [str(card) for card in self.deck],
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
    successful_call: bool

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

    def validate_same_rank_set(self, cards: List[Card]) -> bool:
        if len(cards) < 2:
            return False
        return all(card.rank == cards[0].rank for card in cards)

    def validate_sequence(self, cards: List[Card]) -> bool:
        if len(cards) < 3:
            return False
        if not all(card.suit == cards[0].suit for card in cards):
            return False
        try:
            positions = sorted([self.RANK_ORDER[card.rank] for card in cards])
            if 'A' in [card.rank for card in cards] and positions[0] != 0:
                return False
            return all(positions[i] == positions[i-1] + 1 for i in range(1, len(positions)))
        except KeyError:
            return False

    def validate_discard(self, cards: List[Card]) -> bool:
        if not cards:
            return False
        if len(cards) == 1:
            return True
        return self.validate_same_rank_set(cards) or self.validate_sequence(cards)

    def get_active_players(self) -> List[int]:
        return [i for i, coins in enumerate(self.player_coins) if coins > 0]

    def is_game_over(self) -> bool:
        return len(self.get_active_players()) < MIN_PLAYERS

class RuleBasedAI:
    def __init__(self, player_id: int, style: AIStyle):
        self.player_id = player_id
        self.style = style
        self.name = f"AI_{style.value}_{player_id}"

    def choose_discard(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> List[Card]:
        if not hand: return []
        valid_discards = []
        for card in hand:
            valid_discards.append([card])
        rank_groups = defaultdict(list)
        for card in hand:
            rank_groups[card.rank].append(card)
        for cards in rank_groups.values():
            if len(cards) >= 2:
                for size in range(2, len(cards) + 1):
                    valid_discards.extend(list(combo) for combo in itertools.combinations(cards, size))
        suit_groups = defaultdict(list)
        for card in hand:
            suit_groups[card.suit].append(card)
        for suit, cards in suit_groups.items():
            if len(cards) >= 3:
                cards.sort(key=lambda x: game.RANK_ORDER[x.rank])
                for size in range(3, len(cards) + 1):
                    for combo in itertools.combinations(cards, size):
                        if game.validate_sequence(list(combo)):
                            valid_discards.append(list(combo))
        if not valid_discards:
            return []
        hand_value = sum(card.value for card in hand)
        if self.style == AIStyle.CONSERVATIVE:
            if hand_value <= JHYAP_THRESHOLD + 5:
                return min(valid_discards, key=lambda x: sum(c.value for c in x))
            else:
                return max(valid_discards, key=lambda x: sum(c.value for c in x))
        elif self.style == AIStyle.AGGRESSIVE:
            multi_discards = [d for d in valid_discards if len(d) > 1]
            if multi_discards:
                return max(multi_discards, key=lambda x: sum(c.value for c in x))
            else:
                return max(valid_discards, key=lambda x: sum(c.value for c in x))
        elif self.style == AIStyle.OPPORTUNISTIC:
            my_coins = game_state.player_coins[self.player_id]
            avg_coins = sum(game_state.player_coins) / len(game_state.player_coins)
            if my_coins < avg_coins:
                return max(valid_discards, key=lambda x: sum(c.value for c in x))
            else:
                return random.choice(valid_discards)
        elif self.style == AIStyle.BALANCED:
            multi_discards = [d for d in valid_discards if len(d) > 1]
            if multi_discards:
                return max(multi_discards, key=lambda x: len(x))
            else:
                return random.choice(valid_discards)
        return random.choice(valid_discards)

    def should_pick_from_discard(self, discard_pile: List[Card], current_hand: List[Card], game_state: GameState, game: DhumbalGame) -> Tuple[bool, Optional[Card]]:
        if not discard_pile or len(current_hand) >= HAND_SIZE: return False, None
        top_card = discard_pile[-1]
        current_value = sum(card.value for card in current_hand)
        if self.style == AIStyle.CONSERVATIVE:
            return (True, top_card) if top_card.value <= 3 or (top_card.value <= 5 and current_value > JHYAP_THRESHOLD) else (False, None)
        elif self.style == AIStyle.AGGRESSIVE:
            return (True, top_card) if top_card.value <= 5 else (False, None)
        elif self.style == AIStyle.OPPORTUNISTIC:
            my_coins = game_state.player_coins[self.player_id]
            avg_coins = sum(game_state.player_coins) / len(game_state.player_coins)
            return (True, top_card) if (top_card.value <= 4 and my_coins < avg_coins) or top_card.value <= 2 else (False, None)
        elif self.style == AIStyle.BALANCED:
            for card in current_hand:
                if card.rank == top_card.rank or (card.suit == top_card.suit and abs(game.RANK_ORDER[card.rank] - game.RANK_ORDER[top_card.rank]) == 1):
                    return True, top_card
            return (True, top_card) if top_card.value <= 4 else (False, None)
        return (True, top_card) if random.random() < 0.5 else (False, None)

    def should_call_jhyap(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> bool:
        hand_value = sum(card.value for card in hand)
        if hand_value > JHYAP_THRESHOLD: return False
        if self.style == AIStyle.CONSERVATIVE:
            return hand_value <= 7
        elif self.style == AIStyle.AGGRESSIVE:
            return True
        elif self.style == AIStyle.OPPORTUNISTIC:
            low_discard = any(card.value <= 5 for card in game_state.discard_pile[-3:]) if game_state.discard_pile else False
            return hand_value <= 8 and low_discard
        elif self.style == AIStyle.BALANCED:
            return hand_value <= 8
        return random.random() < 0.5

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
            deck=self.deck[:],
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
            deck=self.deck[:],
            player_coins=game.player_coins.copy(),
            turn_count=0,
            phase='call'
        )
        return self.encode_state()

    def set_state(self, new_state: GameState):
        self.state = new_state.copy()
        self.hands = [hand[:] for hand in new_state.hands]
        self.discard_pile = new_state.discard_pile[:]
        self.deck = new_state.deck[:]

    def encode_state(self) -> np.ndarray:
        hand = self.state.hands[self.state.current_player]
        hand_encoding = np.zeros(52)
        for card in hand:
            suit_idx = self.game.SUITS.index(card.suit)
            rank_idx = self.game.RANK_ORDER[card.rank]
            hand_encoding[suit_idx * 13 + rank_idx] = 1
            
        discard_encoding = np.zeros(52)
        if self.state.discard_pile:
            top_card = self.state.discard_pile[-1]
            suit_idx = self.game.SUITS.index(top_card.suit)
            rank_idx = self.game.RANK_ORDER[top_card.rank]
            discard_encoding[suit_idx * 13 + rank_idx] = 1
            
        player_one_hot = np.zeros(self.game.num_players)
        player_one_hot[self.state.current_player] = 1
        
        hand_sizes_norm = [len(self.state.hands[i]) / HAND_SIZE for i in range(self.game.num_players)]
        # Coin counts and the cross-round counter are constant *within* a round during
        # self-play training (coins change only between rounds, which training does not
        # carry over). They are held constant here so the 128-d encoding is identical in
        # training and evaluation and the learned policy transfers faithfully.
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
        assert state.shape[0] == 128, f"State size mismatch: expected 128, got {state.shape[0]}"
        return state

    def get_actions(self) -> List:
        actions = []
        if self.state.phase == 'call':
            actions = [True, False] if self.game.can_call_jhyap(self.state.hands[self.state.current_player]) else [False]
        elif self.state.phase == 'discard':
            hand = self.state.hands[self.state.current_player]
            if not hand:
                return [[]]
            for card in hand:
                actions.append([card])
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
                    cards.sort(key=lambda x: self.game.RANK_ORDER[x.rank])
                    for size in range(3, len(cards) + 1):
                        for combo in itertools.combinations(cards, size):
                            if self.game.validate_sequence(list(combo)):
                                actions.append(list(combo))
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
        def action_key(act):
            if isinstance(act, bool):
                return str(act)
            elif isinstance(act, str):
                return act
            else:
                return tuple(sorted(str(c) for c in act))
        sorted_actions = sorted(actions, key=action_key)
        for idx, act in enumerate(sorted_actions):
            if action_key(act) == action_key(action):
                return idx
        return 0

    def index_to_action(self, index: int):
        actions = self.get_actions()
        if not actions:
            if self.state.phase == 'call':
                return False
            elif self.state.phase == 'discard':
                return [self.state.hands[self.state.current_player][0]] if self.state.hands[self.state.current_player] else []
            else:
                return 'deck'
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
        return 128

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        new_state = self.state.copy()
        reward = 0.0
        done = False
        player = new_state.current_player
        hand_size = len(new_state.hands[player])
        old_hand_value = self.game.calculate_hand_value(new_state.hands[player])
        log_entry = {
            'turn': new_state.turn_count + 1,
            'player': player,
            'phase': new_state.phase,
            'hand_size': hand_size,
            'old_hand_value': old_hand_value,
            'action': str(action) if isinstance(action, list) else action,
            'state_before': new_state.to_dict()
        }

        if new_state.phase == 'call':
            if isinstance(action, bool):
                if action and self.game.can_call_jhyap(new_state.hands[player]):
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    caller = player
                    min_value = min(hand_values)
                    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                    if len(min_value_players) == 1 and min_value_players[0] == caller:
                        new_state.winner = caller
                        reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != caller)
                    else:
                        non_caller_min = [i for i in min_value_players if i != caller]
                        new_state.winner = min(non_caller_min) if non_caller_min else min_value_players[0]
                        reward = -sum(min(v, MAX_PAYMENT) for v in hand_values)
                    new_state.done = True
                elif action:
                    reward = -100.0
                    new_state.done = True
                    new_state.winner = (player + 1) % self.game.num_players
                else:
                    new_state.phase = 'discard'
                    reward = 0.0

        elif new_state.phase == 'discard':
            if self.game.validate_discard(action) and all(card in new_state.hands[player] for card in action):
                discard_value = sum(c.value for c in action)
                for card in action:
                    new_state.hands[player].remove(card)
                new_state.discard_pile.extend(action)
                new_state.phase = 'pick'
                new_hand_value = self.game.calculate_hand_value(new_state.hands[player])
                reward = (old_hand_value - new_hand_value) / 13.0
            else:
                reward = -100.0
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players

        elif new_state.phase == 'pick':
            if len(new_state.hands[player]) >= HAND_SIZE:
                reward = -100.0
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
            elif action == 'discard' and new_state.discard_pile:
                card = new_state.discard_pile.pop()
                new_state.hands[player].append(card)
                new_hand_value = self.game.calculate_hand_value(new_state.hands[player])
                reward = (old_hand_value - new_hand_value) / 13.0 if new_hand_value < old_hand_value else -0.1
            elif action == 'deck':
                if not self.deck and len(new_state.discard_pile) >= MIN_DISCARD_PILE_SIZE:
                    top = new_state.discard_pile.pop() if new_state.discard_pile else None
                    random.shuffle(new_state.discard_pile)
                    self.deck.extend(new_state.discard_pile[:])
                    new_state.discard_pile = [top] if top else []
                if self.deck:
                    card = self.deck.pop()
                    new_state.hands[player].append(card)
                    new_hand_value = self.game.calculate_hand_value(new_state.hands[player])
                    reward = (old_hand_value - new_hand_value) / 13.0 if new_hand_value < old_hand_value else -0.1
                else:
                    new_state.done = True
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    min_value = min(hand_values)
                    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                    new_state.winner = min_value_players[0]
                    if new_state.winner == player:
                        reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != player)
                    else:
                        reward = -hand_values[player]
            else:
                reward = -100.0
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
            if not new_state.done:
                new_state.turn_count += 1
                new_state.current_player = (new_state.current_player + 1) % self.game.num_players
                new_state.phase = 'call'
                if new_state.turn_count >= MAX_TURNS:
                    new_state.done = True
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    min_value = min(hand_values)
                    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                    new_state.winner = min_value_players[0]
                    if new_state.winner == player:
                        reward += sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != player)
                    else:
                        reward += -hand_values[player]

        self.set_state(new_state)
        # BUGFIX: return new_state.done. The local `done` was never updated from the
        # branches that set new_state.done=True, so step() always returned done=False,
        # i.e. every stored transition looked non-terminal. That broke GAE bootstrapping
        # (terminal next-value never zeroed), degrading value learning.
        done = new_state.done
        log_entry['reward'] = reward
        log_entry['done'] = done
        log_entry['state_after'] = new_state.to_dict()
        return self.encode_state(), reward, done, log_entry

class PPO:
    def __init__(self, state_size: int, max_action_size: int):
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        self.critic_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

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
        action_space_size = env.get_action_space_size()
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        with tf.device('/GPU:0'):
            probs = self.actor(state, training=False)[0].numpy()
        probs = probs[:action_space_size]
        probs = probs / np.sum(probs + 1e-10)  # Normalize
        action = np.random.choice(action_space_size, p=probs)
        return action, probs[action]

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
            advantages[i] = gae
        returns = advantages + values
        return advantages, returns

    def train(self, states, actions, old_probs, rewards, values, next_values, dones):
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        old_probs = np.array(old_probs, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        advantages = np.array(advantages, dtype=np.float32)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)  # Normalize advantages

        dataset = tf.data.Dataset.from_tensor_slices((states, actions, old_probs, returns, advantages)).shuffle(len(states)).batch(BATCH_SIZE)

        for _ in range(PPO_EPOCHS):
            for batch in dataset:
                batch_states, batch_actions, batch_old_probs, batch_returns, batch_advantages = batch

                # Actor update
                with tf.GradientTape() as tape:
                    probs = self.actor(batch_states, training=True)
                    action_probs = tf.gather_nd(probs, tf.stack([tf.range(tf.shape(probs)[0]), batch_actions], axis=1))
                    ratios = action_probs / (batch_old_probs + 1e-10)
                    surr1 = ratios * batch_advantages
                    surr2 = tf.clip_by_value(ratios, 1 - PPO_CLIP, 1 + PPO_CLIP) * batch_advantages
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
                    actor_loss = policy_loss - ENTROPY_COEFF * entropy
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                # Critic update
                with tf.GradientTape() as tape:
                    critic_values = self.critic(batch_states, training=True)[:, 0]
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - critic_values)) * VALUE_LOSS_COEFF
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

class LearningBasedAI:
    def __init__(self, player_id: int, state_size: int, max_action_size: int, model_type: str = 'ppo'):
        self.player_id = player_id
        self.name = f"AI_ppo_{player_id}"
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.model_type = model_type
        self.agent = PPO(state_size, max_action_size)

    def choose_discard(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> List[Card]:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'discard'
        state = env.encode_state()
        action_idx, _ = self.agent.act(state, env)
        action = env.index_to_action(action_idx)
        return action if isinstance(action, list) and game.validate_discard(action) else [max(hand, key=lambda x: x.value)] if hand else []

    def should_pick_from_discard(self, discard_pile: List[Card], current_hand: List[Card], game_state: GameState, game: DhumbalGame) -> Tuple[bool, Optional[Card]]:
        if len(current_hand) >= HAND_SIZE: return False, None
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'pick'
        state = env.encode_state()
        action_idx, _ = self.agent.act(state, env)
        action = env.index_to_action(action_idx)
        return (True, discard_pile[-1]) if action == 'discard' and discard_pile else (False, None)

    def should_call_jhyap(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> bool:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'call'
        state = env.encode_state()
        action_idx, _ = self.agent.act(state, env)
        return env.index_to_action(action_idx)

def simulate_opponent_turn(env: DhumbalEnv, opponent: RuleBasedAI) -> Tuple[np.ndarray, bool, dict]:
    player = env.state.current_player
    old_hand_value = env.game.calculate_hand_value(env.state.hands[player])
    log_entry = {
        'turn': env.state.turn_count + 1,
        'player': player,
        'phase': env.state.phase,
        'hand_size': len(env.state.hands[player]),
        'old_hand_value': old_hand_value,
        'state_before': env.state.to_dict()
    }
    if env.state.phase == 'call':
        action = opponent.should_call_jhyap(env.state.hands[player], env.state, env.game)
    elif env.state.phase == 'discard':
        action = opponent.choose_discard(env.state.hands[player], env.state, env.game)
    else:  # pick
        should_pick, card = opponent.should_pick_from_discard(env.state.discard_pile, env.state.hands[player], env.state, env.game)
        action = 'discard' if should_pick else 'deck'
    log_entry['action'] = str(action) if isinstance(action, list) else action
    next_state, reward, done, log_entry = env.step(action)
    log_entry['reward'] = reward
    log_entry['done'] = done
    return next_state, done, log_entry

class EarlyStopping:
    """Statistically-grounded early stopping for noisy, sparse-reward,
    non-stationary multi-agent RL.

    The rolling win-rate (the task objective) is the primary signal; the rolling
    mean episode reward is a secondary signal. On a fixed cadence (`check_freq`),
    over a rolling `window`, the criterion declares a plateau and stops when, over
    the most recent `patience` checks, ALL hold:

      1. Win-rate is not significantly rising. An ordinary-least-squares slope is
         fit to the recent per-check win-rates; the signal counts as "still
         improving" only if that slope is positive by more than z times its OWN
         standard error. This detects slow-but-consistent gains (small residual ->
         small slope SE) while ignoring noisy fluctuation (large residual -> large
         slope SE), so a sub-noise wiggle is never read as progress.
      2. Win-rate is stable. The de-trended residual std of the recent per-check
         win-rates must be <= max(min_delta, z * SE), where SE = sqrt(p(1-p)/window)
         is the binomial standard error, so a policy oscillating around a flat
         trend (high residual variance) is NOT mistaken for convergence.
      3. Reward shows no significant up-trend. Reward defers stopping only when its
         own OLS slope is positive by > z standard errors; noisy reward gains
         therefore cannot keep training alive (no wasted compute), while a real
         reward climb (e.g. learning to reduce hand value before wins materialise)
         does.

    Cost: one O(window) mean and O(patience) stats, only on check episodes.
    """

    def __init__(self, warmup_episodes, window, check_freq, patience, min_delta, z):
        self.warmup_episodes = warmup_episodes
        self.window = window
        self.check_freq = check_freq
        self.patience = patience
        self.min_delta = min_delta
        self.z = z
        self.wins = deque(maxlen=window)
        self.rewards = deque(maxlen=window)
        self.win_checks = deque(maxlen=patience)   # rolling win-rate at each check
        self.rew_checks = deque(maxlen=patience)   # rolling mean reward at each check
        self.best_win_rate = -np.inf
        self.best_episode = 0
        self.win_rate = 0.0
        self.mean_reward = 0.0
        self.win_trend = 0.0
        self.win_vol = 0.0
        self.rew_trend = 0.0
        self.stop_reason = None

    def update(self, episode, win, reward):
        """Record one episode; return (should_stop, is_new_best).

        `is_new_best` flags a new best rolling win-rate so the caller can
        checkpoint the best policy independently of the stop decision.
        """
        self.wins.append(win)
        self.rewards.append(reward)

        # Only test on the cadence, after warm-up, once the window is full.
        if (episode < self.warmup_episodes
                or len(self.wins) < self.window
                or episode % self.check_freq != 0):
            return False, False

        self.win_rate = float(np.mean(self.wins))
        self.mean_reward = float(np.mean(self.rewards))
        self.win_checks.append(self.win_rate)
        self.rew_checks.append(self.mean_reward)

        is_new_best = self.win_rate > self.best_win_rate
        if is_new_best:
            self.best_win_rate = self.win_rate
            self.best_episode = episode

        # Need a full patience-history of checks before judging a sustained plateau.
        if len(self.win_checks) < self.patience:
            return False, is_new_best

        n = self.patience
        x = np.arange(n, dtype=np.float64)
        sxx = float(np.sum((x - x.mean()) ** 2))         # for the OLS slope standard error
        wc = np.array(self.win_checks, dtype=np.float64)
        rc = np.array(self.rew_checks, dtype=np.float64)
        se_p = np.sqrt(max(self.win_rate * (1.0 - self.win_rate), 1e-6) / self.window)
        tol = max(self.min_delta, self.z * se_p)

        # Fit an OLS slope to each signal over the recent checks and compare it to its OWN
        # standard error. A signal counts as "still rising" only if its slope is positive
        # by more than z standard errors (a statistically significant up-trend). This
        # detects slow-but-consistent improvement (small residual -> small slope SE) yet
        # ignores noisy fluctuation (large residual -> large slope SE).
        sw, iw = np.polyfit(x, wc, 1)
        w_resid = wc - (sw * x + iw)
        self.win_vol = float(np.std(w_resid))                  # oscillation around the trend
        self.win_trend = float(sw * (n - 1))                   # projected change across window (for logs)
        se_slope_w = float(np.sqrt(np.sum(w_resid ** 2) / max(n - 2, 1)) / np.sqrt(sxx)) + 1e-12
        sr, ir = np.polyfit(x, rc, 1)
        r_resid = rc - (sr * x + ir)
        self.rew_trend = float(sr * (n - 1))
        se_slope_r = float(np.sqrt(np.sum(r_resid ** 2) / max(n - 2, 1)) / np.sqrt(sxx)) + 1e-12

        win_rising = sw > self.z * se_slope_w                  # statistically significant win-rate up-trend
        win_stable = self.win_vol <= tol                       # low oscillation around the trend
        reward_rising = sr > self.z * se_slope_r               # statistically significant reward up-trend

        if (not win_rising) and win_stable and (not reward_rising):
            self.stop_reason = (
                f"win-rate not significantly rising (slope {sw:+.5f}/check <= "
                f"{self.z}*SE {self.z * se_slope_w:.5f}) and stable (residual std "
                f"{self.win_vol:.3f} <= tol {tol:.3f}) with no reward up-trend, over "
                f"{self.patience} checks ({self.patience * self.check_freq} episodes)"
            )
            return True, is_new_best
        return False, is_new_best

    def report(self, episode):
        return (
            "\n" + "=" * 68 + "\n"
            + "EARLY STOPPING TRIGGERED\n"
            + "=" * 68 + "\n"
            + f"  Reason              : {self.stop_reason}\n"
            + f"  Episodes trained    : {episode}\n"
            + f"  Rolling win-rate    : {self.win_rate:.4f}  (best {self.best_win_rate:.4f} @ ep {self.best_episode})\n"
            + f"  Win-rate trend / std: {self.win_trend:+.4f} / {self.win_vol:.4f}\n"
            + f"  Rolling mean reward : {self.mean_reward:.4f}  (trend {self.rew_trend:+.4f})\n"
            + f"  Config              : window={self.window}, check_freq={self.check_freq}, "
            + f"patience={self.patience}, warmup={self.warmup_episodes}, min_delta={self.min_delta}, z={self.z}\n"
            + "=" * 68
        )


def _runtime_info():
    """Best-effort software/hardware fingerprint for reproducibility (never raises)."""
    info = {}
    try:
        import platform as _pf
        info['python'] = _pf.python_version()
        info['platform'] = _pf.platform()
    except Exception:
        pass
    try:
        info['tensorflow'] = tf.__version__
    except Exception:
        pass
    try:
        info['numpy'] = np.__version__
    except Exception:
        pass
    try:
        gpus = tf.config.list_physical_devices('GPU')
        names = []
        for g in gpus:
            try:
                names.append(tf.config.experimental.get_device_details(g).get('device_name', g.name))
            except Exception:
                names.append(g.name)
        info['gpus'] = names if names else 'CPU-only'
    except Exception:
        pass
    return info


def _write_es_log(path, rows):
    """Write the per-check early-stopping diagnostics to CSV (no-op if empty)."""
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_agent(agent: LearningBasedAI, game: DhumbalGame, opponents_pool: List['RuleBasedAI'], batch_size: int):
    win_rates = []
    episode_turns = []
    best_win_rate = 0
    best_model = None
    stopper = EarlyStopping(
        warmup_episodes=ES_WARMUP_EPISODES,
        window=ES_WINDOW,
        check_freq=ES_CHECK_FREQ,
        patience=ES_PATIENCE,
        min_delta=ES_MIN_DELTA,
        z=ES_Z,
    )
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'ppo_training_results.csv')
    json_path = os.path.join(base_dir, 'ppo_training_log.json')
    meta_path = os.path.join(base_dir, 'ppo_run_metadata.json')
    summary_path = os.path.join(base_dir, 'ppo_training_summary.json')
    es_log_path = os.path.join(base_dir, 'ppo_earlystop_log.csv')
    json_log = []
    es_history = []
    run_start = datetime.now()

    # --- Reproducibility metadata: full config written once, before training starts ---
    metadata = {
        'algorithm': 'PPO',
        'timestamp_start': run_start.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': 42,
        'num_players': game.num_players,
        'training_episodes_budget': TRAINING_EPISODES,
        'state_size': getattr(agent, 'state_size', None),
        'action_size': getattr(agent, 'max_action_size', None),
        'opponents_pool': [o.style.value for o in opponents_pool],
        'hyperparameters': {
            'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'gamma': GAMMA,
            'gae_lambda': GAE_LAMBDA, 'ppo_clip': PPO_CLIP, 'ppo_epochs': PPO_EPOCHS,
            'entropy_coeff': ENTROPY_COEFF, 'value_loss_coeff': VALUE_LOSS_COEFF,
            'replay_memory': MAX_MEMORY,
        },
        'early_stopping': {
            'warmup_episodes': ES_WARMUP_EPISODES, 'window': ES_WINDOW,
            'check_freq': ES_CHECK_FREQ, 'patience': ES_PATIENCE,
            'min_delta': ES_MIN_DELTA, 'z': ES_Z,
        },
        'environment': {
            'starting_coins': STARTING_COINS, 'max_turns': MAX_TURNS, 'hand_size': HAND_SIZE,
            'jhyap_threshold': JHYAP_THRESHOLD, 'max_payment': MAX_PAYMENT,
        },
        'runtime': _runtime_info(),
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Run metadata written to {os.path.basename(meta_path)}")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Win', 'Turns', 'Reward', 'Seat'])
        csvfile.flush()
        for episode in range(TRAINING_EPISODES):
            # Dynamic Seating and Opponent Randomization
            selected_opponents = random.sample(opponents_pool, game.num_players - 1)
            players = [agent] + selected_opponents
            random.shuffle(players)
            
            # Reassign player IDs based on randomized seating
            for idx, p in enumerate(players):
                p.player_id = idx
            
            # Initialize environment. The round always starts at seat 0; because the
            # agent occupies a randomized seat each episode (assigned above), it
            # experiences every turn position over training. This matches evaluation
            # (compete.py / championship), where the agent is usually not first to act.
            env = DhumbalEnv(game, players, 0)
            
            states, actions, probs, rewards, values, next_values, dones = [], [], [], [], [], [], []
            state = env.reset()
            episode_reward = 0
            turns = 0
            while not env.state.done:
                if env.state.current_player == agent.player_id:
                    action_idx, prob = agent.agent.act(state, env)
                    action = env.index_to_action(action_idx)
                    value = agent.agent.critic(np.array(state, dtype=np.float32).reshape(1, -1))[0].numpy()[0]
                    next_state, reward, done, log_entry = env.step(action)
                    next_value = agent.agent.critic(np.array(next_state, dtype=np.float32).reshape(1, -1))[0].numpy()[0] if not done else 0
                    states.append(state)
                    actions.append(action_idx)
                    probs.append(prob)
                    rewards.append(reward)
                    values.append(value)
                    next_values.append(next_value)
                    dones.append(done)
                    episode_reward += reward
                    state = next_state
                else:
                    opponent = players[env.state.current_player]
                    state, done, _ = simulate_opponent_turn(env, opponent)
                turns += 1
                if done:
                    break
            if states:  # Only train if the agent took actions
                agent.agent.train(states, actions, probs, rewards, values, next_values, dones)

            win = 1 if env.state.winner == agent.player_id else 0
            writer.writerow([episode + 1, win, turns, episode_reward, agent.player_id])
            csvfile.flush()
            win_rates.append(win)
            episode_turns.append(turns)
            # Lightweight, bounded-memory episode log (no per-turn state dumps): avoids the
            # old O(n^2) full-log rewrite and the unbounded RAM growth of per-turn logging.
            json_log.append({
                'episode': episode + 1,
                'win': win,
                'turns': turns,
                'reward': round(episode_reward, 4),
                'winner': env.state.winner,
                'agent_seat': agent.player_id,
            })

            print(f"Episode {episode + 1}/{TRAINING_EPISODES} (Seat {agent.player_id}): Win={win}, Turns={turns}, Reward={episode_reward:.2f}")

            # Early stopping on the rolling win-rate (primary) + reward (secondary) signals.
            should_stop, is_new_best = stopper.update(episode + 1, win, episode_reward)
            if is_new_best:
                best_win_rate = stopper.best_win_rate
                best_model = (agent.agent.actor.get_weights(), agent.agent.critic.get_weights())
                agent.agent.actor.save_weights(os.path.join(base_dir, 'ppo_actor_best.weights.h5'))
                agent.agent.critic.save_weights(os.path.join(base_dir, 'ppo_critic_best.weights.h5'))

            # Record early-stopping diagnostics on every check episode: a reproducible
            # trace of how/when the win-rate converged and where training was halted.
            if (episode + 1) >= ES_WARMUP_EPISODES and (episode + 1) % ES_CHECK_FREQ == 0:
                es_history.append({
                    'episode': episode + 1,
                    'rolling_win_rate': round(stopper.win_rate, 5),
                    'rolling_mean_reward': round(stopper.mean_reward, 5),
                    'win_trend': round(stopper.win_trend, 5),
                    'win_residual_std': round(stopper.win_vol, 5),
                    'reward_trend': round(stopper.rew_trend, 5),
                    'best_win_rate': round(stopper.best_win_rate, 5),
                    'is_new_best': int(is_new_best),
                    'stopped': int(should_stop),
                })

            # Periodic (not per-episode) disk writes so the GPU is not stalled on I/O.
            # The CSV above is the per-episode live record; this is a coarse snapshot.
            if (episode + 1) % ES_CHECK_FREQ == 0 or should_stop or (episode + 1) == TRAINING_EPISODES:
                with open(json_path, 'w') as f:
                    json.dump(json_log, f, indent=2)
                _write_es_log(es_log_path, es_history)
                agent.agent.actor.save_weights(os.path.join(base_dir, 'ppo_actor_ep_last.weights.h5'))
                agent.agent.critic.save_weights(os.path.join(base_dir, 'ppo_critic_ep_last.weights.h5'))

            if should_stop:
                print(stopper.report(episode + 1))
                break

    if best_model:
        agent.agent.actor.set_weights(best_model[0])
        agent.agent.critic.set_weights(best_model[1])
        agent.agent.actor.save_weights(os.path.join(base_dir, 'ppo_actor_final.weights.h5'))
        agent.agent.critic.save_weights(os.path.join(base_dir, 'ppo_critic_final.weights.h5'))
        print("Final PPO models saved to ppo_actor_final.weights.h5 and ppo_critic_final.weights.h5")
    else:
        # Fallback if no best model
        agent.agent.actor.save_weights(os.path.join(base_dir, 'ppo_actor_final.weights.h5'))
        agent.agent.critic.save_weights(os.path.join(base_dir, 'ppo_critic_final.weights.h5'))
        print("Final PPO models saved to ppo_actor_final.weights.h5 and ppo_critic_final.weights.h5")

    # --- Training summary: quotable final results + run provenance for the paper ---
    duration = (datetime.now() - run_start).total_seconds()
    episodes_trained = episode + 1
    _write_es_log(es_log_path, es_history)
    summary = {
        'algorithm': 'PPO',
        'timestamp_start': run_start.strftime('%Y-%m-%d %H:%M:%S'),
        'timestamp_end': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'wall_clock_seconds': round(duration, 1),
        'wall_clock_human': f"{int(duration // 3600)}h {int((duration % 3600) // 60)}m {int(duration % 60)}s",
        'episodes_trained': episodes_trained,
        'training_episodes_budget': TRAINING_EPISODES,
        'early_stopped': stopper.stop_reason is not None,
        'stop_reason': stopper.stop_reason,
        'best_rolling_win_rate': (round(stopper.best_win_rate, 5)
                                  if stopper.best_win_rate != float('-inf') else None),
        'best_episode': stopper.best_episode,
        'final_rolling_win_rate': round(stopper.win_rate, 5),
        'final_rolling_mean_reward': round(stopper.mean_reward, 5),
        'overall_win_rate': round(float(np.mean(win_rates)), 5) if win_rates else None,
        'overall_mean_turns': round(float(np.mean(episode_turns)), 2) if episode_turns else None,
        'artifacts': {
            'metadata': os.path.basename(meta_path),
            'episode_results_csv': os.path.basename(csv_path),
            'episode_log_json': os.path.basename(json_path),
            'earlystop_log_csv': os.path.basename(es_log_path),
            'best_weights': ['ppo_actor_best.weights.h5', 'ppo_critic_best.weights.h5'],
            'final_weights': ['ppo_actor_final.weights.h5', 'ppo_critic_final.weights.h5'],
        },
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    if stopper.stop_reason:
        with open(os.path.join(base_dir, 'ppo_earlystop_report.txt'), 'w') as f:
            f.write(stopper.report(episodes_trained))
    print(f"Training summary written to {os.path.basename(summary_path)} "
          f"({episodes_trained} episodes trained, wall-clock {summary['wall_clock_human']})")
    return best_win_rate

if __name__ == "__main__":
    game = DhumbalGame(num_players=4)
    state_size = 128
    max_action_size = 128  # Supports all possible actions
    ppo_player = LearningBasedAI(0, state_size, max_action_size, model_type='ppo')
    opponents_pool = [
        RuleBasedAI(1, AIStyle.CONSERVATIVE),
        RuleBasedAI(2, AIStyle.AGGRESSIVE),
        RuleBasedAI(3, AIStyle.OPPORTUNISTIC),
        RuleBasedAI(4, AIStyle.BALANCED)
    ]
    print("Training PPO Agent...")
    win_rate = train_agent(ppo_player, game, opponents_pool, BATCH_SIZE)
    print(f"PPO Training Complete. Best Win Rate: {win_rate:.3f}")