import tensorflow as tf
import numpy as np
import random
import csv
import os
import json
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Any
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
NUM_PLAYERS = 5
MAX_PLAYERS = 5
MIN_PLAYERS = 2
HAND_SIZE = 5
JHYAP_THRESHOLD = 10
STARTING_COINS = 1000000
MAX_TURNS = 50
MIN_DISCARD_PILE_SIZE = 2
MAX_PAYMENT = 100
TRAINING_EPISODES = 50000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
MAX_MEMORY = 2000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 100
CONVERGENCE_THRESHOLD = 0.05  # Reverted to original value
CONVERGENCE_EPISODES = 500

class AIStyle(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    OPPORTUNISTIC = "opportunistic"
    BALANCED = "balanced"
    DQN = "dqn"

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
            return hand_value <= 10
        elif self.style == AIStyle.OPPORTUNISTIC:
            my_coins = game_state.player_coins[self.player_id]
            avg_coins = sum(game_state.player_coins) / len(game_state.player_coins)
            return hand_value <= 9 if my_coins < avg_coins else hand_value <= 8
        elif self.style == AIStyle.BALANCED:
            return hand_value <= 8 or (hand_value <= 10 and game_state.turn_count > 10)
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
            round_number=game.round_number + 1,
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
            card_idx = suit_idx * 13 + rank_idx
            hand_encoding[card_idx] = 1
        discard_encoding = np.zeros(52)
        if self.state.discard_pile:
            top_card = self.state.discard_pile[-1]
            suit_idx = self.game.SUITS.index(top_card.suit)
            rank_idx = self.game.RANK_ORDER[top_card.rank]
            discard_encoding[suit_idx * 13 + rank_idx] = 1
        player_one_hot = np.zeros(self.game.num_players)
        player_one_hot[self.state.current_player] = 1
        hand_value = self.game.calculate_hand_value(hand) / (13 * HAND_SIZE)
        turn_norm = self.state.turn_count / MAX_TURNS
        opp_hand_sizes = [len(h) / HAND_SIZE for i, h in enumerate(self.state.hands) if i != self.state.current_player]
        avg_opp_hand_size = np.mean(opp_hand_sizes) if opp_hand_sizes else 0
        my_coins_norm = self.state.player_coins[self.state.current_player] / STARTING_COINS
        avg_opp_coins = np.mean([c / STARTING_COINS for i, c in enumerate(self.state.player_coins) if i != self.state.current_player])
        phase_one_hot = np.zeros(3)
        phase_map = {'call': 0, 'discard': 1, 'pick': 2}
        phase_one_hot[phase_map[self.state.phase]] = 1
        state = np.concatenate([hand_encoding, discard_encoding, player_one_hot, [hand_value, turn_norm, avg_opp_hand_size, my_coins_norm, avg_opp_coins], phase_one_hot])
        assert state.shape[0] == 117, f"State size mismatch: expected 117, got {state.shape[0]}"
        return state

    def get_actions(self) -> List[Any]:
        actions = []
        if self.state.phase == 'call':
            actions = [False, True] if self.game.can_call_jhyap(self.state.hands[self.state.current_player]) else [False]
        elif self.state.phase == 'discard':
            hand = self.state.hands[self.state.current_player]
            if not hand:
                return []
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
                            combo_list = list(combo)
                            if self.game.validate_sequence(combo_list):
                                actions.append(combo_list)
        elif self.state.phase == 'pick':
            if len(self.state.hands[self.state.current_player]) < HAND_SIZE:
                actions = ['deck']
                if self.state.discard_pile:
                    actions.append('discard')
        return actions

    def action_to_index(self, action: Any) -> int:
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

    def index_to_action(self, index: int) -> Any:
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

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, dict]:
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
                    non_caller_min = [i for i in min_value_players if i != caller]
                    if non_caller_min:
                        winner = non_caller_min[0]
                    else:
                        winner = caller
                    successful_call = (caller == winner)
                    if successful_call:
                        reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != caller)
                    else:
                        total_payment = sum(min(v, MAX_PAYMENT) for v in hand_values)
                        reward = -total_payment
                    new_state.done = True
                    new_state.winner = winner
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
                if not new_state.deck and len(new_state.discard_pile) >= MIN_DISCARD_PILE_SIZE:
                    top = new_state.discard_pile.pop() if new_state.discard_pile else None
                    random.shuffle(new_state.discard_pile)
                    new_state.deck.extend(new_state.discard_pile[:])
                    new_state.discard_pile = [top] if top else []
                if new_state.deck:
                    card = new_state.deck.pop()
                    new_state.hands[player].append(card)
                    new_hand_value = self.game.calculate_hand_value(new_state.hands[player])
                    reward = (old_hand_value - new_hand_value) / 13.0 if new_hand_value < old_hand_value else -0.1
                else:
                    new_state.done = True
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    min_value = min(hand_values)
                    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                    non_caller_min = [i for i in min_value_players if i != player]
                    new_state.winner = non_caller_min[0] if non_caller_min else player
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
                    non_caller_min = [i for i in min_value_players if i != player]
                    new_state.winner = non_caller_min[0] if non_caller_min else player
                    if new_state.winner == player:
                        reward += sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != player)
                    else:
                        reward += -hand_values[player]

        self.set_state(new_state)
        log_entry['reward'] = reward
        log_entry['done'] = done
        log_entry['state_after'] = new_state.to_dict()
        return self.encode_state(), reward, done, log_entry

class DQN:
    def __init__(self, state_size: int, max_action_size: int):
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.memory = deque(maxlen=MAX_MEMORY)
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        self.step_count = 0

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.max_action_size, activation='linear')
        ])
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray, env: DhumbalEnv) -> int:
        action_space_size = len(env.get_actions())
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        if np.random.rand() <= self.epsilon:
            return random.randrange(action_space_size)
        with tf.device('/GPU:0'):
            q_values = self.model(state, training=False)[0].numpy()
        q_values = q_values[:action_space_size]
        valid_mask = np.ones(action_space_size)
        q_values_masked = q_values * valid_mask + (1 - valid_mask) * -np.inf
        return np.argmax(q_values_masked)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch], dtype=np.float32)
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch], dtype=np.float32)
        dones = np.array([m[4] for m in minibatch])

        with tf.device('/GPU:0'):
            q_values_next = self.target_model(next_states, training=False).numpy()
            max_q_next = np.max(q_values_next, axis=1)
            targets = rewards + GAMMA * max_q_next * (1 - dones)
            with tf.GradientTape() as tape:
                q_values = self.model(states, training=True)
                indices = tf.stack([tf.range(batch_size), actions], axis=1)
                q_values_selected = tf.gather_nd(q_values, indices)
                loss = tf.reduce_mean(tf.square(targets - q_values_selected))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.epsilon = max(self.epsilon_min, self.epsilon * EPSILON_DECAY)

class LearningBasedAI:
    def __init__(self, player_id: int, state_size: int, max_action_size: int, model_type: str = 'dqn'):
        self.player_id = player_id
        self.name = f"AI_dqn_{player_id}"
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.model_type = model_type
        self.agent = DQN(state_size, max_action_size)

    def choose_discard(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> List[Card]:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'discard'
        state = env.encode_state()
        action_idx = self.agent.act(state, env)
        action = env.index_to_action(action_idx)
        return action if isinstance(action, list) and game.validate_discard(action) else [max(hand, key=lambda x: x.value)] if hand else []

    def should_pick_from_discard(self, discard_pile: List[Card], current_hand: List[Card], game_state: GameState, game: DhumbalGame) -> Tuple[bool, Optional[Card]]:
        if len(current_hand) >= HAND_SIZE: return False, None
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'pick'
        state = env.encode_state()
        action_idx = self.agent.act(state, env)
        action = env.index_to_action(action_idx)
        return (True, discard_pile[-1]) if action == 'discard' and discard_pile else (False, None)

    def should_call_jhyap(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> bool:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'call'
        state = env.encode_state()
        action_idx = self.agent.act(state, env)
        action = env.index_to_action(action_idx)
        return action

def simulate_opponent_turn(env: DhumbalEnv, opponent: RuleBasedAI):
    player = env.state.current_player
    if env.state.phase == 'call':
        action = opponent.should_call_jhyap(env.state.hands[player], env.state, env.game)
    elif env.state.phase == 'discard':
        action = opponent.choose_discard(env.state.hands[player], env.state, env.game)
    else:  # pick
        should_pick, _ = opponent.should_pick_from_discard(env.state.discard_pile, env.state.hands[player], env.state, env.game)
        action = 'discard' if should_pick else 'deck'
    next_state, _, done, log_entry = env.step(action)
    return next_state, done, log_entry

def train_agent(agent: LearningBasedAI, game: DhumbalGame, opponents: List[RuleBasedAI], batch_size: int):
    env = DhumbalEnv(game, [agent] + opponents, agent.player_id)
    win_rates = []
    episode_turns = []
    best_win_rate = 0
    best_model = None
    csv_path = 'dqn_training_results.csv'
    json_path = 'dqn_training_log.json'
    json_log = []

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Win', 'Turns', 'Reward'])
        csvfile.flush()
        for episode in range(TRAINING_EPISODES):
            state = env.reset()
            episode_reward = 0
            turns = 0
            episode_log = {
                'episode': episode + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'turns': [],
                'total_reward': 0,
                'winner': None,
                'epsilon': agent.agent.epsilon
            }
            while not env.state.done:
                if env.state.current_player == agent.player_id:
                    action_idx = agent.agent.act(state, env)
                    action = env.index_to_action(action_idx)
                    next_state, reward, done, log_entry = env.step(action)
                    agent.agent.remember(state, action_idx, reward, next_state, done)
                    episode_reward += reward
                    state = next_state
                else:
                    opponent_idx = env.state.current_player - 1 if env.state.current_player > agent.player_id else env.state.current_player
                    opponent = opponents[opponent_idx]
                    next_state, done, log_entry = simulate_opponent_turn(env, opponent)
                episode_log['turns'].append(log_entry)
                turns += 1
                if done:
                    break
            agent.agent.replay(batch_size)
            if (agent.agent.step_count + 1) % TARGET_UPDATE_FREQ == 0:
                agent.agent.update_target_model()
            agent.agent.step_count += 1

            episode_log['total_reward'] = episode_reward
            episode_log['winner'] = env.state.winner
            episode_log['epsilon'] = agent.agent.epsilon
            json_log.append(episode_log)

            win = 1 if env.state.winner == agent.player_id else 0
            writer.writerow([episode + 1, win, turns, episode_reward])
            csvfile.flush()
            win_rates.append(win)
            episode_turns.append(turns)

            print(f"Episode {episode + 1}/{TRAINING_EPISODES}: Win={win}, Turns={turns}, Reward={episode_reward:.2f}, Epsilon={agent.agent.epsilon:.4f}")

            if len(win_rates) >= CONVERGENCE_EPISODES:
                recent_win_rate = np.mean(win_rates[-CONVERGENCE_EPISODES:])
                if len(win_rates) >= 2 * CONVERGENCE_EPISODES and abs(recent_win_rate - np.mean(win_rates[-2*CONVERGENCE_EPISODES:-CONVERGENCE_EPISODES])) < CONVERGENCE_THRESHOLD:
                    break
                if recent_win_rate > best_win_rate:
                    best_win_rate = recent_win_rate
                    best_model = agent.agent.model.get_weights()
                    agent.agent.model.save_weights(f'dqn_model_ep{episode+1}.weights.h5')
                    agent.agent.target_model.save_weights(f'dqn_target_model_ep{episode+1}.weights.h5')

            with open(json_path, 'w') as f:
                json.dump(json_log, f, indent=2)
            
            agent.agent.model.save_weights(f'dqn_model_ep_last.weights.h5')
            agent.agent.target_model.save_weights(f'dqn_target_ model_ep_last.weights.h5')


    if best_model:
        agent.agent.model.set_weights(best_model)
    return best_win_rate

if __name__ == "__main__":
    game = DhumbalGame(num_players=NUM_PLAYERS)
    state_size = 52 + 52 + 5 + 5 + 3  # Hand + discard + player_one_hot + features + phase = 117
    max_action_size = 128
    dqn_player = LearningBasedAI(0, state_size, max_action_size, model_type='dqn')
    opponents = [
        RuleBasedAI(1, AIStyle.CONSERVATIVE),
        RuleBasedAI(2, AIStyle.AGGRESSIVE),
        RuleBasedAI(3, AIStyle.OPPORTUNISTIC),
        RuleBasedAI(4, AIStyle.BALANCED)
    ]
    win_rate = train_agent(dqn_player, game, opponents, BATCH_SIZE)
    print(f"DQN Training Complete. Best Win Rate: {win_rate:.3f}")