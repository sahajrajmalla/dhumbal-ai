import tensorflow as tf
import numpy as np
import random
import csv
import os
import json
from collections import deque
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from tensorflow.keras import models, layers, optimizers
from datetime import datetime

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
STARTING_COINS = 10000
MAX_TURNS = 50
MIN_DISCARD_PILE_SIZE = 2
MAX_PAYMENT = 100
TRAINING_EPISODES = 10000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
MAX_MEMORY = 2000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 100
CONVERGENCE_THRESHOLD = 0.02
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

    def validate_discard(self, cards: List[Card]) -> bool:
        return len(cards) == 1

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
        hand_value = sum(card.value for card in hand)
        if self.style == AIStyle.CONSERVATIVE:
            return [min(hand, key=lambda x: x.value)] if hand_value <= JHYAP_THRESHOLD else [max(hand, key=lambda x: x.value)]
        elif self.style == AIStyle.AGGRESSIVE:
            return [min(hand, key=lambda x: x.value)]
        elif self.style == AIStyle.OPPORTUNISTIC:
            return [min(hand, key=lambda x: x.value)] if hand_value <= JHYAP_THRESHOLD else [random.choice(hand)]
        elif self.style == AIStyle.BALANCED:
            mid_cards = [c for c in hand if 4 <= c.value <= 8]
            return [random.choice(mid_cards)] if mid_cards else [max(hand, key=lambda x: x.value)]
        return [random.choice(hand)]

    def should_pick_from_discard(self, available_cards: List[Card], current_hand: List[Card], game_state: GameState, game: DhumbalGame) -> Tuple[bool, Optional[Card]]:
        if not available_cards or len(current_hand) >= HAND_SIZE: return False, None
        current_value = sum(card.value for card in current_hand)
        if self.style == AIStyle.CONSERVATIVE:
            pickup_threshold = 3 if current_value <= 12 else 5
            good_cards = [card for card in available_cards if card.value <= pickup_threshold]
            return (True, min(good_cards, key=lambda x: x.value)) if good_cards else (False, None)
        elif self.style == AIStyle.AGGRESSIVE:
            good_cards = [card for card in available_cards if card.value <= 5]
            return (True, min(good_cards, key=lambda x: x.value)) if good_cards else (False, None)
        elif self.style == AIStyle.OPPORTUNISTIC:
            top_card = available_cards[-1]
            return (True, top_card) if top_card.value <= 5 and current_value > JHYAP_THRESHOLD else (False, None)
        elif self.style == AIStyle.BALANCED:
            good_cards = [card for card in available_cards if card.value <= 4]
            return (True, min(good_cards, key=lambda x: x.value)) if good_cards else (False, None)
        return (True, random.choice(available_cards)) if random.random() < 0.5 else (False, None)

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
            player_coins=game.player_coins.copy(),
            turn_count=0,
            phase='call'
        )
        print(f"Reset environment: Round {self.state.round_number}, Player {self.state.current_player}, Deck size {self.state.deck_size}, Initial hands {[self.game.calculate_hand_value(hand) for hand in self.hands]}")
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
        return np.concatenate([hand_encoding, discard_encoding])

    def get_actions(self) -> List:
        actions = []
        if self.state.phase == 'call':
            actions = [True, False] if self.game.can_call_jhyap(self.state.hands[self.state.current_player]) else [False]
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
            all_combinations.sort(key=lambda x: tuple(str(c) for c in x) if x else ())
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

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        new_state = self.state.copy()
        reward = 0.0
        done = False
        player = new_state.current_player
        hand_size = len(new_state.hands[player])
        hand_value = self.game.calculate_hand_value(new_state.hands[player])
        print(f"Turn {new_state.turn_count+1}, Player {player}, Phase {new_state.phase}, Hand size {hand_size}, Hand value {hand_value}, Action {action}")
        log_entry = {
            'turn': new_state.turn_count + 1,
            'player': player,
            'phase': new_state.phase,
            'hand_size': hand_size,
            'hand_value': hand_value,
            'action': str(action) if isinstance(action, list) else action,
            'state_before': new_state.to_dict()
        }

        if new_state.phase == 'call':
            if action and self.game.can_call_jhyap(new_state.hands[player]):
                hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                caller = player
                caller_value = hand_values[caller]
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
                print(f"Episode ended: Jhyap call, Hand values {hand_values}, Winner {new_state.winner}, Reward {reward}")
            elif action and not self.game.can_call_jhyap(new_state.hands[player]):
                reward = -100.0
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                print(f"Episode ended: Invalid Jhyap call, Winner {new_state.winner}, Reward {reward}")
            else:
                new_state.phase = 'discard'
                print(f"Player {player} chose not to call Jhyap, moving to discard phase")

        elif new_state.phase == 'discard':
            if self.game.validate_discard(action) and all(card in new_state.hands[player] for card in action):
                for card in action:
                    new_state.hands[player].remove(card)
                new_state.discard_pile.extend(action)
                new_state.phase = 'pick'
                print(f"Player {player} discarded {action}, moving to pick phase")
            else:
                reward = -100.0
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                print(f"Episode ended: Invalid discard, Winner {new_state.winner}, Reward {reward}")

        elif new_state.phase == 'pick':
            if len(new_state.hands[player]) >= HAND_SIZE:
                reward = -100.0
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                print(f"Episode ended: Hand size limit exceeded, Winner {new_state.winner}, Reward {reward}")
            elif action == 'discard' and new_state.discard_pile:
                card = new_state.discard_pile.pop()
                new_state.hands[player].append(card)
                new_state.deck_size = len(self.deck)
                print(f"Player {player} picked {card} from discard pile, new hand value {self.game.calculate_hand_value(new_state.hands[player])}")
            elif action == 'deck':
                if not self.deck and len(new_state.discard_pile) >= MIN_DISCARD_PILE_SIZE:
                    top = new_state.discard_pile.pop() if new_state.discard_pile else None
                    random.shuffle(new_state.discard_pile)
                    self.deck.extend(new_state.discard_pile[:])
                    new_state.discard_pile = [top] if top else []
                    new_state.deck_size = len(self.deck)
                    print(f"Shuffled discard pile into deck, new deck size {new_state.deck_size}")
                if self.deck:
                    card = self.deck.pop()
                    new_state.hands[player].append(card)
                    new_state.deck_size = len(self.deck)
                    print(f"Player {player} picked {card} from deck, new hand value {self.game.calculate_hand_value(new_state.hands[player])}")
                else:
                    new_state.done = True
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    new_state.winner = hand_values.index(min(hand_values))
                    reward = -sum(min(v, MAX_PAYMENT) for v in hand_values) if new_state.winner != player else sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != player)
                    print(f"Episode ended: Deck exhausted, Hand values {hand_values}, Winner {new_state.winner}, Reward {reward}")
            else:
                reward = -100.0
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                print(f"Episode ended: Invalid pick, Winner {new_state.winner}, Reward {reward}")
            if not new_state.done:
                new_state.turn_count += 1
                new_state.current_player = (new_state.current_player + 1) % self.game.num_players
                new_state.phase = 'call'
                print(f"Advancing to next turn: Turn {new_state.turn_count+1}, Next player {new_state.current_player}")
                if new_state.turn_count >= MAX_TURNS:
                    new_state.done = True
                    hand_values = [self.game.calculate_hand_value(hand) for hand in new_state.hands]
                    new_state.winner = hand_values.index(min(hand_values))
                    reward = -sum(min(v, MAX_PAYMENT) for v in hand_values) if new_state.winner != player else sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != player)
                    print(f"Episode ended: Max turns reached, Hand values {hand_values}, Winner {new_state.winner}, Reward {reward}")

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
            layers.Dense(128, input_shape=(self.state_size,), activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.max_action_size, activation='linear')
        ])
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("Updated target model weights")

    def act(self, state: np.ndarray, env: DhumbalEnv) -> int:
        action_space_size = env.get_action_space_size()
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(action_space_size)
            print(f"Exploration: Selected random action {action} (epsilon: {self.epsilon:.3f})")
            return action
        with tf.device('/GPU:0'):
            q_values = self.model(state, training=False)[0].numpy()
        q_values = q_values[:action_space_size]
        action = np.argmax(q_values)
        print(f"Exploitation: Selected action {action} with Q-values {q_values[:action_space_size]}")
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        print(f"Stored transition: Action {action}, Reward {reward}, Done {done}")

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            print(f"Memory size {len(self.memory)} is less than batch size {batch_size}, skipping replay")
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch], dtype=np.float32)
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch], dtype=np.float32)
        dones = np.array([m[4] for m in minibatch])

        with tf.device('/GPU:0'):
            targets = self.model(states, training=False).numpy()
            target_next = self.target_model(next_states, training=False).numpy()
            for i in range(batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + GAMMA * np.max(target_next[i])
            
            with tf.GradientTape() as tape:
                q_values = self.model(states, training=True)
                q_values_selected = tf.reduce_sum(
                    q_values * tf.one_hot(actions, self.max_action_size), axis=1)
                loss = tf.reduce_mean(tf.square(targets - q_values))
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            print(f"Replay: Trained on batch of {batch_size}, Loss {loss:.4f}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print(f"Epsilon decayed to {self.epsilon:.3f}")

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
        print(f"Choose discard: Action index {action_idx}, Selected cards {action}")
        return action if isinstance(action, list) and game.validate_discard(action) else [max(hand, key=lambda x: x.value)] if hand else []

    def should_pick_from_discard(self, discard_pile: List[Card], current_hand: List[Card], game_state: GameState, game: DhumbalGame) -> Tuple[bool, Optional[Card]]:
        if len(current_hand) >= HAND_SIZE: return False, None
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'pick'
        state = env.encode_state()
        action_idx = self.agent.act(state, env)
        action = env.index_to_action(action_idx)
        print(f"Pick decision: Action index {action_idx}, Selected {action}")
        return (True, discard_pile[-1]) if action == 'discard' and discard_pile else (False, None)

    def should_call_jhyap(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> bool:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'call'
        state = env.encode_state()
        action_idx = self.agent.act(state, env)
        action = env.index_to_action(action_idx)
        print(f"Jhyap decision: Action index {action_idx}, Call Jhyap: {action}")
        return action

def train_agent(agent: LearningBasedAI, game: DhumbalGame, opponents: List[RuleBasedAI], batch_size: int):
    env = DhumbalEnv(game, [agent] + opponents, agent.player_id)
    win_rates = []
    episode_turns = []
    best_win_rate = 0
    best_model = None
    os.makedirs('/content', exist_ok=True)
    csv_path = '/content/dqn_training_results.csv'
    json_path = '/content/dqn_training_log.json'
    json_log = []

    try:
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
                print(f"\nStarting Episode {episode+1}/{TRAINING_EPISODES}")
                while not env.state.done:
                    action_idx = agent.agent.act(state, env)
                    action = env.index_to_action(action_idx)
                    next_state, reward, done, log_entry = env.step(action)
                    agent.agent.remember(state, action_idx, reward, next_state, done)
                    agent.agent.replay(batch_size)
                    episode_reward += reward
                    turns += 1
                    state = next_state
                    episode_log['turns'].append(log_entry)
                    if done:
                        break
                    if (agent.agent.step_count + 1) % TARGET_UPDATE_FREQ == 0:
                        agent.agent.update_target_model()
                    agent.agent.step_count += 1
                
                episode_log['total_reward'] = episode_reward
                episode_log['winner'] = env.state.winner
                episode_log['epsilon'] = agent.agent.epsilon
                json_log.append(episode_log)
                
                print(f"Episode {episode+1} ended after {turns} turns, Reward {episode_reward}, Winner {env.state.winner}")
                writer.writerow([episode+1, 1 if env.state.winner == agent.player_id else 0, turns, episode_reward])
                csvfile.flush()
                win_rates.append(1 if env.state.winner == agent.player_id else 0)
                episode_turns.append(turns)
                
                if episode == 0 or (episode + 1) % 500 == 0:
                    agent.agent.model.save_weights('/content/dqn_model.weights.h5')
                    agent.agent.target_model.save_weights('/content/dqn_target_model.weights.h5')
                    print(f"Saved DQN models at episode {episode+1}")
                    if os.path.exists('/content/dqn_model.weights.h5') and os.path.exists('/content/dqn_target_model.weights.h5'):
                        print(f"Confirmed models saved: /content/dqn_model.weights.h5 and /content/dqn_target_model.weights.h5")
                    else:
                        print(f"Warning: Model files not found after saving attempt")
                
                if len(win_rates) >= CONVERGENCE_EPISODES:
                    recent_win_rate = np.mean(win_rates[-CONVERGENCE_EPISODES:])
                    if len(win_rates) >= 2 * CONVERGENCE_EPISODES and abs(recent_win_rate - np.mean(win_rates[-2*CONVERGENCE_EPISODES:-CONVERGENCE_EPISODES])) < CONVERGENCE_THRESHOLD:
                        print(f"Convergence reached at episode {episode+1} with win rate {recent_win_rate:.3f}")
                        break
                    if recent_win_rate > best_win_rate:
                        best_win_rate = recent_win_rate
                        best_model = agent.agent.model.get_weights()
                        agent.agent.model.save_weights('/content/dqn_model.weights.h5')
                        agent.agent.target_model.save_weights('/content/dqn_target_model.weights.h5')
                        print(f"Saved best DQN model at episode {episode+1} with win rate {best_win_rate:.3f}")
                        if os.path.exists('/content/dqn_model.weights.h5') and os.path.exists('/content/dqn_target_model.weights.h5'):
                            print(f"Confirmed best models saved: /content/dqn_model.weights.h5 and /content/dqn_target_model.weights.h5")
                
                if episode % 500 == 0:
                    print(f"Training {agent.name}, Episode {episode+1}/{TRAINING_EPISODES}, Recent Win Rate: {np.mean(win_rates[-500:]) if win_rates else 0:.3f}, Avg Turns: {np.mean(episode_turns[-500:]) if episode_turns else 0:.1f}, Epsilon: {agent.agent.epsilon:.3f}")
                
                # Save JSON log periodically
                if episode % 100 == 0 or done:
                    with open(json_path, 'w') as f:
                        json.dump(json_log, f, indent=2)
                    if os.path.exists(json_path):
                        file_size = os.path.getsize(json_path)
                        print(f"JSON log saved to {json_path}, File size: {file_size} bytes")
            
            if best_model:
                agent.agent.model.set_weights(best_model)
                print("Restored best model weights")
            agent.agent.model.save_weights('/content/dqn_model.weights.h5')
            agent.agent.target_model.save_weights('/content/dqn_target_model.weights.h5')
            print("Final DQN models saved to /content/dqn_model.weights.h5 and /content/dqn_target_model.weights.h5")
            if os.path.exists('/content/dqn_model.weights.h5') and os.path.exists('/content/dqn_target_model.weights.h5'):
                print(f"Confirmed final models saved: /content/dqn_model.weights.h5 and /content/dqn_target_model.weights.h5")
            if os.path.exists(csv_path):
                file_size = os.path.getsize(csv_path)
                print(f"Training results saved to {csv_path}, File size: {file_size} bytes")
                with open(csv_path, 'r') as f:
                    content = f.read()
                    print(f"CSV content preview: {content[:200]}...")
            if os.path.exists(json_path):
                file_size = os.path.getsize(json_path)
                print(f"Final JSON log saved to {json_path}, File size: {file_size} bytes")
                with open(json_path, 'r') as f:
                    content = f.read()
                    print(f"JSON log preview: {content[:200]}...")
    except Exception as e:
        print(f"Error during training: {e}")
        with open(json_path, 'w') as f:
            json.dump(json_log, f, indent=2)
        raise
    
    return best_win_rate

if __name__ == "__main__":
    game = DhumbalGame(num_players=NUM_PLAYERS)
    state_size = 52 + 52  # Hand + discard pile encoding
    max_action_size = 5   # Max single-card discards
    dqn_player = LearningBasedAI(0, state_size, max_action_size, model_type='dqn')
    opponents = [
        RuleBasedAI(1, AIStyle.CONSERVATIVE),
        RuleBasedAI(2, AIStyle.AGGRESSIVE),
        RuleBasedAI(3, AIStyle.OPPORTUNISTIC),
        RuleBasedAI(4, AIStyle.BALANCED)
    ]
    print("Training DQN Agent...")
    win_rate = train_agent(dqn_player, game, opponents, BATCH_SIZE)
    print(f"DQN Training Complete. Best Win Rate: {win_rate:.3f}")