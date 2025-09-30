import tensorflow as tf
import numpy as np
import random
import csv
import os
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from tensorflow.keras import models, layers
from datetime import datetime
import math

# Configure TensorFlow for T4 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
EVALUATION_EPISODES = 1000
MCTS_ITERATIONS = 100
ISMCTS_SAMPLES = 5
UCB_C = 1.4  # Exploration constant for UCB1

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
        print(f"Reset environment: Round {self.state.round_number}, Deck size {self.state.deck_size}, Initial hands {[self.game.calculate_hand_value(hand) for hand in self.hands]}")
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
        initial_coins = new_state.player_coins[player]
        log_entry = {
            'turn': new_state.turn_count + 1,
            'player': player,
            'phase': new_state.phase,
            'hand_size': hand_size,
            'hand_value': hand_value,
            'action': [{'suit': card.suit, 'rank': card.rank} for card in action] if isinstance(action, list) else action,
            'state_before': new_state.to_dict()
        }
        print(f"Turn {new_state.turn_count+1}, Player {player}, Phase {new_state.phase}, Hand size {hand_size}, Hand value {hand_value}, Action {action}")

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
                print(f"Episode ended: Jhyap call, Hand values {hand_values}, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
            elif action and not self.game.can_call_jhyap(new_state.hands[player]):
                reward = -100.0
                coin_change = -100
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                new_state.player_coins[player] += coin_change
                print(f"Episode ended: Invalid Jhyap call, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
            else:
                new_state.phase = 'discard'
                print(f"Player {player} chose not to call Jhyap, moving to discard phase")

        elif new_state.phase == 'discard':
            if self.game.validate_discard(action) and all(isinstance(card, Card) and card in new_state.hands[player] for card in action):
                for card in action:
                    new_state.hands[player].remove(card)
                new_state.discard_pile.extend(action)
                new_state.phase = 'pick'
                print(f"Player {player} discarded {action}, moving to pick phase")
            else:
                reward = -100.0
                coin_change = -100
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                new_state.player_coins[player] += coin_change
                print(f"Episode ended: Invalid discard, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")

        elif new_state.phase == 'pick':
            if len(new_state.hands[player]) >= HAND_SIZE:
                reward = -100.0
                coin_change = -100
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                new_state.player_coins[player] += coin_change
                print(f"Episode ended: Hand size limit exceeded, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
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
                    coin_change = reward
                    for i in range(self.game.num_players):
                        if i == new_state.winner:
                            new_state.player_coins[i] += coin_change
                        else:
                            new_state.player_coins[i] -= min(hand_values[i], MAX_PAYMENT)
                    print(f"Episode ended: Deck exhausted, Hand values {hand_values}, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
            else:
                reward = -100.0
                coin_change = -100
                new_state.done = True
                new_state.winner = (player + 1) % self.game.num_players
                new_state.player_coins[player] += coin_change
                print(f"Episode ended: Invalid pick, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")
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
                    coin_change = reward
                    for i in range(self.game.num_players):
                        if i == new_state.winner:
                            new_state.player_coins[i] += coin_change
                        else:
                            new_state.player_coins[i] -= min(hand_values[i], MAX_PAYMENT)
                    print(f"Episode ended: Max turns reached, Hand values {hand_values}, Winner {new_state.winner}, Reward {reward}, Coin change {coin_change}")

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
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.max_action_size, activation='softmax')(x)
        return models.Model(inputs, outputs)

    def build_critic(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        return models.Model(inputs, outputs)

    def build_dqn_model(self):
        model = models.Sequential([
            layers.Dense(128, input_shape=(self.state_size,), activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.max_action_size, activation='linear')
        ])
        return model

    def load_ppo_models(self):
        ppo_actor_path = '/content/ppo_actor.weights.h5'
        ppo_critic_path = '/content/ppo_critic.weights.h5'
        if not (os.path.exists(ppo_actor_path) and os.path.exists(ppo_critic_path)):
            raise FileNotFoundError(f"PPO model files not found: {ppo_actor_path}, {ppo_critic_path}")
        self.actor.load_weights(ppo_actor_path)
        self.critic.load_weights(ppo_critic_path)
        print(f"Loaded PPO models: {ppo_actor_path}, {ppo_critic_path}")

    def load_dqn_model(self):
        dqn_model_path = '/content/dqn_model.weights.h5'
        if not os.path.exists(dqn_model_path):
            raise FileNotFoundError(f"DQN model file not found: {dqn_model_path}")
        self.model.load_weights(dqn_model_path)
        print(f"Loaded DQN model: {dqn_model_path}")

    def get_policy(self, state: np.ndarray, env: DhumbalEnv) -> np.ndarray:
        action_space_size = env.get_action_space_size()
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        with tf.device('/GPU:0'):
            if self.model_type in ['ppo_mcts', 'ppo_ismcts']:
                probs = self.actor(state, training=False)[0].numpy()
            else:  # dqn_mcts, dqn_ismcts
                q_values = self.model(state, training=False)[0].numpy()
                probs = np.exp(q_values) / np.sum(np.exp(q_values) + 1e-10)  # Softmax Q-values
        probs = probs[:action_space_size]
        probs = probs / np.sum(probs + 1e-10)  # Normalize
        return probs

    def get_value(self, state: np.ndarray) -> float:
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        with tf.device('/GPU:0'):
            if self.model_type in ['ppo_mcts', 'ppo_ismcts']:
                value = self.critic(state, training=False).numpy()[0][0]
            else:  # dqn_mcts, dqn_ismcts
                q_values = self.model(state, training=False)[0].numpy()
                value = np.max(q_values)  # Use max Q-value as state value
        return value

    def mcts_search(self, env: DhumbalEnv, root_state: GameState) -> Tuple[int, any]:
        if not isinstance(root_state, GameState):
            raise TypeError(f"Expected GameState for root_state, got {type(root_state)}")
        root = MCTSNode(root_state, env) if self.model_type in ['ppo_mcts', 'dqn_mcts'] else ISMCTSNode(root_state, env)
        root_state_encoded = env.encode_state()

        # Set prior probabilities for root children
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

        for _ in range(MCTS_ITERATIONS):
            node = root
            # Selection
            while not node.is_leaf() and node.is_fully_expanded():
                node = node.select_child()
            # Expansion
            if node and not node.is_leaf() and not node.state.done:
                node = node.expand()
            # Simulation
            if node and not node.state.done:
                if self.model_type in ['ppo_mcts', 'dqn_mcts']:
                    value = self.get_value(node.env.encode_state())
                else:
                    values = []
                    for _ in range(ISMCTS_SAMPLES):
                        sim_env = DhumbalEnv(node.env.game, node.env.ai_players, node.state.current_player)
                        sim_env.set_state(node.state)
                        sim_value = self.get_value(sim_env.encode_state())
                        values.append(sim_value)
                    value = np.mean(values)
            else:
                value = 1.0 if node.state.winner == root_state.current_player else -1.0 if node.state.winner is not None else 0.0
                value *= 100  # Scale to match game rewards
            # Backpropagation
            while node:
                node.update(value)
                node = node.parent

        best_child = max(root.children, key=lambda c: c.visits) if root.children else None
        if best_child:
            action_idx = env.action_to_index(best_child.action)
            print(f"{self.model_type.upper()} selected action {action_idx} (visits: {best_child.visits}, value: {best_child.total_value / best_child.visits:.2f})")
            return action_idx, best_child.action
        action_idx = random.randint(0, env.get_action_space_size() - 1)
        action = env.index_to_action(action_idx)
        print(f"{self.model_type.upper()} fallback to random action {action_idx}")
        return action_idx, action

class LearningBasedAI:
    def __init__(self, player_id: int, state_size: int, max_action_size: int, model_type: str):
        self.player_id = player_id
        self.name = f"AI_{model_type}_{player_id}"
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.model_type = model_type
        self.agent = HybridAgent(state_size, max_action_size, model_type)

    def choose_discard(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> List[Card]:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'discard'
        _, action = self.agent.mcts_search(env, game_state)
        return action if isinstance(action, list) and game.validate_discard(action) else [max(hand, key=lambda x: x.value)] if hand else []

    def should_pick_from_discard(self, discard_pile: List[Card], current_hand: List[Card], game_state: GameState, game: DhumbalGame) -> Tuple[bool, Optional[Card]]:
        if len(current_hand) >= HAND_SIZE: return False, None
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'pick'
        _, action = self.agent.mcts_search(env, game_state)
        return (True, discard_pile[-1]) if action == 'discard' and discard_pile else (False, None)

    def should_call_jhyap(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> bool:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'call'
        _, action = self.agent.mcts_search(env, game_state)
        return action

def evaluate_hybrid_agents(game: DhumbalGame):
    state_size = 52 + 52  # Hand + discard pile encoding
    max_action_size = 5   # Max single-card discards
    hybrid_agents = [
        LearningBasedAI(0, state_size, max_action_size, model_type='ppo_mcts'),
        LearningBasedAI(1, state_size, max_action_size, model_type='dqn_mcts'),
        LearningBasedAI(2, state_size, max_action_size, model_type='ppo_ismcts'),
        LearningBasedAI(3, state_size, max_action_size, model_type='dqn_ismcts')
    ]
    results = {agent.model_type: {'wins': [], 'rewards': [], 'turns': [], 'coin_changes': [], 'hand_values': []} for agent in hybrid_agents}
    os.makedirs('/content', exist_ok=True)
    csv_path = '/content/hybrid_tournament_results.csv'
    json_path = '/content/hybrid_tournament_logs.json'
    json_log = []

    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Agent_Type', 'Player_ID', 'Win', 'Turns', 'Reward', 'Coin_Change', 'Hand_Value'])
            csvfile.flush()

            print("\nEvaluating Hybrid Agents in Tournament...")
            env = DhumbalEnv(game, hybrid_agents, 0)
            for episode in range(EVALUATION_EPISODES):
                state = env.reset()
                episode_rewards = {agent.model_type: 0 for agent in hybrid_agents}
                episode_coin_changes = {agent.model_type: 0 for agent in hybrid_agents}
                turns = 0
                episode_log = {
                    'episode': episode + 1,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'turns': [],
                    'winner': None,
                    'hand_values': []
                }
                print(f"\nStarting Episode {episode+1}/{EVALUATION_EPISODES}")

                while not env.state.done:
                    current_player = env.state.current_player
                    if not (0 <= current_player < NUM_PLAYERS):
                        raise ValueError(f"Invalid current_player index: {current_player}")
                    player = hybrid_agents[current_player]
                    _, action = player.agent.mcts_search(env, env.state)

                    next_state, reward, done, log_entry = env.step(action, current_player)
                    episode_rewards[player.model_type] += reward
                    episode_coin_changes[player.model_type] += log_entry['coin_change']
                    turns += 1
                    state = next_state
                    episode_log['turns'].append(log_entry)
                    if done:
                        break

                episode_log['winner'] = env.state.winner
                episode_log['hand_values'] = [game.calculate_hand_value(h) for h in env.state.hands]
                json_log.append(episode_log)
                for agent in hybrid_agents:
                    win = 1 if env.state.winner == agent.player_id else 0
                    final_hand_value = game.calculate_hand_value(env.state.hands[agent.player_id])
                    results[agent.model_type]['wins'].append(win)
                    results[agent.model_type]['rewards'].append(episode_rewards[agent.model_type])
                    results[agent.model_type]['turns'].append(turns)
                    results[agent.model_type]['coin_changes'].append(episode_coin_changes[agent.model_type])
                    results[agent.model_type]['hand_values'].append(final_hand_value)
                    writer.writerow([
                        episode + 1,
                        agent.model_type,
                        agent.player_id,
                        win,
                        turns,
                        episode_rewards[agent.model_type],
                        episode_coin_changes[agent.model_type],
                        final_hand_value
                    ])
                csvfile.flush()
                print(f"Episode {episode+1} ended after {turns} turns, Winner: Player {env.state.winner} ({hybrid_agents[env.state.winner].model_type.upper()})")

                if episode % 100 == 0 or episode == EVALUATION_EPISODES - 1:
                    with open(json_path, 'w') as f:
                        json.dump(json_log, f, indent=2)
                    if os.path.exists(json_path):
                        file_size = os.path.getsize(json_path)
                        print(f"JSON log saved to {json_path}, File size: {file_size} bytes")

                if episode % 100 == 0:
                    for agent in hybrid_agents:
                        win_rate = np.mean(results[agent.model_type]['wins'][-100:])
                        avg_reward = np.mean(results[agent.model_type]['rewards'][-100:])
                        avg_coin_change = np.mean(results[agent.model_type]['coin_changes'][-100:])
                        avg_hand_value = np.mean(results[agent.model_type]['hand_values'][-100:])
                        print(f"Episode {episode+1}/{EVALUATION_EPISODES}, "
                              f"{agent.model_type.upper()} (Player {agent.player_id}) "
                              f"Win Rate: {win_rate:.3f}, "
                              f"Avg Reward: {avg_reward:.1f}, "
                              f"Avg Coin Change: {avg_coin_change:.1f}, "
                              f"Avg Hand Value: {avg_hand_value:.1f}")

        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            print(f"Tournament results saved to {csv_path}, File size: {file_size} bytes")
            with open(csv_path, 'r') as f:
                content = f.read()
                print(f"CSV content preview: {content[:200]}...")

        # Compare agents
        comparison = []
        for agent_type in results:
            win_rate = np.mean(results[agent_type]['wins'])
            avg_reward = np.mean(results[agent_type]['rewards'])
            avg_coin_change = np.mean(results[agent_type]['coin_changes'])
            avg_turns = np.mean(results[agent_type]['turns'])
            avg_hand_value = np.mean(results[agent_type]['hand_values'])
            comparison.append({
                'agent_type': agent_type,
                'win_rate': win_rate,
                'avg_reward': avg_reward,
                'avg_coin_change': avg_coin_change,
                'avg_turns': avg_turns,
                'avg_hand_value': avg_hand_value
            })
            print(f"\n{agent_type.upper()} Results:")
            print(f"Win Rate: {win_rate:.3f}")
            print(f"Average Reward: {avg_reward:.1f}")
            print(f"Average Coin Change: {avg_coin_change:.1f}")
            print(f"Average Turns: {avg_turns:.1f}")
            print(f"Average Hand Value: {avg_hand_value:.1f}")

        best_agent = max(comparison, key=lambda x: (x['win_rate'], x['avg_reward']))
        print(f"\nBest Hybrid Agent for Final Tournament: {best_agent['agent_type'].upper()}")
        print(f"Win Rate: {best_agent['win_rate']:.3f}")
        print(f"Average Reward: {best_agent['avg_reward']:.1f}")
        print(f"Average Coin Change: {best_agent['avg_coin_change']:.1f}")
        print(f"Average Turns: {best_agent['avg_turns']:.1f}")
        print(f"Average Hand Value: {best_agent['avg_hand_value']:.1f}")

        return comparison

    except Exception as e:
        print(f"Error during tournament: {e}")
        with open(json_path, 'w') as f:
            json.dump(json_log, f, indent=2)
        raise

if __name__ == "__main__":
    game = DhumbalGame(num_players=NUM_PLAYERS)
    try:
        print("Starting Hybrid Agents Tournament...")
        comparison = evaluate_hybrid_agents(game)
        print("\nTournament Complete.")
    except FileNotFoundError as e:
        print(f"Model loading failed: {e}")
    except Exception as e:
        print(f"Tournament failed: {e}")