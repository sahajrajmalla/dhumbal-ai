import random
import itertools
import json
import logging
import csv
import time
from collections import defaultdict, Counter, OrderedDict
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import math
import copy
from scipy.stats import ttest_ind
from tqdm import tqdm

# Configuring logging for thread-safe output
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
logger = logging.getLogger(__name__)

# Defining game constants
MAX_PLAYERS = 5
MIN_PLAYERS = 2
HAND_SIZE = 5
JHYAP_THRESHOLD = 10
STARTING_COINS = 10000
MAX_TURNS = 100
MIN_DISCARD_PILE_SIZE = 2
MAX_PAYMENT = 100
MCTS_ITERATIONS = 100
ISMCTS_DETERMINIZATIONS = 3
EXPLORATION_CONSTANT = math.sqrt(2)
MAX_DECISION_TIME = 1.0  # seconds
NUM_ROUNDS = 1024
ACTION_CACHE_SIZE = 1000

class AIStyle(Enum):
    """Enumerates AI strategy types."""
    MCTS = "mcts"
    ISMCTS = "ismcts"

@dataclass
class BeliefState:
    """Represents the AI's belief about opponent hands."""
    possible_opponent_hands: List[Set['Card']]
    known_not_in_hands: List[Set['Card']]
    num_players: int
    opponent_hand_sizes: List[int]

    def copy(self) -> 'BeliefState':
        """Creates a deep copy of the belief state."""
        return BeliefState(
            possible_opponent_hands=[set(h) for h in self.possible_opponent_hands],
            known_not_in_hands=[set(h) for h in self.known_not_in_hands],
            num_players=self.num_players,
            opponent_hand_sizes=self.opponent_hand_sizes.copy()
        )

    def __hash__(self) -> int:
        """Generates a hash for the belief state."""
        return hash((
            tuple(tuple(sorted(str(c) for c in h)) for h in self.possible_opponent_hands),
            tuple(tuple(sorted(str(c) for c in h)) for h in self.known_not_in_hands),
            tuple(self.opponent_hand_sizes)
        ))

@dataclass
class GameState:
    """Represents the current state of the game."""
    round_number: int
    current_player: int
    hands: List[List['Card']]
    discard_pile: List['Card']
    deck_size: int
    player_coins: List[int]
    turn_count: int
    phase: str
    belief_state: BeliefState
    done: bool = False
    winner: Optional[int] = None
    action: Optional[Any] = None

    def copy(self) -> 'GameState':
        """Creates a deep copy of the game state."""
        return GameState(
            round_number=self.round_number,
            current_player=self.current_player,
            hands=[hand[:] for hand in self.hands],
            discard_pile=self.discard_pile[:],
            deck_size=self.deck_size,
            player_coins=self.player_coins.copy(),
            turn_count=self.turn_count,
            phase=self.phase,
            belief_state=self.belief_state.copy(),
            done=self.done,
            winner=self.winner,
            action=self.action
        )

    def __hash__(self) -> int:
        """Generates a hash for the game state."""
        own_hand = tuple(sorted(str(c) for c in self.hands[self.current_player]))
        public = (
            self.current_player,
            tuple(str(c) for c in self.discard_pile),
            self.deck_size,
            tuple(self.player_coins),
            self.turn_count,
            self.phase
        )
        return hash((own_hand, public, self.belief_state.__hash__()))

@dataclass
class RoundResult:
    """Stores the results of a game round."""
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
        """Converts the round result to a dictionary for JSON output."""
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
    """Represents a playing card with suit and rank."""
    def __init__(self, suit: str, rank: str):
        """Initializes a card with suit and rank."""
        self.suit = suit
        self.rank = rank
        self.value = self._calculate_value()

    def _calculate_value(self) -> int:
        """Calculates the point value of the card."""
        if self.rank == 'A':
            return 1
        elif self.rank == 'J':
            return 11
        elif self.rank == 'Q':
            return 12
        elif self.rank == 'K':
            return 13
        return int(self.rank)

    def __str__(self) -> str:
        """Returns the string representation of the card."""
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        """Returns the string representation for debugging."""
        return str(self)

    def __eq__(self, other) -> bool:
        """Checks if two cards are equal."""
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self) -> int:
        """Generates a hash for the card."""
        return hash((self.suit, self.rank))

class DhumbalGame:
    """Implements the core Dhumbal game mechanics."""
    def __init__(self, num_players: int = 2, shared_coins: Optional[List[int]] = None, shared_history: Optional[List[Dict]] = None):
        """Initializes the game with the specified number of players."""
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(f"Dhumbal requires {MIN_PLAYERS}-{MAX_PLAYERS} players")
        self.num_players = num_players
        self.player_coins = shared_coins if shared_coins is not None else [STARTING_COINS] * num_players
        self.game_history = shared_history if shared_history is not None else []
        self.round_number = 0
        self.SUITS = ['♠', '♥', '♦', '♣']
        self.RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.RANK_ORDER = {rank: i for i, rank in enumerate(self.RANKS)}
        self.full_deck = self.create_full_deck()

    def create_full_deck(self) -> List[Card]:
        """Creates a full 52-card deck."""
        return [Card(suit, rank) for suit in self.SUITS for rank in self.RANKS]

    def create_deck(self) -> List[Card]:
        """Creates and shuffles a deck."""
        deck = self.full_deck[:]
        random.shuffle(deck)
        return deck

    def deal_cards(self) -> Tuple[List[List[Card]], List[Card]]:
        """Deals 5 cards to each player and returns remaining deck."""
        deck = self.create_deck()
        hands = [[] for _ in range(self.num_players)]
        for _ in range(HAND_SIZE):
            for player in range(self.num_players):
                if deck:
                    hands[player].append(deck.pop())
        return hands, deck

    def calculate_hand_value(self, hand: List[Card]) -> int:
        """Calculates the total point value of a hand."""
        return sum(card.value for card in hand)

    def can_call_jhyap(self, hand: List[Card]) -> bool:
        """Checks if a hand is eligible to call Jhyap."""
        return self.calculate_hand_value(hand) <= JHYAP_THRESHOLD

    def validate_same_rank_set(self, cards: List[Card]) -> bool:
        """Validates if cards form a same-rank set."""
        if len(cards) < 1:
            return False
        if len(cards) == 1:
            return True
        return all(card.rank == cards[0].rank for card in cards)

    def validate_sequence(self, cards: List[Card]) -> bool:
        """Validates if cards form a consecutive same-suit sequence."""
        if len(cards) < 3:
            return False
        if not all(card.suit == cards[0].suit for card in cards):
            return False
        try:
            positions = sorted([self.RANK_ORDER[card.rank] for card in cards])
            return all(positions[i] == positions[i-1] + 1 for i in range(1, len(positions)))
        except KeyError:
            return False

    def validate_discard(self, cards: List[Card]) -> bool:
        """Validates if a discard is legal."""
        if not cards:
            return False
        return self.validate_same_rank_set(cards) or self.validate_sequence(cards)

    def get_active_players(self) -> List[int]:
        """Returns indices of players with positive coins."""
        return [i for i, coins in enumerate(self.player_coins) if coins > 0]

    def is_game_over(self) -> bool:
        """Checks if the game is over due to insufficient active players."""
        return len(self.get_active_players()) < MIN_PLAYERS

class DhumbalEnv:
    """Manages the game environment for AI decision-making."""
    def __init__(self, game: DhumbalGame, ai_players: List['SearchBasedAI'], current_player: int):
        """Initializes the environment for a specific player."""
        self.game = game
        self.ai_players = ai_players
        self.current_player = current_player
        self.hands, self.deck = game.deal_cards()
        self.discard_pile = [self.deck.pop()] if self.deck else []
        self.cards_seen = set(self.discard_pile)
        full_deck = set(game.full_deck)
        known_cards = set(self.discard_pile)
        possible_cards = full_deck - known_cards
        opp_hands = [possible_cards.copy() for _ in range(game.num_players)]
        not_in_hands = [set() for _ in range(game.num_players)]
        self.belief_state = BeliefState(
            possible_opponent_hands=opp_hands,
            known_not_in_hands=not_in_hands,
            num_players=game.num_players,
            opponent_hand_sizes=[len(self.hands[i]) for i in range(game.num_players)]
        )
        self.state = GameState(
            round_number=game.round_number + 1,
            current_player=current_player,
            hands=[self.hands[current_player][:] if i == current_player else [] for i in range(game.num_players)],
            discard_pile=self.discard_pile[:],
            deck_size=len(self.deck),
            player_coins=game.player_coins[:],
            turn_count=0,
            phase='call',
            belief_state=self.belief_state
        )
        self.action_cache: OrderedDict = OrderedDict()

    def reset(self):
        """Resets the environment for a new round."""
        self.hands, self.deck = self.game.deal_cards()
        self.discard_pile = [self.deck.pop()] if self.deck else []
        self.cards_seen = set(self.discard_pile)
        full_deck = set(self.game.full_deck)
        known_cards = set(self.discard_pile)
        possible_cards = full_deck - known_cards
        opp_hands = [possible_cards.copy() for _ in range(self.game.num_players)]
        not_in_hands = [set() for _ in range(self.game.num_players)]
        self.belief_state = BeliefState(
            possible_opponent_hands=opp_hands,
            known_not_in_hands=not_in_hands,
            num_players=self.game.num_players,
            opponent_hand_sizes=[len(self.hands[i]) for i in range(self.game.num_players)]
        )
        self.state = GameState(
            round_number=self.game.round_number + 1,
            current_player=self.current_player,
            hands=[self.hands[self.current_player][:] if i == self.current_player else [] for i in range(self.game.num_players)],
            discard_pile=self.discard_pile[:],
            deck_size=len(self.deck),
            player_coins=self.game.player_coins[:],
            turn_count=0,
            phase='call',
            belief_state=self.belief_state
        )
        self.action_cache.clear()

    def set_state(self, new_state: GameState):
        """Sets the environment to a specific game state."""
        self.state = new_state.copy()
        self.hands[self.current_player] = new_state.hands[self.current_player][:]
        self.discard_pile = new_state.discard_pile[:]
        self.deck = [Card('♠', 'A')] * new_state.deck_size
        self.cards_seen = set(self.discard_pile)
        self.belief_state = new_state.belief_state.copy()
        self.belief_state.opponent_hand_sizes[self.current_player] = len(self.hands[self.current_player])
        assert 0 <= self.belief_state.opponent_hand_sizes[self.current_player] <= 52, \
            f"Invalid hand size for player {self.current_player}: {self.belief_state.opponent_hand_sizes[self.current_player]}"

    def update_belief(self, player_id: int, action: Any, phase: str):
        """Updates belief state based on observed actions."""
        if phase == 'discard' and isinstance(action, list):
            new_size = max(0, self.belief_state.opponent_hand_sizes[player_id] - len(action))
            assert 0 <= new_size <= 52, f"Invalid hand size for player {player_id}: {new_size}"
            self.belief_state.opponent_hand_sizes[player_id] = new_size
            for i in range(self.game.num_players):
                if i != player_id:
                    self.belief_state.known_not_in_hands[i].update(action)
                    for card in action:
                        self.belief_state.possible_opponent_hands[i].discard(card)
        elif phase == 'pick':
            new_size = self.belief_state.opponent_hand_sizes[player_id] + 1
            assert 0 <= new_size <= 52, f"Invalid hand size for player {player_id}: {new_size}"
            self.belief_state.opponent_hand_sizes[player_id] = new_size
            if action == 'discard' and self.discard_pile:
                card = self.discard_pile[-1]
                for i in range(self.game.num_players):
                    if i != player_id:
                        self.belief_state.known_not_in_hands[i].add(card)
                        self.belief_state.possible_opponent_hands[i].discard(card)
        if isinstance(action, list):
            self.cards_seen.update(action)

    def get_actions(self) -> List[Any]:
        """Generates legal actions for the current game state."""
        state_hash = hash(self.state)
        if state_hash in self.action_cache:
            return self.action_cache[state_hash]
        actions = []
        if self.state.phase == 'call':
            actions = [True, False] if self.game.can_call_jhyap(self.state.hands[self.current_player]) else [False]
        elif self.state.phase == 'discard':
            hand = self.state.hands[self.current_player]
            if not hand:
                return []
            actions = [[card] for card in hand]
            rank_groups = defaultdict(list)
            for card in hand:
                rank_groups[card.rank].append(card)
            for cards in rank_groups.values():
                for size in range(1, len(cards) + 1):
                    actions.extend(list(combo) for combo in itertools.combinations(cards, size))
            suit_groups = defaultdict(list)
            for card in hand:
                suit_groups[card.suit].append(card)
            for suit, cards in suit_groups.items():
                if len(cards) >= 3:
                    cards_sorted = sorted(cards, key=lambda x: self.game.RANK_ORDER[x.rank])
                    for size in range(3, len(cards_sorted) + 1):
                        for combo in itertools.combinations(cards_sorted, size):
                            combo_list = list(combo)
                            if self.game.validate_sequence(combo_list):
                                actions.append(combo_list)
        elif self.state.phase == 'pick':
            actions = ['deck']
            if self.state.discard_pile:
                actions.append('discard')
        self.action_cache[state_hash] = actions
        if len(self.action_cache) > ACTION_CACHE_SIZE:
            self.action_cache.popitem(last=False)
        return actions

    def step(self, action: Any) -> Tuple[GameState, float, bool]:
        """Advances the game state by applying an action."""
        new_state = self.state.copy()
        reward = 0.0
        done = False
        new_state.action = action

        if new_state.phase == 'call':
            if action:
                det_hands = self._determinize_hands(new_state)
                hand_values = [self.game.calculate_hand_value(hand) for hand in det_hands]
                caller = new_state.current_player
                min_value = min(hand_values)
                min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                non_caller_min = [i for i in min_value_players if i != caller]
                winner = non_caller_min[0] if non_caller_min else caller
                successful_call = (caller == winner)
                if successful_call:
                    reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != caller)
                else:
                    reward = -sum(min(v, MAX_PAYMENT) for v in hand_values)
                new_state.done = True
                new_state.winner = winner
                self.set_state(new_state)
                return new_state, reward, True
            new_state.phase = 'discard'

        elif new_state.phase == 'discard':
            if not new_state.hands[new_state.current_player]:
                new_state.done = True
                det_hands = self._determinize_hands(new_state)
                hand_values = [self.game.calculate_hand_value(hand) for hand in det_hands]
                min_value = min(hand_values)
                min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                winner = min_value_players[0]
                new_state.winner = winner
                reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != new_state.current_player) if winner == new_state.current_player else 0
                self.set_state(new_state)
                return new_state, reward, True
            if self.game.validate_discard(action):
                for card in action:
                    if card in new_state.hands[new_state.current_player]:
                        new_state.hands[new_state.current_player].remove(card)
                new_state.discard_pile.extend(action)
                self.update_belief(new_state.current_player, action, 'discard')
                new_state.phase = 'pick'
            else:
                reward = -10.0
                new_state.done = True
                new_state.winner = (new_state.current_player + 1) % self.game.num_players
                self.set_state(new_state)
                return new_state, reward, True

        elif new_state.phase == 'pick':
            if action == 'discard' and new_state.discard_pile:
                card = new_state.discard_pile.pop()
                new_state.hands[new_state.current_player].append(card)
                self.update_belief(new_state.current_player, action, 'pick')
            elif action == 'deck':
                if new_state.deck_size == 0 and len(new_state.discard_pile) >= MIN_DISCARD_PILE_SIZE:
                    top = new_state.discard_pile.pop() if new_state.discard_pile else None
                    random.shuffle(new_state.discard_pile)
                    new_state.deck_size = len(new_state.discard_pile)
                    new_state.discard_pile[:] = [top] if top else []
                if new_state.deck_size > 0:
                    new_state.deck_size -= 1
                    full_deck = set(self.game.full_deck)
                    known_cards = (set(new_state.hands[new_state.current_player]) |
                                   set(new_state.discard_pile) |
                                   self.cards_seen)
                    available = list(full_deck - known_cards)
                    if available:
                        card = random.choice(available)
                        new_state.hands[new_state.current_player].append(card)
                        self.update_belief(new_state.current_player, action, 'pick')
                    else:
                        new_state.done = True
                        det_hands = self._determinize_hands(new_state)
                        hand_values = [self.game.calculate_hand_value(hand) for hand in det_hands]
                        min_value = min(hand_values)
                        min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                        winner = min_value_players[0]
                        new_state.winner = winner
                        if new_state.winner == new_state.current_player:
                            reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != new_state.current_player)
                        else:
                            reward = -min(hand_values[new_state.current_player], MAX_PAYMENT)
                        self.set_state(new_state)
                        return new_state, reward, True
                else:
                    new_state.done = True
                    det_hands = self._determinize_hands(new_state)
                    hand_values = [self.game.calculate_hand_value(hand) for hand in det_hands]
                    min_value = min(hand_values)
                    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                    winner = min_value_players[0]
                    new_state.winner = winner
                    if new_state.winner == new_state.current_player:
                        reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != new_state.current_player)
                    else:
                        reward = -min(hand_values[new_state.current_player], MAX_PAYMENT)
                    self.set_state(new_state)
                    return new_state, reward, True
            new_state.turn_count += 1
            new_state.current_player = (new_state.current_player + 1) % self.game.num_players
            new_state.phase = 'call'
            if new_state.turn_count >= MAX_TURNS:
                new_state.done = True
                det_hands = self._determinize_hands(new_state)
                hand_values = [self.game.calculate_hand_value(hand) for hand in det_hands]
                min_value = min(hand_values)
                min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                winner = min_value_players[0]
                new_state.winner = winner
                if new_state.winner == new_state.current_player:
                    reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != new_state.current_player)
                else:
                    reward = -min(hand_values[new_state.current_player], MAX_PAYMENT)
                self.set_state(new_state)
                return new_state, reward, True

        self.set_state(new_state)
        return new_state, reward, done

    def _determinize_hands(self, state: GameState) -> List[List[Card]]:
        """Determinizes opponent hands based on belief state, adapting to variable hand sizes."""
        det_hands = [state.hands[state.current_player][:] if i == state.current_player else [] for i in range(self.game.num_players)]
        full_deck = set(self.game.full_deck)
        known_cards = set(state.hands[state.current_player]) | set(state.discard_pile) | self.cards_seen
        available = list(full_deck - known_cards)
        # Include discard pile in available cards if deck is insufficient
        if len(available) < sum(state.belief_state.opponent_hand_sizes[i] for i in range(self.game.num_players) if i != state.current_player):
            available.extend([card for card in state.discard_pile if card not in known_cards])
        random.shuffle(available)
        for i in range(self.game.num_players):
            if i != state.current_player:
                hand_size = state.belief_state.opponent_hand_sizes[i]
                possible_cards = list(state.belief_state.possible_opponent_hands[i] & set(available))
                random.shuffle(possible_cards)
                # Assign up to hand_size cards, or fewer if not enough available
                num_cards_to_sample = min(hand_size, len(possible_cards))
                det_hands[i] = possible_cards[:num_cards_to_sample]
                available = [c for c in available if c not in det_hands[i]]
        return det_hands

def get_utility(state: GameState, player_id: int, game: DhumbalGame) -> float:
    """Calculates the utility for a player in a terminal state."""
    if not state.done:
        return 0.0
    det_hands = [state.hands[player_id] if i == player_id else [Card('♠', 'A')] * state.belief_state.opponent_hand_sizes[i]
                 for i in range(game.num_players)]
    hand_values = [game.calculate_hand_value(hand) for hand in det_hands]
    min_value = min(hand_values)
    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
    caller = None
    successful = True
    winner = min_value_players[0]
    if state.phase == 'call' and state.action is True:
        caller = state.current_player
        non_caller_min = [i for i in min_value_players if i != caller]
        if non_caller_min:
            winner = non_caller_min[0]
            successful = False
        else:
            winner = caller
    coin_changes = [0] * len(hand_values)
    if successful and caller is not None:
        pay_to = winner
        for i in range(len(hand_values)):
            if i != pay_to:
                payment = min(hand_values[i], MAX_PAYMENT)
                coin_changes[i] = -payment
                coin_changes[pay_to] += payment
    elif caller is not None:
        total = sum(min(v, MAX_PAYMENT) for v in hand_values)
        coin_changes[caller] = -total
        coin_changes[winner] = total
    else:
        pay_to = winner
        for i in range(len(hand_values)):
            if i != pay_to:
                payment = min(hand_values[i], MAX_PAYMENT)
                coin_changes[i] = -payment
                coin_changes[pay_to] += payment
    return coin_changes[player_id]

def action_key(action):
    """Generates a unique key for an action."""
    if isinstance(action, bool):
        return ('call', action)
    elif isinstance(action, str):
        return ('pick', action)
    elif isinstance(action, list):
        return ('discard', tuple(sorted(str(c) for c in action)))
    return None

class InfoSetNode:
    """Represents a node in the MCTS/ISMCTS search tree."""
    def __init__(self, parent: Optional['InfoSetNode'] = None, action: Optional[Any] = None):
        """Initializes a node in the search tree."""
        self.parent = parent
        self.action = action
        self.children: Dict[Tuple[str, Any], 'InfoSetNode'] = {}
        self.visits = 0
        self.reward = 0.0
        self.action_key = action_key(action) if action is not None else None
        self.legal_count: Dict[Tuple[str, Any], int] = defaultdict(int)
        self.total_determinizations = 0

    def get_or_create_child(self, action: Any) -> 'InfoSetNode':
        """Gets or creates a child node for an action."""
        key = action_key(action)
        if key not in self.children:
            self.children[key] = InfoSetNode(parent=self, action=action)
        return self.children[key]

    def ucb_select(self, legal_actions: List[Any]) -> Optional['InfoSetNode']:
        """Selects a child node using modified UCB1."""
        if self.visits == 0 or self.total_determinizations == 0:
            return None
        log_N = math.log(self.visits)
        best_child = None
        best_score = -float('inf')
        for key, child in self.children.items():
            if any(action_key(la) == key for la in legal_actions):
                legality_prob = child.legal_count.get(key, 0) / self.total_determinizations
                if legality_prob == 0:
                    continue
                exploit = child.reward / child.visits if child.visits > 0 else 0
                explore = EXPLORATION_CONSTANT * math.sqrt(log_N / (child.visits + 1e-6))
                score = exploit + explore * legality_prob
                if score > best_score:
                    best_score = score
                    best_child = child
        return best_child

    def update(self, reward: float, legal_actions: List[Any]):
        """Updates node statistics after a simulation."""
        self.visits += 1
        self.reward += reward
        self.total_determinizations += 1
        for action in legal_actions:
            key = action_key(action)
            self.legal_count[key] += 1

    def best_child(self) -> Optional['InfoSetNode']:
        """Selects the best child node based on average reward."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: (c.reward / c.visits if c.visits > 0 else -float('inf')) *
                   (c.legal_count.get(c.action_key, 0) / (self.total_determinizations + 1e-6)))

class MCTS:
    """Implements Monte Carlo Tree Search."""
    def __init__(self, iterations: int = MCTS_ITERATIONS):
        """Initializes MCTS with specified iterations."""
        self.iterations = iterations
        self.state_cache: Dict[int, Any] = {}

    def search(self, state: GameState, env: DhumbalEnv, player_id: int) -> InfoSetNode:
        """Performs MCTS search to select an action."""
        state_hash = hash(state)
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]
        root = InfoSetNode()
        start_time = time.time()
        i = 0
        while i < self.iterations and (time.time() - start_time) < MAX_DECISION_TIME:
            det_state = self._determinize(state, env, player_id)
            sim_env = copy.deepcopy(env)
            sim_env.set_state(det_state)
            node = root
            path = []
            while not sim_env.state.done:
                legal_actions = sim_env.get_actions()
                if not legal_actions:
                    break
                child = node.ucb_select(legal_actions)
                if child is None:
                    break
                path.append((node, child.action, legal_actions))
                sim_env.step(child.action)
                node = child
            if not sim_env.state.done:
                legal_actions = sim_env.get_actions()
                untried = [a for a in legal_actions if action_key(a) not in node.children]
                if untried:
                    action = random.choice(untried)
                    _, _, done = sim_env.step(action)
                    node = node.get_or_create_child(action)
                    path.append((node, action, legal_actions))
                    if done:
                        reward = get_utility(sim_env.state, player_id, env.game)
                    else:
                        reward = self._rollout(sim_env, player_id, env.game)
                else:
                    reward = self._rollout(sim_env, player_id, env.game)
            else:
                reward = get_utility(sim_env.state, player_id, env.game)
            for node, _, legal_actions in path:
                node.update(reward, legal_actions)
            node.update(reward, legal_actions if path else sim_env.get_actions())
            i += 1
        self.state_cache[state_hash] = root
        return root

    def _determinize(self, state: GameState, env: DhumbalEnv, player_id: int) -> GameState:
        """Creates a determinized game state."""
        det_state = state.copy()
        det_state.hands = [state.hands[player_id][:] if i == player_id else [] for i in range(env.game.num_players)]
        full_deck = set(env.game.full_deck)
        known_cards = set(state.hands[player_id]) | set(state.discard_pile) | env.cards_seen
        available = list(full_deck - known_cards)
        # Include discard pile if necessary
        if len(available) < sum(state.belief_state.opponent_hand_sizes[i] for i in range(env.game.num_players) if i != player_id):
            available.extend([card for card in state.discard_pile if card not in known_cards])
        random.shuffle(available)
        for i in range(env.game.num_players):
            if i != player_id:
                hand_size = state.belief_state.opponent_hand_sizes[i]
                possible_cards = list(state.belief_state.possible_opponent_hands[i] & set(available))
                random.shuffle(possible_cards)
                num_cards_to_sample = min(hand_size, len(possible_cards))
                det_state.hands[i] = possible_cards[:num_cards_to_sample]
                available = [c for c in available if c not in det_state.hands[i]]
        det_state.deck_size = max(0, len(available))
        return det_state

    def _rollout(self, env: DhumbalEnv, player_id: int, game: DhumbalGame) -> float:
        """Performs a random simulation from the current state."""
        sim_env = copy.deepcopy(env)
        while not sim_env.state.done:
            actions = sim_env.get_actions()
            if not actions:
                return -10.0
            action = random.choice(actions)
            sim_env.step(action)
        return get_utility(sim_env.state, player_id, game)

class ISMCTS(MCTS):
    """Implements Information Set Monte Carlo Tree Search."""
    def __init__(self, iterations: int = MCTS_ITERATIONS, determinizations: int = ISMCTS_DETERMINIZATIONS):
        """Initializes ISMCTS with specified iterations and determinizations."""
        super().__init__(iterations)
        self.determinizations = determinizations

    def search(self, state: GameState, env: DhumbalEnv, player_id: int) -> InfoSetNode:
        """Performs ISMCTS search with multiple determinizations."""
        state_hash = hash(state)
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]
        root = InfoSetNode()
        start_time = time.time()
        i = 0
        while i < self.iterations and (time.time() - start_time) < MAX_DECISION_TIME:
            for _ in range(self.determinizations):
                det_state = self._determinize(state, env, player_id)
                sim_env = copy.deepcopy(env)
                sim_env.set_state(det_state)
                node = root
                path = []
                while not sim_env.state.done:
                    legal_actions = sim_env.get_actions()
                    if not legal_actions:
                        break
                    child = node.ucb_select(legal_actions)
                    if child is None:
                        break
                    path.append((node, child.action, legal_actions))
                    sim_env.step(child.action)
                    node = child
                if not sim_env.state.done:
                    legal_actions = sim_env.get_actions()
                    untried = [a for a in legal_actions if action_key(a) not in node.children]
                    if untried:
                        action = random.choice(untried)
                        _, _, done = sim_env.step(action)
                        node = node.get_or_create_child(action)
                        path.append((node, action, legal_actions))
                        if done:
                            reward = get_utility(sim_env.state, player_id, env.game)
                        else:
                            reward = self._rollout(sim_env, player_id, env.game)
                    else:
                        reward = self._rollout(sim_env, player_id, env.game)
                else:
                    reward = get_utility(sim_env.state, player_id, env.game)
                for node, _, legal_actions in path:
                    node.update(reward, legal_actions)
                node.update(reward, legal_actions if path else sim_env.get_actions())
            i += 1
        self.state_cache[state_hash] = root
        return root

class SearchBasedAI:
    """Implements an AI player using MCTS or ISMCTS."""
    def __init__(self, player_id: int, style: AIStyle, decision_times: List[float]):
        """Initializes an AI player with specified strategy."""
        self.player_id = player_id
        self.style = style
        self.name = f"AI_{style.value}_{player_id}"
        self.mcts = MCTS() if style == AIStyle.MCTS else ISMCTS()
        self.cards_seen: Set[Card] = set()
        self.decision_times = decision_times

    def choose_discard(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> List[Card]:
        """Selects cards to discard using MCTS/ISMCTS."""
        start_time = time.time()
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'discard'
        env.cards_seen = self.cards_seen.copy()
        root = self.mcts.search(env.state, env, self.player_id)
        best_child = root.best_child()
        action = best_child.action if best_child else None
        self.decision_times.append(time.time() - start_time)
        if isinstance(action, list) and game.validate_discard(action):
            return action
        return [max(hand, key=lambda x: x.value)] if hand else []

    def should_pick_from_discard(self, discard_pile: List[Card], current_hand: List[Card], game_state: GameState, game: DhumbalGame) -> Tuple[bool, Optional[Card]]:
        """Decides whether to pick from the discard pile."""
        start_time = time.time()
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'pick'
        env.cards_seen = self.cards_seen.copy()
        root = self.mcts.search(env.state, env, self.player_id)
        best_child = root.best_child()
        action = best_child.action if best_child else None
        self.decision_times.append(time.time() - start_time)
        if action == 'discard' and discard_pile:
            return True, discard_pile[-1]
        return False, None

    def should_call_jhyap(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> bool:
        """Decides whether to call Jhyap."""
        start_time = time.time()
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'call'
        env.cards_seen = self.cards_seen.copy()
        root = self.mcts.search(env.state, env, self.player_id)
        best_child = root.best_child()
        action = best_child.action if best_child else None
        self.decision_times.append(time.time() - start_time)
        return bool(action)

def simulate_round(round_idx: int, game: DhumbalGame, ai_players: List[SearchBasedAI], shared_coins: List[int], verbose: bool, debug: bool) -> RoundResult:
    """Simulates a single round of Dhumbal."""
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    game.round_number = round_idx + 1
    hands, deck = game.deal_cards()
    discard_pile = [deck.pop()] if deck else []
    envs = [DhumbalEnv(game, ai_players, i) for i in range(game.num_players)]
    for env in envs:
        env.reset()
        env.cards_seen = set(discard_pile)
        full_deck = set(game.full_deck)
        env.belief_state.possible_opponent_hands = [full_deck - set(discard_pile) for _ in range(game.num_players)]
        env.belief_state.opponent_hand_sizes = [len(hands[j]) for j in range(game.num_players)]
        env.state.hands[env.current_player] = hands[env.current_player][:]
        env.state.player_coins = list(shared_coins)
        env.state.belief_state = env.belief_state
    game_state = GameState(
        round_number=game.round_number,
        current_player=0,
        hands=[[] for _ in range(game.num_players)],
        discard_pile=discard_pile[:],
        deck_size=len(deck),
        player_coins=list(shared_coins),
        turn_count=0,
        phase='call',
        belief_state=envs[0].belief_state
    )
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {game.round_number}")
        logger.info(f"{'='*60}")
        logger.info(f"Initial discard: {discard_pile[0] if discard_pile else 'None'}")
        for i in range(game.num_players):
            hand_str = [str(card) for card in hands[i]]
            value = game.calculate_hand_value(hands[i])
            logger.info(f"Player {i} ({ai_players[i].name}): {hand_str} (value: {value})")
    current_player = 0
    while game_state.turn_count < MAX_TURNS and not game_state.done:
        ai = ai_players[current_player]
        env = envs[current_player]
        player_hand = hands[current_player]
        if not player_hand:
            if verbose:
                logger.info(f"Player {current_player} has no cards left, ending round")
            return end_round(game, hands, current_player, game_state.turn_count, discard_pile, deck, shared_coins, verbose, debug)
        game_state.current_player = current_player
        game_state.hands[current_player] = hands[current_player][:]
        game_state.discard_pile = discard_pile[:]
        game_state.deck_size = len(deck)
        game_state.player_coins = list(shared_coins)
        game_state.turn_count += 1
        game_state.belief_state = env.belief_state
        if verbose:
            logger.info(f"\n--- Turn {game_state.turn_count}: Player {current_player} ({ai.name}) ---")
            logger.info(f"Hand: {[str(c) for c in player_hand]} (value: {game.calculate_hand_value(player_hand)})")
            if discard_pile:
                logger.info(f"Discard top: {discard_pile[-1]}")
        game_state.phase = 'call'
        if game.can_call_jhyap(player_hand) and ai.should_call_jhyap(player_hand, game_state, game):
            if verbose:
                logger.info(f"Player {current_player} calls JHYAP with {game.calculate_hand_value(player_hand)} points!")
            return end_round(game, hands, current_player, game_state.turn_count, discard_pile, deck, shared_coins, verbose, debug)
        game_state.phase = 'discard'
        cards_to_discard = ai.choose_discard(player_hand, game_state, game)
        if debug:
            logger.debug(f"Evaluated discard: {[str(c) for c in cards_to_discard]}")
        if not game.validate_discard(cards_to_discard) and player_hand:
            cards_to_discard = [max(player_hand, key=lambda x: x.value)] if player_hand else []
            if debug:
                logger.debug(f"Invalid discard, falling back to: {[str(c) for c in cards_to_discard]}")
        for card in cards_to_discard:
            if card in player_hand:
                player_hand.remove(card)
        discard_pile.extend(cards_to_discard)
        for env in envs:
            env.update_belief(current_player, cards_to_discard, 'discard')
        if verbose:
            logger.info(f"Discarded: {[str(c) for c in cards_to_discard]}")
        if not player_hand:
            if verbose:
                logger.info(f"Player {current_player} has no cards left, ending round")
            return end_round(game, hands, current_player, game_state.turn_count, discard_pile, deck, shared_coins, verbose, debug)
        game_state.phase = 'pick'
        top_discard = discard_pile[-1] if discard_pile else None
        should_pick, specific_card = ai.should_pick_from_discard(discard_pile, player_hand, game_state, game)
        if should_pick and top_discard:
            player_hand.append(discard_pile.pop())
            for env in envs:
                env.update_belief(current_player, 'discard', 'pick')
            if verbose:
                logger.info(f"Picked from discard: {top_discard}")
        else:
            if not deck and len(discard_pile) >= MIN_DISCARD_PILE_SIZE:
                top = discard_pile.pop() if discard_pile else None
                random.shuffle(discard_pile)
                deck.extend(discard_pile[:])
                discard_pile[:] = [top] if top else []
                if debug:
                    logger.debug(f"Reshuffled discard pile into deck, new deck size: {len(deck)}")
            if deck:
                picked_card = deck.pop()
                player_hand.append(picked_card)
                for env in envs:
                    env.update_belief(current_player, 'deck', 'pick')
                if verbose:
                    logger.info(f"Picked from deck: {picked_card}")
        if not deck and not discard_pile and not any(len(h) > 0 for h in hands):
            if verbose:
                logger.info("No cards remain, ending round")
            hand_values = [game.calculate_hand_value(hand) for hand in hands]
            caller = hand_values.index(min(hand_values))
            return end_round(game, hands, caller, game_state.turn_count, discard_pile, deck, shared_coins, verbose, debug)
        current_player = (current_player + 1) % game.num_players
    if verbose:
        logger.info("\nRound exceeded maximum turns, forcing showdown...")
    hand_values = [game.calculate_hand_value(hand) for hand in hands]
    caller = hand_values.index(min(hand_values))
    return end_round(game, hands, caller, game_state.turn_count, discard_pile, deck, shared_coins, verbose, debug)

def end_round(game: DhumbalGame, hands: List[List[Card]], caller: int, turns_played: int, discard_pile: List[Card], deck: List[Card], shared_coins: List[int], verbose: bool, debug: bool) -> RoundResult:
    """Ends a round and calculates scores."""
    hand_values = [game.calculate_hand_value(hand) for hand in hands]
    caller_value = hand_values[caller]
    min_value = min(hand_values)
    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
    non_caller_min = [i for i in min_value_players if i != caller]
    winner = non_caller_min[0] if non_caller_min else caller
    successful_call = (caller == winner)
    coin_changes = [0] * game.num_players
    if successful_call:
        for i in range(game.num_players):
            if i != caller:
                payment = min(hand_values[i], MAX_PAYMENT)
                coin_changes[i] = -payment
                coin_changes[caller] += payment
    else:
        total_payment = sum(min(v, MAX_PAYMENT) for v in hand_values)
        coin_changes[caller] = -total_payment
        coin_changes[winner] = total_payment
    # Update shared player_coins
    for i in range(game.num_players):
        shared_coins[i] = shared_coins[i] + coin_changes[i]
    if verbose:
        logger.info(f"\n{'='*40}")
        logger.info(f"ROUND {game.round_number} RESULTS")
        logger.info(f"{'='*40}")
        logger.info(f"Caller: Player {caller} with {caller_value} points")
        for i, value in enumerate(hand_values):
            status = " (WINNER)" if i == winner else ""
            logger.info(f"Player {i}: {value} points {[str(c) for c in hands[i]]}{status}")
        if successful_call:
            logger.info(f"\nSuccessful Jhyap call! Each player pays Player {caller} their hand value (capped at {MAX_PAYMENT})")
        else:
            logger.info(f"\nFailed Jhyap call! Player {caller} pays {total_payment} coins to Player {winner}")
        logger.info(f"Coin changes: {coin_changes}")
        logger.info(f"Final balances: {[f'P{i}:{shared_coins[i]}' for i in range(game.num_players)]}")
    if debug:
        logger.debug(f"End round state: hands={[len(h) for h in hands]}, deck_size={len(deck)}, discard_size={len(discard_pile)}")
    result = RoundResult(
        round_number=game.round_number,
        caller=caller,
        winner=winner,
        hand_values=hand_values,
        coin_changes=coin_changes,
        final_coins=list(shared_coins),
        turns_played=turns_played,
        successful_call=successful_call,
        hands=[hand[:] for hand in hands]
    )
    game.game_history.append(result.to_dict())
    return result

def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculates Cohen's d for two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    pooled_std = np.sqrt(((len(group1)-1)*np.var(group1) + (len(group2)-1)*np.var(group2)) /
                        (len(group1)+len(group2)-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0.0

def simulate_game(game: DhumbalGame, ai_players: List[SearchBasedAI], max_rounds: int = NUM_ROUNDS, verbose: bool = True, debug: bool = False) -> Dict[str, Any]:
    """Simulates the entire game for the specified number of rounds."""
    shared_coins = [STARTING_COINS] * game.num_players
    shared_history = []
    shared_decision_times = {i: [] for i in range(game.num_players)}
    game.player_coins = shared_coins
    game.game_history = shared_history
    ai_styles = [ai.style for ai in ai_players]
    for ai in ai_players:
        ai.decision_times = shared_decision_times[ai.player_id]
    round_results = []
    for i in tqdm(range(max_rounds), desc="Simulating rounds"):
        random.seed(42 + i)
        np.random.seed(42 + i)
        try:
            result = simulate_round(i, game, ai_players, shared_coins, verbose, debug)
            shared_history.append(result.to_dict())
            round_results.append(result)
        except Exception as e:
            logger.error(f"Error in round {i + 1}: {str(e)}")
            continue
    game.player_coins = list(shared_coins)
    game.game_history = [RoundResult(**r) for r in shared_history]
    return analyze_game_results(game, ai_players, round_results, verbose)

def analyze_game_results(game: DhumbalGame, ai_players: List[SearchBasedAI], round_results: List[RoundResult], verbose: bool = True) -> Dict[str, Any]:
    """Analyzes game results and generates statistics."""
    if not round_results:
        return {"error": "No rounds completed"}
    total_rounds = len(round_results)
    final_coins = game.player_coins
    winner_id = max(range(game.num_players), key=lambda i: final_coins[i])
    winner_counts = Counter(r.winner for r in round_results)
    caller_counts = Counter(r.caller for r in round_results)
    successful_calls = [r for r in round_results if r.successful_call]
    success_rates = {}
    jhyap_calls = [[] for _ in range(game.num_players)]
    avg_decision_times = [np.mean(list(ai.decision_times)) * 1000 if ai.decision_times else 0.0 for ai in ai_players]
    win_data = [[] for _ in range(game.num_players)]
    economic_data = [[] for _ in range(game.num_players)]
    jhyap_data = [[] for _ in range(game.num_players)]
    cards_data = [[] for _ in range(game.num_players)]
    risk_data = [[] for _ in range(game.num_players)]
    for r in round_results:
        for i in range(game.num_players):
            win_data[i].append(1 if r.winner == i else 0)
            economic_data[i].append(r.coin_changes[i])
            jhyap_data[i].append(1 if r.caller == i else 0)
            temp_state = GameState(
                round_number=r.round_number,
                current_player=i,
                hands=[r.hands[i] if j == i else [] for j in range(game.num_players)],
                discard_pile=[],
                deck_size=0,
                player_coins=r.final_coins,
                turn_count=r.turns_played,
                phase='discard',
                belief_state=BeliefState(
                    possible_opponent_hands=[set(game.full_deck) for _ in range(game.num_players)],
                    known_not_in_hands=[set() for _ in range(game.num_players)],
                    num_players=game.num_players,
                    opponent_hand_sizes=[len(r.hands[j]) for j in range(game.num_players)]
                )
            )
            env = DhumbalEnv(game, ai_players, i)
            env.set_state(temp_state)
            discard_action = ai_players[i].choose_discard(r.hands[i], temp_state, game)
            cards_data[i].append(len(discard_action) if isinstance(discard_action, list) else 1)
            if r.caller == i:
                jhyap_calls[i].append(r.hand_values[i])
                risk_data[i].append(1 if r.successful_call else 0)
    for i in range(game.num_players):
        calls_made = caller_counts.get(i, 0)
        successful = len([r for r in successful_calls if r.caller == i])
        success_rates[i] = (successful / calls_made * 100) if calls_made > 0 else 0
    avg_winning_hand = sum(min(r.hand_values) for r in round_results) / total_rounds if total_rounds > 0 else 0
    avg_turns_per_round = sum(r.turns_played for r in round_results) / total_rounds if total_rounds > 0 else 0
    total_coins_transferred = sum(sum(abs(change) for change in r.coin_changes) for r in round_results) / 2
    win_rates = [winner_counts.get(i, 0) / total_rounds * 100 for i in range(game.num_players)] if total_rounds > 0 else [0] * game.num_players
    win_ci = [1.96 * np.std(win_data[i]) / np.sqrt(total_rounds) * 100 for i in range(game.num_players)] if total_rounds > 0 else [0] * game.num_players
    economic_performance = [sum(r.coin_changes[i] for r in round_results) / total_rounds for i in range(game.num_players)] if total_rounds > 0 else [0] * game.num_players
    jhyap_success_rates = [success_rates[i] for i in range(game.num_players)]
    cards_discarded_avg = [sum(cards_data[i]) / total_rounds for i in range(game.num_players)] if total_rounds > 0 else [0] * game.num_players
    risk_assessment = []
    for i in range(game.num_players):
        calls = jhyap_calls[i]
        successes = risk_data[i]
        if len(calls) >= 2 and len(successes) == len(calls):
            corr = np.corrcoef(calls, successes)[0, 1] if len(calls) > 1 else 0
            risk_assessment.append(corr)
        else:
            risk_assessment.append(None)
    cohens_d = {}
    p_values = {}
    metrics = ['win', 'economic', 'jhyap', 'cards']
    data_lists = [win_data, economic_data, jhyap_data, cards_data]
    comparisons = [(i, j) for i in range(game.num_players) for j in range(i+1, game.num_players)]
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
            'winner_name': ai_players[winner_id].name,
            'final_coins': final_coins,
            'starting_coins': STARTING_COINS
        },
        'player_performance': {
            'win_rates': win_rates,
            'win_ci': win_ci,
            'economic_performance': economic_performance,
            'jhyap_success_rates': jhyap_success_rates,
            'cards_discarded': cards_discarded_avg,
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
    with open(f'game_metrics_rounds_{total_rounds}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Player', 'Win Rate (%)', 'Win CI (%)', 'Economic Perf.', 'Jhyap Success (%)', 'Cards Disc./Round', 'Risk Corr.', 'Avg Dec. Time (ms)'])
        for i in range(game.num_players):
            risk_val = risk_assessment[i] if risk_assessment[i] is not None else 'N/A'
            writer.writerow([
                ai_players[i].name,
                f"{win_rates[i]:.1f}",
                f"{win_ci[i]:.1f}",
                f"{economic_performance[i]:.1f}",
                f"{jhyap_success_rates[i]:.1f}",
                f"{cards_discarded_avg[i]:.1f}",
                risk_val,
                f"{avg_decision_times[i]:.1f}"
            ])
    with open(f'game_cohens_d_rounds_{total_rounds}.csv', 'w', newline='') as f:
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
    with open(f'game_results_rounds_{total_rounds}.json', 'w') as f:
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
            logger.info(f"  {rank}. {ai_players[pid].name}: {final_coins[pid]:,} coins {f'(+{change:,})' if change > 0 else f'({change:,})'}")
        logger.info("\nPlayer Statistics:")
        for i in range(game.num_players):
            logger.info(f"  {ai_players[i].name}:")
            logger.info(f"    • Win Rate: {win_rates[i]:.1f}% ± {win_ci[i]:.1f}%")
            logger.info(f"    • Avg. Coins/Round: {economic_performance[i]:.1f}")
            logger.info(f"    • Jhyap Success: {jhyap_success_rates[i]:.1f}%")
            logger.info(f"    • Avg. Cards Discarded: {cards_discarded_avg[i]:.1f}")
            risk_val = f"{risk_assessment[i]:.3f}" if risk_assessment[i] is not None else "N/A"
            logger.info(f"    • Risk Correlation: {risk_val}")
            logger.info(f"    • Avg. Decision Time: {avg_decision_times[i]:.1f} ms")
        logger.info("\nGame Statistics:")
        logger.info(f"  • Avg. Winning Hand: {avg_winning_hand:.1f} points")
        logger.info(f"  • Avg. Turns/Round: {avg_turns_per_round:.1f}")
        logger.info(f"  • Successful Calls: {len(successful_calls)} / {total_rounds} ({len(successful_calls)/total_rounds*100:.1f}%)")
        logger.info(f"\nFiles saved: game_metrics_rounds_{total_rounds}.csv, game_cohens_d_rounds_{total_rounds}.csv, game_results_rounds_{total_rounds}.json")
    return results

def test_hand_size_and_discards():
    """Tests hand size consistency, discard behavior, and low-card determinization."""
    logger.setLevel(logging.DEBUG)
    shared_coins = [STARTING_COINS] * 2
    shared_history = []
    game = DhumbalGame(num_players=2, shared_coins=shared_coins, shared_history=shared_history)
    shared_decision_times = {0: [], 1: []}
    ai_players = [SearchBasedAI(0, AIStyle.ISMCTS, shared_decision_times[0]),
                  SearchBasedAI(1, AIStyle.ISMCTS, shared_decision_times[1])]
    env = DhumbalEnv(game, ai_players, 0)
    hands = [
        [Card('♠', '2'), Card('♥', '3'), Card('♦', '4'), Card('♣', '5'), Card('♠', '6')],
        [Card('♠', '9'), Card('♥', '9'), Card('♦', '9'), Card('♣', '10'), Card('♥', 'J')]
    ]
    env.hands = hands
    env.reset()
    env.state.hands[0] = hands[0][:]
    env.belief_state.opponent_hand_sizes = [5, 5]
    logger.debug(f"Initial hand sizes: {env.belief_state.opponent_hand_sizes}")
    discard = [Card('♠', '9'), Card('♥', '9'), Card('♦', '9')]
    game_state = GameState(
        round_number=1,
        current_player=1,
        hands=[[], [Card('♣', '10'), Card('♥', 'J')]],
        discard_pile=[Card('♠', '6')] + discard,
        deck_size=42,
        player_coins=[STARTING_COINS] * 2,
        turn_count=1,
        phase='pick',
        belief_state=env.belief_state
    )
    env.set_state(game_state)
    env.update_belief(1, discard, 'discard')
    logger.debug(f"Hand sizes after discard: {env.belief_state.opponent_hand_sizes}")
    assert env.belief_state.opponent_hand_sizes[1] == 2, f"Hand size should be 2, got {env.belief_state.opponent_hand_sizes[1]}"
    env.update_belief(1, 'deck', 'pick')
    game_state.hands[1].append(Card('♣', '2'))
    logger.debug(f"Hand sizes after pick: {env.belief_state.opponent_hand_sizes}")
    logger.debug(f"Player 1 hand: {[str(c) for c in game_state.hands[1]]}")
    assert env.belief_state.opponent_hand_sizes[1] == 3, f"Hand size after pick should be 3, got {env.belief_state.opponent_hand_sizes[1]}"
    det_hands = env._determinize_hands(game_state)
    logger.debug(f"Determinized hands: {[len(h) for h in det_hands]}")
    logger.debug(f"Determinized hand for Player 1: {[str(c) for c in det_hands[1]]}")
    assert len(det_hands[1]) <= 3, f"Determinized hand size should be <= 3, got {len(det_hands[1])}"
    empty_state = GameState(
        round_number=1,
        current_player=1,
        hands=[[], []],
        discard_pile=[Card('♠', '6')],
        deck_size=42,
        player_coins=[STARTING_COINS] * 2,
        turn_count=1,
        phase='discard',
        belief_state=env.belief_state
    )
    env.set_state(empty_state)
    discard = ai_players[1].choose_discard([], empty_state, game)
    logger.debug(f"Discard for empty hand: {discard}")
    assert discard == [], f"Expected empty discard for empty hand, got {discard}"
    low_card_state = GameState(
        round_number=1,
        current_player=0,
        hands=[hands[0][:], []],
        discard_pile=[Card('♠', '6')] + discard,
        deck_size=2,
        player_coins=[STARTING_COINS] * 2,
        turn_count=1,
        phase='discard',
        belief_state=BeliefState(
            possible_opponent_hands=[set(game.full_deck) - set(discard), set(game.full_deck) - set(discard)],
            known_not_in_hands=[set(discard), set(discard)],
            num_players=2,
            opponent_hand_sizes=[5, 3]
        )
    )
    env.set_state(low_card_state)
    env.cards_seen = set(discard)
    det_hands = env._determinize_hands(low_card_state)
    logger.debug(f"Low-card determinized hands: {[len(h) for h in det_hands]}")
    logger.debug(f"Low-card determinized hand for Player 1: {[str(c) for c in det_hands[1]]}")
    assert len(det_hands[1]) <= 3, f"Determinized hand size for low cards should be <= 3, got {len(det_hands[1])}"
    logger.info("All tests passed!")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    try:
        test_hand_size_and_discards()
        shared_coins = [STARTING_COINS] * 2
        shared_history = []
        game = DhumbalGame(num_players=2, shared_coins=shared_coins, shared_history=shared_history)
        ai_styles = [AIStyle.MCTS, AIStyle.ISMCTS]
        shared_decision_times = {i: [] for i in range(game.num_players)}
        ai_players = [SearchBasedAI(i, ai_styles[i], shared_decision_times[i]) for i in range(game.num_players)]
        logger.info("🃏 DHUMBAL (Jhyap) SEARCH-BASED AI SIMULATION")
        logger.info("=" * 70)
        logger.info(f"Configuration: {NUM_ROUNDS} rounds, {game.num_players} players, seed=42")
        logger.info(f"Starting Coins/Player: {STARTING_COINS:,}")
        logger.info(f"MCTS Iterations: {MCTS_ITERATIONS}, ISMCTS Determinizations: {ISMCTS_DETERMINIZATIONS}")
        logger.info(f"Max Decision Time: {MAX_DECISION_TIME}s")
        logger.info("\nAI Agents:")
        for ai in ai_players:
            logger.info(f"  • {ai.name}: {ai.style.value.upper()}")
        results = simulate_game(game, ai_players, verbose=True, debug=False)
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")