import tensorflow as tf
import numpy as np
import random
import itertools
import time
import json
from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass
from enum import Enum
from tensorflow.keras import models, layers
from scipy.stats import ttest_ind
from tqdm import tqdm
import math
import copy
import csv

NUM_PLAYERS = 4
HAND_SIZE = 5
JHYAP_THRESHOLD = 10
STARTING_COINS = 10000
MAX_TURNS = 100
MIN_DISCARD_PILE_SIZE = 2
MAX_PAYMENT = 100
NUM_ROUNDS = 1024
MCTS_ITERATIONS = 100
ISMCTS_DETERMINIZATIONS = 3
EXPLORATION_CONSTANT = math.sqrt(2)
MAX_DECISION_TIME = 1.5
ACTION_CACHE_SIZE = 1000
MAX_ACTION_SIZE = 128
STATE_SIZE = 117

class AIStyle(Enum):
    AGGRESSIVE = "aggressive"
    ISMCTS = "ismcts"
    PPO = "ppo"
    RANDOM = "random"

@dataclass
class GameState:
    round_number: int
    current_player: int
    hands: list
    discard_pile: list
    deck_size: int
    player_coins: list
    turn_count: int
    phase: str

    def copy(self):
        return GameState(
            round_number=self.round_number,
            current_player=self.current_player,
            hands=[hand[:] for hand in self.hands],
            discard_pile=self.discard_pile[:],
            deck_size=self.deck_size,
            player_coins=self.player_coins[:],
            turn_count=self.turn_count,
            phase=self.phase
        )

@dataclass
class RoundResult:
    round_number: int
    caller: int
    winner: int
    hand_values: list
    coin_changes: list
    final_coins: list
    turns_played: int
    successful_call: bool
    hands: list

    def to_dict(self):
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
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.value = self._calculate_value()

    def _calculate_value(self):
        if self.rank == 'A':
            return 1
        elif self.rank == 'J':
            return 11
        elif self.rank == 'Q':
            return 12
        elif self.rank == 'K':
            return 13
        return int(self.rank)

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))

class DhumbalGame:
    def __init__(self, num_players=NUM_PLAYERS):
        self.num_players = num_players
        self.player_coins = [STARTING_COINS] * num_players
        self.round_number = 0
        self.game_history = []
        self.SUITS = ['♠', '♥', '♦', '♣']
        self.RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.RANK_ORDER = {rank: i for i, rank in enumerate(self.RANKS)}
        self.full_deck = [Card(suit, rank) for suit in self.SUITS for rank in self.RANKS]

    def create_deck(self):
        deck = self.full_deck[:]
        random.shuffle(deck)
        return deck

    def deal_cards(self):
        deck = self.create_deck()
        hands = [[] for _ in range(self.num_players)]
        for _ in range(HAND_SIZE):
            for player in range(self.num_players):
                if deck:
                    hands[player].append(deck.pop())
        return hands, deck

    def calculate_hand_value(self, hand):
        return sum(card.value for card in hand)

    def can_call_jhyap(self, hand):
        return self.calculate_hand_value(hand) <= JHYAP_THRESHOLD

    def validate_same_rank_set(self, cards):
        if len(cards) < 2:
            return False
        return all(card.rank == cards[0].rank for card in cards)

    def validate_sequence(self, cards):
        if len(cards) < 3:
            return False
        if not all(card.suit == cards[0].suit for card in cards):
            return False
        positions = sorted([self.RANK_ORDER[card.rank] for card in cards])
        return all(positions[i] == positions[i-1] + 1 for i in range(1, len(positions)))

    def validate_discard(self, cards):
        if not cards:
            return False
        if len(cards) == 1:
            return True
        return self.validate_same_rank_set(cards) or self.validate_sequence(cards)

class BaseAI:
    def __init__(self, player_id, name):
        self.player_id = player_id
        self.name = name
        self.decision_times = []
        self.cards_seen = []

    def should_call_jhyap(self, hand, game_state, game):
        pass

    def choose_discard(self, hand, game_state, game):
        pass

    def should_pick_from_discard(self, available_cards, current_hand, game_state, game):
        pass

class RandomBaselineAI(BaseAI):
    def __init__(self, player_id, name=None):
        super().__init__(player_id, name or f"RandomBaseline_{player_id}")

    def should_call_jhyap(self, hand, game_state, game):
        start_time = time.perf_counter()
        can_call = game.can_call_jhyap(hand)
        decision = random.choice([True, False]) if can_call else False
        self.decision_times.append(time.perf_counter() - start_time)
        return decision

    def choose_discard(self, hand, game_state, game):
        start_time = time.perf_counter()
        if not hand:
            self.decision_times.append(time.perf_counter() - start_time)
            return []
        valid_discards = [[card] for card in hand]
        rank_groups = defaultdict(list)
        for card in hand:
            rank_groups[card.rank].append(card)
        for cards in rank_groups.values():
            for size in range(2, len(cards) + 1):
                valid_discards.extend(list(combo) for combo in itertools.combinations(cards, size))
        suit_groups = defaultdict(list)
        for card in hand:
            suit_groups[card.suit].append(card)
        for suit, cards in suit_groups.items():
            if len(cards) >= 3:
                cards_sorted = sorted(cards, key=lambda x: game.RANK_ORDER[x.rank])
                for size in range(3, len(cards_sorted) + 1):
                    for combo in itertools.combinations(cards_sorted, size):
                        combo_list = list(combo)
                        if game.validate_sequence(combo_list):
                            valid_discards.append(combo_list)
        discard = random.choice(valid_discards) if valid_discards else [random.choice(hand)]
        self.decision_times.append(time.perf_counter() - start_time)
        return discard

    def should_pick_from_discard(self, available_cards, current_hand, game_state, game):
        start_time = time.perf_counter()
        decision = random.choice([True, False]) if available_cards else False
        card = available_cards[0] if decision and available_cards else None
        self.decision_times.append(time.perf_counter() - start_time)
        return decision, card

class RuleBasedAI(BaseAI):
    def __init__(self, player_id, style, name=None):
        super().__init__(player_id, name or f"AI_{style.value}_{player_id}")
        self.style = style
        self.strategy_params = self._initialize_strategy_params()
        self.RANK_ORDER = {'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, 'J': 10, 'Q': 11, 'K': 12}

    def _initialize_strategy_params(self):
        return {
            'pickup_threshold': 4,
            'discard_high_preference': 1.0,
            'multi_card_bonus': 2.0,
            'sequence_bonus': 3.0,
            'jhyap_threshold_strict': JHYAP_THRESHOLD,
            'jhyap_prob_base': 0.8,
            'risk_adjustment': 1.2
        }

    def analyze_hand(self, hand):
        total_value = sum(card.value for card in hand)
        analysis = {
            'total_value': total_value,
            'high_cards': [card for card in hand if card.value >= 10],
            'low_cards': [card for card in hand if card.value <= 3],
            'same_rank_groups': self._find_same_rank_groups(hand),
            'sequences': self._find_sequences(hand),
            'can_call_jhyap': total_value <= JHYAP_THRESHOLD
        }
        analysis['improvement_potential'] = self._calculate_improvement_potential(analysis)
        return analysis

    def _find_same_rank_groups(self, hand):
        rank_groups = defaultdict(list)
        for card in hand:
            rank_groups[card.rank].append(card)
        groups = []
        for cards in rank_groups.values():
            if len(cards) >= 2:
                for size in range(2, len(cards) + 1):
                    groups.extend(list(combo) for combo in itertools.combinations(cards, size))
        return groups

    def _find_sequences(self, hand):
        sequences = []
        suit_groups = defaultdict(list)
        for card in hand:
            suit_groups[card.suit].append(card)
        for suit, cards in suit_groups.items():
            if len(cards) < 3:
                continue
            cards.sort(key=lambda x: self.RANK_ORDER[x.rank])
            current_seq = [cards[0]]
            for i in range(1, len(cards)):
                if self.RANK_ORDER[cards[i].rank] == self.RANK_ORDER[cards[i-1].rank] + 1:
                    current_seq.append(cards[i])
                else:
                    if len(current_seq) >= 3:
                        sequences.append(current_seq[:])
                    current_seq = [cards[i]]
            if len(current_seq) >= 3:
                sequences.append(current_seq)
        return sequences

    def _calculate_improvement_potential(self, analysis):
        current_value = analysis['total_value']
        if current_value <= JHYAP_THRESHOLD:
            return 0
        max_reduction = 0
        for group in analysis['same_rank_groups']:
            reduction = sum(card.value for card in group)
            max_reduction = max(max_reduction, reduction)
        for seq in analysis['sequences']:
            reduction = sum(card.value for card in seq)
            max_reduction = max(max_reduction, reduction)
        if analysis['high_cards']:
            max_reduction = max(max_reduction, max(card.value for card in analysis['high_cards']))
        return max_reduction

    def choose_discard(self, hand, game_state, game):
        start_time = time.perf_counter()
        analysis = self.analyze_hand(hand)
        if analysis['can_call_jhyap']:
            discard = self._choose_jhyap_level_discard(hand, analysis)
        else:
            discard_options = []
            for card in hand:
                remaining_hand = [c for c in hand if c != card]
                remaining_value = sum(c.value for c in remaining_hand)
                score = self._score_discard_option([card], remaining_value, analysis)
                discard_options.append(([card], score))
            for group in analysis['same_rank_groups']:
                remaining_hand = [c for c in hand if c not in group]
                remaining_value = sum(c.value for c in remaining_hand)
                score = self._score_discard_option(group, remaining_value, analysis)
                discard_options.append((group, score))
            for seq in analysis['sequences']:
                remaining_hand = [c for c in hand if c not in seq]
                remaining_value = sum(c.value for c in remaining_hand)
                score = self._score_discard_option(seq, remaining_value, analysis)
                discard_options.append((seq, score))
            if discard_options:
                discard = max(discard_options, key=lambda x: x[1])[0]
            else:
                discard = [max(hand, key=lambda x: x.value)] if hand else []
        self.decision_times.append(time.perf_counter() - start_time)
        return discard

    def _choose_jhyap_level_discard(self, hand, analysis):
        current_value = analysis['total_value']
        safe_discards = []
        for card in hand:
            remaining_value = current_value - card.value
            if remaining_value + 5 <= JHYAP_THRESHOLD:
                safe_discards.append(card)
        if safe_discards:
            return [max(safe_discards, key=lambda x: x.value)]
        for group in analysis['same_rank_groups']:
            group_value = sum(c.value for c in group)
            remaining_value = current_value - group_value
            if remaining_value + 5 <= JHYAP_THRESHOLD:
                return group
        return [min(hand, key=lambda x: x.value)] if hand else []

    def _score_discard_option(self, discard, remaining_value, analysis):
        discard_value = sum(card.value for card in discard)
        score = discard_value * self.strategy_params['discard_high_preference']
        if len(discard) > 1:
            score += len(discard) * self.strategy_params['multi_card_bonus']
        if len(discard) >= 3 and all(card.suit == discard[0].suit for card in discard):
            score += self.strategy_params['sequence_bonus']
        if remaining_value <= JHYAP_THRESHOLD:
            score += 50
        score += max(0, (analysis['total_value'] - remaining_value) / analysis['total_value']) * 10
        return score * self.strategy_params['risk_adjustment']

    def should_pick_from_discard(self, available_cards, current_hand, game_state, game):
        start_time = time.perf_counter()
        if not available_cards:
            self.decision_times.append(time.perf_counter() - start_time)
            return False, None
        current_value = sum(card.value for card in current_hand)
        pickup_threshold = self.strategy_params['pickup_threshold']
        if current_value > 15:
            pickup_threshold += 2
        elif current_value <= 12:
            pickup_threshold -= 1
        good_cards = [card for card in available_cards if card.value <= pickup_threshold]
        if good_cards:
            best_card = min(good_cards, key=lambda x: x.value)
            if best_card.value == 1 or self._helps_with_combinations(best_card, current_hand):
                self.decision_times.append(time.perf_counter() - start_time)
                return True, best_card
            self.decision_times.append(time.perf_counter() - start_time)
            return True, best_card
        self.decision_times.append(time.perf_counter() - start_time)
        return False, None

    def _helps_with_combinations(self, new_card, current_hand):
        for card in current_hand:
            if card.rank == new_card.rank:
                return True
        new_pos = self.RANK_ORDER[new_card.rank]
        for card in current_hand:
            if card.suit == new_card.suit:
                pos = self.RANK_ORDER[card.rank]
                if abs(pos - new_pos) == 1:
                    return True
        return False

    def should_call_jhyap(self, hand, game_state, game):
        start_time = time.perf_counter()
        hand_value = sum(card.value for card in hand)
        decision = hand_value <= JHYAP_THRESHOLD if hand_value <= JHYAP_THRESHOLD else False
        self.decision_times.append(time.perf_counter() - start_time)
        return decision

class BeliefState:
    def __init__(self, possible_opponent_hands, known_not_in_hands, num_players, opponent_hand_sizes):
        self.possible_opponent_hands = possible_opponent_hands
        self.known_not_in_hands = known_not_in_hands
        self.num_players = num_players
        self.opponent_hand_sizes = opponent_hand_sizes

    def copy(self):
        return BeliefState(
            [set(h) for h in self.possible_opponent_hands],
            [set(h) for h in self.known_not_in_hands],
            self.num_players,
            self.opponent_hand_sizes.copy()
        )

    def __hash__(self):
        return hash((
            tuple(tuple(sorted(str(c) for c in h)) for h in self.possible_opponent_hands),
            tuple(tuple(sorted(str(c) for c in h)) for h in self.known_not_in_hands),
            tuple(self.opponent_hand_sizes)
        ))

class SearchGameState:
    def __init__(self, round_number, current_player, hands, discard_pile, deck_size, player_coins, turn_count, phase, belief_state, done=False, winner=None, action=None):
        self.round_number = round_number
        self.current_player = current_player
        self.hands = hands
        self.discard_pile = discard_pile
        self.deck_size = deck_size
        self.player_coins = player_coins
        self.turn_count = turn_count
        self.phase = phase
        self.belief_state = belief_state
        self.done = done
        self.winner = winner
        self.action = action

    def copy(self):
        return SearchGameState(
            self.round_number,
            self.current_player,
            [hand[:] for hand in self.hands],
            self.discard_pile[:],
            self.deck_size,
            self.player_coins.copy(),
            self.turn_count,
            self.phase,
            self.belief_state.copy(),
            self.done,
            self.winner,
            self.action
        )

    def __hash__(self):
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

class SearchDhumbalEnv:
    def __init__(self, game, current_player):
        self.game = game
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
            opp_hands,
            not_in_hands,
            game.num_players,
            [len(self.hands[i]) for i in range(game.num_players)]
        )
        self.state = SearchGameState(
            game.round_number + 1,
            current_player,
            [self.hands[current_player][:] if i == current_player else [] for i in range(game.num_players)],
            self.discard_pile[:],
            len(self.deck),
            game.player_coins[:],
            0,
            'jhyap_check',
            self.belief_state
        )
        self.action_cache = OrderedDict()

    def reset(self):
        self.hands, self.deck = self.game.deal_cards()
        self.discard_pile = [self.deck.pop()] if self.deck else []
        self.cards_seen = set(self.discard_pile)
        full_deck = set(self.game.full_deck)
        known_cards = set(self.discard_pile)
        possible_cards = full_deck - known_cards
        opp_hands = [possible_cards.copy() for _ in range(self.game.num_players)]
        not_in_hands = [set() for _ in range(self.game.num_players)]
        self.belief_state = BeliefState(
            opp_hands,
            not_in_hands,
            self.game.num_players,
            [len(self.hands[i]) for i in range(self.game.num_players)]
        )
        self.state = SearchGameState(
            game.round_number + 1,
            self.current_player,
            [self.hands[self.current_player][:] if i == self.current_player else [] for i in range(self.game.num_players)],
            self.discard_pile[:],
            len(self.deck),
            game.player_coins[:],
            0,
            'jhyap_check',
            self.belief_state
        )
        self.action_cache.clear()

    def set_state(self, new_state):
        self.state = new_state.copy()
        self.hands[self.current_player] = new_state.hands[self.current_player][:]
        self.discard_pile = new_state.discard_pile[:]
        self.deck = [Card('♠', 'A')] * new_state.deck_size
        self.cards_seen = set(self.discard_pile)
        self.belief_state = new_state.belief_state.copy()
        self.belief_state.opponent_hand_sizes[self.current_player] = len(self.hands[self.current_player])

    def update_belief(self, player_id, action, phase):
        if phase == 'discard' and isinstance(action, list):
            new_size = max(0, self.belief_state.opponent_hand_sizes[player_id] - len(action))
            self.belief_state.opponent_hand_sizes[player_id] = new_size
            for i in range(self.game.num_players):
                if i != player_id:
                    self.belief_state.known_not_in_hands[i].update(action)
                    for card in action:
                        if card in self.belief_state.possible_opponent_hands[i]:
                            self.belief_state.possible_opponent_hands[i].remove(card)
        elif phase == 'pick':
            new_size = self.belief_state.opponent_hand_sizes[player_id] + 1
            self.belief_state.opponent_hand_sizes[player_id] = new_size
            if action == 'discard' and self.discard_pile:
                card = self.discard_pile[-1]
                for i in range(self.game.num_players):
                    if i != player_id:
                        self.belief_state.known_not_in_hands[i].add(card)
                        if card in self.belief_state.possible_opponent_hands[i]:
                            self.belief_state.possible_opponent_hands[i].remove(card)
        if isinstance(action, list):
            self.cards_seen.update(action)

    def get_actions(self):
        state_hash = hash(self.state)
        if state_hash in self.action_cache:
            return self.action_cache[state_hash]
        actions = []
        if self.state.phase == 'jhyap_check':
            actions = [True, False] if self.game.can_call_jhyap(self.state.hands[self.current_player]) else [False]
        elif self.state.phase == 'discard':
            hand = self.state.hands[self.current_player]
            if not hand:
                actions = []
            else:
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

    def step(self, action):
        new_state = self.state.copy()
        reward = 0.0
        done = False
        new_state.action = action

        if new_state.phase == 'jhyap_check':
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
                        reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != new_state.current_player) if winner == new_state.current_player else -min(hand_values[new_state.current_player], MAX_PAYMENT)
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
                    reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != new_state.current_player) if winner == new_state.current_player else -min(hand_values[new_state.current_player], MAX_PAYMENT)
                    self.set_state(new_state)
                    return new_state, reward, True
            new_state.turn_count += 1
            new_state.current_player = (new_state.current_player + 1) % self.game.num_players
            new_state.phase = 'jhyap_check'
            if new_state.turn_count >= MAX_TURNS:
                new_state.done = True
                det_hands = self._determinize_hands(new_state)
                hand_values = [self.game.calculate_hand_value(hand) for hand in det_hands]
                min_value = min(hand_values)
                min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                winner = min_value_players[0]
                new_state.winner = winner
                reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != new_state.current_player) if winner == new_state.current_player else -min(hand_values[new_state.current_player], MAX_PAYMENT)
                self.set_state(new_state)
                return new_state, reward, True

        self.set_state(new_state)
        return new_state, reward, done

    def _determinize_hands(self, state):
        det_hands = [state.hands[state.current_player][:] if i == state.current_player else [] for i in range(self.game.num_players)]
        full_deck = set(self.game.full_deck)
        known_cards = set(state.hands[state.current_player]) | set(state.discard_pile) | self.cards_seen
        available = list(full_deck - known_cards)
        if len(available) < sum(state.belief_state.opponent_hand_sizes[i] for i in range(self.game.num_players) if i != state.current_player):
            available.extend([card for card in state.discard_pile if card not in known_cards])
        random.shuffle(available)
        for i in range(self.game.num_players):
            if i != state.current_player:
                hand_size = state.belief_state.opponent_hand_sizes[i]
                possible_cards = list(state.belief_state.possible_opponent_hands[i] & set(available))
                random.shuffle(possible_cards)
                num_cards_to_sample = min(hand_size, len(possible_cards))
                det_hands[i] = possible_cards[:num_cards_to_sample]
                available = [c for c in available if c not in det_hands[i]]
        return det_hands

def get_utility(state, player_id, game):
    if not state.done:
        return 0.0
    det_hands = [state.hands[player_id][:] if i == player_id else [Card('♠', 'A')] * state.belief_state.opponent_hand_sizes[i]
                 for i in range(game.num_players)]
    hand_values = [game.calculate_hand_value(hand) for hand in det_hands]
    min_value = min(hand_values)
    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
    caller = None
    successful = True
    winner = min_value_players[0]
    if state.phase == 'jhyap_check' and state.action is True:
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
    if isinstance(action, bool):
        return ('jhyap_check', action)
    elif isinstance(action, str):
        return ('pick', action)
    elif isinstance(action, list):
        return ('discard', tuple(sorted(str(c) for c in action)))
    return None

class InfoSetNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.reward = 0.0
        self.action_key = action_key(action) if action is not None else None
        self.legal_count = defaultdict(int)
        self.total_determinizations = 0

    def get_or_create_child(self, action):
        key = action_key(action)
        if key not in self.children:
            self.children[key] = InfoSetNode(parent=self, action=action)
        return self.children[key]

    def ucb_select(self, legal_actions):
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

    def update(self, reward, legal_actions):
        self.visits += 1
        self.reward += reward
        self.total_determinizations += 1
        for action in legal_actions:
            key = action_key(action)
            self.legal_count[key] += 1

    def best_child(self):
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: (c.reward / c.visits if c.visits > 0 else -float('inf')) *
                   (c.legal_count.get(c.action_key, 0) / (self.total_determinizations + 1e-6)))

class MCTS:
    def __init__(self, iterations=MCTS_ITERATIONS):
        self.iterations = iterations
        self.state_cache = {}

    def search(self, state, env, player_id):
        state_hash = hash(state)
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]
        root = InfoSetNode()
        start_time = time.perf_counter()
        i = 0
        while i < self.iterations and (time.perf_counter() - start_time) < MAX_DECISION_TIME:
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

    def _determinize(self, state, env, player_id):
        det_state = state.copy()
        det_state.hands = [state.hands[player_id][:] if i == player_id else [] for i in range(env.game.num_players)]
        full_deck = set(env.game.full_deck)
        known_cards = set(state.hands[player_id]) | set(state.discard_pile) | env.cards_seen
        available = list(full_deck - known_cards)
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

    def _rollout(self, env, player_id, game):
        sim_env = copy.deepcopy(env)
        while not sim_env.state.done:
            actions = sim_env.get_actions()
            if not actions:
                return -10.0
            action = random.choice(actions)
            sim_env.step(action)
        return get_utility(sim_env.state, player_id, game)

class ISMCTS(MCTS):
    def __init__(self, iterations=MCTS_ITERATIONS, determinizations=ISMCTS_DETERMINIZATIONS):
        super().__init__(iterations)
        self.determinizations = determinizations

    def search(self, state, env, player_id):
        state_hash = hash(state)
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]
        root = InfoSetNode()
        start_time = time.perf_counter()
        i = 0
        while i < self.iterations and (time.perf_counter() - start_time) < MAX_DECISION_TIME:
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

class SearchBasedAI(BaseAI):
    def __init__(self, player_id, style, name=None):
        super().__init__(player_id, name or f"AI_{style.value}_{player_id}")
        self.style = style
        self.mcts = ISMCTS() if style == AIStyle.ISMCTS else MCTS()

    def choose_discard(self, hand, game_state, game):
        start_time = time.perf_counter()
        env = SearchDhumbalEnv(game, self.player_id)
        search_state = SearchGameState(
            game_state.round_number,
            self.player_id,
            [hand[:] if i == self.player_id else [] for i in range(game.num_players)],
            game_state.discard_pile[:],
            game_state.deck_size,
            game_state.player_coins.copy(),
            game_state.turn_count,
            'discard',
            BeliefState(
                [set(game.full_deck) for _ in range(game.num_players)],
                [set() for _ in range(game.num_players)],
                game.num_players,
                [len(game_state.hands[j]) for j in range(game.num_players)]
            )
        )
        env.set_state(search_state)
        env.cards_seen = set(self.cards_seen)
        root = self.mcts.search(env.state, env, self.player_id)
        best_child = root.best_child()
        action = best_child.action if best_child else None
        self.decision_times.append(time.perf_counter() - start_time)
        if isinstance(action, list) and game.validate_discard(action):
            return action
        return [max(hand, key=lambda x: x.value)] if hand else []

    def should_pick_from_discard(self, available_cards, current_hand, game_state, game):
        start_time = time.perf_counter()
        env = SearchDhumbalEnv(game, self.player_id)
        search_state = SearchGameState(
            game_state.round_number,
            self.player_id,
            [current_hand[:] if i == self.player_id else [] for i in range(game.num_players)],
            game_state.discard_pile[:],
            game_state.deck_size,
            game_state.player_coins.copy(),
            game_state.turn_count,
            'pick',
            BeliefState(
                [set(game.full_deck) for _ in range(game.num_players)],
                [set() for _ in range(game.num_players)],
                game.num_players,
                [len(game_state.hands[j]) for j in range(game.num_players)]
            )
        )
        env.set_state(search_state)
        env.cards_seen = set(self.cards_seen)
        root = self.mcts.search(env.state, env, self.player_id)
        best_child = root.best_child()
        action = best_child.action if best_child else None
        self.decision_times.append(time.perf_counter() - start_time)
        if action == 'discard' and game_state.discard_pile:
            return True, game_state.discard_pile[-1]
        return False, None

    def should_call_jhyap(self, hand, game_state, game):
        start_time = time.perf_counter()
        env = SearchDhumbalEnv(game, self.player_id)
        search_state = SearchGameState(
            game_state.round_number,
            self.player_id,
            [hand[:] if i == self.player_id else [] for i in range(game.num_players)],
            game_state.discard_pile[:],
            game_state.deck_size,
            game_state.player_coins.copy(),
            game_state.turn_count,
            'jhyap_check',
            BeliefState(
                [set(game.full_deck) for _ in range(game.num_players)],
                [set() for _ in range(game.num_players)],
                game.num_players,
                [len(game_state.hands[j]) for j in range(game.num_players)]
            )
        )
        env.set_state(search_state)
        env.cards_seen = set(self.cards_seen)
        root = self.mcts.search(env.state, env, self.player_id)
        best_child = root.best_child()
        action = best_child.action if best_child else None
        self.decision_times.append(time.perf_counter() - start_time)
        return bool(action)

class LearningDhumbalEnv:
    def __init__(self, game):
        self.game = game
        self.state = None
        self.action_map = {}
        self.reverse_action_map = {}

    def set_state(self, state):
        self.state = state
        self._build_action_map()

    def _build_action_map(self):
        self.action_map = {}
        self.reverse_action_map = {}
        if self.state.phase == 'jhyap_check':
            if self.game.can_call_jhyap(self.state.hands[self.state.current_player]):
                self.action_map[0] = True
                self.action_map[1] = False
                self.reverse_action_map[True] = 0
                self.reverse_action_map[False] = 1
        elif self.state.phase == 'discard':
            hand = self.state.hands[self.state.current_player]
            idx = 0
            for card in hand:
                self.action_map[idx] = [card]
                self.reverse_action_map[tuple([card])] = idx
                idx += 1
            rank_groups = defaultdict(list)
            for card in hand:
                rank_groups[card.rank].append(card)
            for cards in rank_groups.values():
                for size in range(2, len(cards) + 1):
                    for combo in itertools.combinations(cards, size):
                        combo_list = list(combo)
                        if self.game.validate_same_rank_set(combo_list):
                            self.action_map[idx] = combo_list
                            self.reverse_action_map[tuple(combo_list)] = idx
                            idx += 1
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
                                self.action_map[idx] = combo_list
                                self.reverse_action_map[tuple(combo_list)] = idx
                                idx += 1
        elif self.state.phase == 'pick':
            self.action_map[0] = 'deck'
            self.reverse_action_map['deck'] = 0
            if self.state.discard_pile:
                self.action_map[1] = 'discard'
                self.reverse_action_map['discard'] = 1

    def get_actions(self):
        return list(self.action_map.values())

    def encode_state(self):
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        hand = self.state.hands[self.state.current_player]
        for card in hand:
            suit_idx = self.game.SUITS.index(card.suit)
            rank_idx = self.game.RANKS.index(card.rank)
            state[suit_idx * 13 + rank_idx] = 1
        if self.state.discard_pile:
            top_card = self.state.discard_pile[-1]
            suit_idx = self.game.SUITS.index(top_card.suit)
            rank_idx = self.game.RANKS.index(top_card.rank)
            state[52 + suit_idx * 13 + rank_idx] = 1
        state[104] = self.state.deck_size / 52
        state[105] = self.state.turn_count / MAX_TURNS
        state[106] = self.state.player_coins[self.state.current_player] / STARTING_COINS
        for i, coins in enumerate(self.state.player_coins):
            state[107 + i] = coins / STARTING_COINS
        phase_idx = {'jhyap_check': 0, 'discard': 1, 'pick': 2}
        state[110 + phase_idx[self.state.phase]] = 1
        state[113] = sum(card.value for card in hand) / 65
        state[114] = len(hand) / HAND_SIZE
        state[115] = len(self.state.discard_pile) / 52
        state[116] = self.state.round_number / NUM_ROUNDS
        return state

    def index_to_action(self, index):
        return self.action_map.get(index, None)

class PPO:
    def __init__(self, state_size, max_action_size):
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.actor = self.build_actor()
        self.critic = self.build_critic()

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

    def act(self, state, env):
        action_space_size = len(env.get_actions())
        state = np.array(state).reshape(1, -1)
        probs = self.actor(state)[0].numpy()
        probs = probs[:action_space_size]
        probs /= np.sum(probs + 1e-10)
        return np.random.choice(action_space_size, p=probs)

class LearningBasedAI(BaseAI):
    def __init__(self, player_id, state_size, max_action_size, model_type, name=None):
        super().__init__(player_id, name or f"AI_{model_type}_{player_id}")
        self.state_size = state_size
        self.max_action_size = max_action_size
        self.model_type = model_type
        self.agent = PPO(state_size, max_action_size)
        try:
            self.agent.actor.load_weights('ppo_actor_final.weights.h5')
            self.agent.critic.load_weights('ppo_critic_final.weights.h5')
        except:
            print(f"Warning: Could not load PPO weights for {self.name}. Using untrained model.")

    def should_call_jhyap(self, hand, game_state, game):
        start_time = time.perf_counter()
        temp_env = LearningDhumbalEnv(game)
        temp_state = game_state.copy()
        temp_state.phase = 'jhyap_check'
        temp_env.set_state(temp_state)
        state = temp_env.encode_state()
        action_idx = self.agent.act(state, temp_env)
        action = temp_env.index_to_action(action_idx)
        self.decision_times.append(time.perf_counter() - start_time)
        return bool(action)

    def choose_discard(self, hand, game_state, game):
        start_time = time.perf_counter()
        temp_env = LearningDhumbalEnv(game)
        temp_state = game_state.copy()
        temp_state.phase = 'discard'
        temp_env.set_state(temp_state)
        state = temp_env.encode_state()
        action_idx = self.agent.act(state, temp_env)
        action = temp_env.index_to_action(action_idx)
        self.decision_times.append(time.perf_counter() - start_time)
        if isinstance(action, list) and game.validate_discard(action):
            return action
        return [max(hand, key=lambda x: x.value)] if hand else []

    def should_pick_from_discard(self, available_cards, current_hand, game_state, game):
        start_time = time.perf_counter()
        temp_env = LearningDhumbalEnv(game)
        temp_state = game_state.copy()
        temp_state.phase = 'pick'
        temp_env.set_state(temp_state)
        state = temp_env.encode_state()
        action_idx = self.agent.act(state, temp_env)
        action = temp_env.index_to_action(action_idx)
        self.decision_times.append(time.perf_counter() - start_time)
        if action == 'discard' and available_cards:
            return True, available_cards[0]
        return False, None

def simulate_round(game, ai_players, cards_discarded):
    game.round_number += 1
    hands, deck = game.deal_cards()
    discard_pile = [deck.pop()] if deck else []
    game_state = GameState(
        round_number=game.round_number,
        current_player=0,
        hands=[hand[:] for hand in hands],
        discard_pile=discard_pile,
        deck_size=len(deck),
        player_coins=game.player_coins.copy(),
        turn_count=0,
        phase='jhyap_check'
    )
    current_player = 0
    while game_state.turn_count < MAX_TURNS:
        ai = ai_players[current_player]
        player_hand = hands[current_player]
        game_state.current_player = current_player
        game_state.hands = [hand[:] for hand in hands]
        game_state.discard_pile = discard_pile[:]
        game_state.deck_size = len(deck)
        game_state.turn_count += 1
        game_state.phase = 'jhyap_check'

        # Step 1: Check for jhyap call
        if game.can_call_jhyap(player_hand) and ai.should_call_jhyap(player_hand, game_state, game):
            return end_round(game, hands, current_player, game_state.turn_count)

        # Step 2: Discard
        game_state.phase = 'discard'
        cards_to_discard = ai.choose_discard(player_hand, game_state, game)
        if not game.validate_discard(cards_to_discard) and player_hand:
            cards_to_discard = [max(player_hand, key=lambda x: x.value)] if player_hand else []
        for card in cards_to_discard:
            if card in player_hand:
                player_hand.remove(card)
        discard_pile.extend(cards_to_discard)
        cards_discarded[current_player] += len(cards_to_discard)
        for other_ai in ai_players:
            if other_ai.player_id != current_player:
                other_ai.cards_seen.extend(cards_to_discard)
        if not player_hand:
            return end_round(game, hands, current_player, game_state.turn_count)

        # Step 3: Pick
        game_state.phase = 'pick'
        top_discard = discard_pile[-1] if discard_pile else None
        should_pick, specific_card = ai.should_pick_from_discard([top_discard] if top_discard else [], player_hand, game_state, game)
        if should_pick and top_discard:
            player_hand.append(discard_pile.pop())
        else:
            if not deck and len(discard_pile) >= MIN_DISCARD_PILE_SIZE:
                top = discard_pile.pop() if discard_pile else None
                random.shuffle(discard_pile)
                deck.extend(discard_pile)
                discard_pile[:] = [top] if top else []
            if deck:
                picked_card = deck.pop()
                player_hand.append(picked_card)
            else:
                # If no cards are available, end the round
                hand_values = [game.calculate_hand_value(hand) for hand in hands]
                caller = hand_values.index(min(hand_values))
                return end_round(game, hands, caller, game_state.turn_count)

        if not deck and not discard_pile and not any(len(h) > 0 for h in hands):
            hand_values = [game.calculate_hand_value(hand) for hand in hands]
            caller = hand_values.index(min(hand_values))
            return end_round(game, hands, caller, game_state.turn_count)

        current_player = (current_player + 1) % game.num_players

    hand_values = [game.calculate_hand_value(hand) for hand in hands]
    caller = hand_values.index(min(hand_values))
    return end_round(game, hands, caller, game_state.turn_count)

def end_round(game, hands, caller, turns_played):
    hand_values = [game.calculate_hand_value(hand) for hand in hands]
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
                game.player_coins[i] -= payment
                game.player_coins[caller] += payment
                coin_changes[i] = -payment
                coin_changes[caller] += payment
    else:
        total_payment = sum(min(v, MAX_PAYMENT) for v in hand_values)
        game.player_coins[caller] -= total_payment
        game.player_coins[winner] += total_payment
        coin_changes[caller] = -total_payment
        coin_changes[winner] = total_payment
    result = RoundResult(
        game.round_number,
        caller,
        winner,
        hand_values,
        coin_changes,
        game.player_coins.copy(),
        turns_played,
        successful_call,
        [hand[:] for hand in hands]
    )
    game.game_history.append(result)
    return result

def calculate_cohens_d(group1, group2):
    if len(group1) < 2 or len(group2) < 2 or np.std(group1) == 0 or np.std(group2) == 0:
        return 0.0
    pooled_std = np.sqrt(((len(group1)-1)*np.var(group1) + (len(group2)-1)*np.var(group2)) / (len(group1)+len(group2)-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0.0

def simulate_game(game, ai_players, max_rounds=NUM_ROUNDS):
    print("Starting Dhumbal game simulation...")
    round_results = []
    cards_discarded = [0] * game.num_players
    for r in tqdm(range(max_rounds), desc="Simulating rounds"):
        result = simulate_round(game, ai_players, cards_discarded)
        round_results.append(result)
        if (r + 1) % 100 == 0:
            print(f"Completed {r + 1} rounds")
    print("Simulation completed. Analyzing results...")
    return analyze_game_results(game, ai_players, round_results, cards_discarded)

def analyze_game_results(game, ai_players, round_results, cards_discarded):
    total_rounds = len(round_results)
    final_coins = game.player_coins.copy()
    winner_id = max(range(game.num_players), key=lambda i: final_coins[i])
    winner_counts = Counter(r.winner for r in round_results)
    caller_counts = Counter(r.caller for r in round_results)
    successful_calls = [r for r in round_results if r.successful_call]
    success_rates = {}
    jhyap_calls = [[] for _ in range(game.num_players)]
    win_data = [[] for _ in range(game.num_players)]
    economic_data = [[] for _ in range(game.num_players)]
    jhyap_data = [[] for _ in range(game.num_players)]
    risk_data = [[] for _ in range(game.num_players)]
    avg_decision_times = [np.mean(ai.decision_times) * 1000 if ai.decision_times else 0.0 for ai in ai_players]
    
    for r in round_results:
        for i in range(game.num_players):
            win_data[i].append(1 if r.winner == i else 0)
            economic_data[i].append(r.coin_changes[i])
            jhyap_data[i].append(1 if r.caller == i else 0)
            if r.caller == i:
                jhyap_calls[i].append(r.hand_values[i])
                risk_data[i].append(1 if r.successful_call else 0)
    
    for i in range(game.num_players):
        calls_made = caller_counts.get(i, 0)
        successful = len([r for r in successful_calls if r.caller == i])
        success_rates[i] = (successful / calls_made * 100) if calls_made > 0 else 0.0
    
    avg_winning_hand = sum(min(r.hand_values) for r in round_results) / total_rounds if total_rounds > 0 else 0.0
    avg_turns_per_round = sum(r.turns_played for r in round_results) / total_rounds if total_rounds > 0 else 0.0
    total_coins_transferred = sum(sum(abs(change) for change in r.coin_changes) / 2 for r in round_results)
    win_rates = [winner_counts.get(i, 0) / total_rounds * 100 for i in range(game.num_players)]
    win_ci = [1.96 * np.std(win_data[i]) / np.sqrt(total_rounds) * 100 if total_rounds > 1 and np.std(win_data[i]) > 0 else 0.0 for i in range(game.num_players)]
    economic_performance = [sum(economic_data[i]) / total_rounds if total_rounds > 0 else 0.0 for i in range(game.num_players)]
    jhyap_success_rates = [success_rates[i] for i in range(game.num_players)]
    cards_discarded_avg = [cards / total_rounds for cards in cards_discarded]
    
    risk_assessment = []
    for i in range(game.num_players):
        calls = jhyap_calls[i]
        successes = risk_data[i]
        if len(calls) > 1 and np.std(calls) > 0 and np.std(successes) > 0:
            try:
                corr = np.corrcoef(calls, successes)[0, 1]
                risk_assessment.append(float(corr) if not np.isnan(corr) else 0.0)
            except:
                risk_assessment.append(0.0)
        else:
            risk_assessment.append(0.0)
    
    cohens_d = {}
    p_values = {}
    metrics = ['win', 'economic', 'jhyap']
    data_lists = [win_data, economic_data, jhyap_data]
    comparisons = [(i, j) for i in range(game.num_players) for j in range(i+1, game.num_players)]
    adjusted_alpha = 0.05 / len(comparisons) if comparisons else 0.05
    
    for m, metric in enumerate(metrics):
        cohens_d[metric] = {}
        p_values[metric] = {}
        for pair in comparisons:
            i, j = pair
            comp_key = f'{ai_players[i].name} vs {ai_players[j].name}'
            cohens_d[metric][comp_key] = calculate_cohens_d(data_lists[m][i], data_lists[m][j])
            if len(data_lists[m][i]) >= 2 and len(data_lists[m][j]) >= 2 and np.std(data_lists[m][i]) > 0 and np.std(data_lists[m][j]) > 0:
                try:
                    _, p_val = ttest_ind(data_lists[m][i], data_lists[m][j], equal_var=False)
                    p_values[metric][comp_key] = float(p_val) if not np.isnan(p_val) else 1.0
                except:
                    p_values[metric][comp_key] = 1.0
            else:
                p_values[metric][comp_key] = 1.0
    
    player_names = [ai.name for ai in ai_players]
    results = {
        'game_summary': {
            'total_rounds': total_rounds,
            'final_winner': winner_id,
            'winner_name': ai_players[winner_id].name,
            'final_coins': {name: coin for name, coin in zip(player_names, final_coins)},
            'starting_coins': STARTING_COINS,
            'player_names': player_names
        },
        'player_performance': {
            'win_rates': {name: round(rate, 2) for name, rate in zip(player_names, win_rates)},
            'win_ci': {name: round(ci, 2) for name, ci in zip(player_names, win_ci)},
            'economic_performance': {name: round(perf, 2) for name, perf in zip(player_names, economic_performance)},
            'jhyap_success_rates': {name: round(rate, 2) for name, rate in zip(player_names, jhyap_success_rates)},
            'risk_assessment': {name: round(risk, 2) for name, risk in zip(player_names, risk_assessment)},
            'avg_decision_times_ms': {name: round(time, 2) for name, time in zip(player_names, avg_decision_times)},
            'cards_discarded_avg': {name: round(avg, 2) for name, avg in zip(player_names, cards_discarded_avg)}
        },
        'game_statistics': {
            'avg_winning_hand_value': round(avg_winning_hand, 1),
            'avg_turns_per_round': round(avg_turns_per_round, 1),
            'successful_calls': len(successful_calls),
            'total_coins_transferred': int(total_coins_transferred)
        },
        'statistical_analysis': {
            'cohens_d': {metric: {k: round(v, 2) for k, v in pairs.items()} for metric, pairs in cohens_d.items()},
            'p_values': {metric: {k: round(v, 4) if isinstance(v, float) else v for k, v in pairs.items()} for metric, pairs in p_values.items()},
            'adjusted_alpha': round(adjusted_alpha, 4)
        },
        'round_details': [r.to_dict() for r in round_results]
    }
    
    print("Writing results to final_results.json...")
    try:
        with open('final_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Results successfully written to final_results.json")
    except Exception as e:
        print(f"Error writing results to file: {e}")
    
    with open(f'game_metrics_rounds_{total_rounds}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Player', 'Win Rate (%)', 'Win CI (%)', 'Economic Perf.', 'Jhyap Success (%)', 'Cards Disc./Round', 'Risk Corr.', 'Avg Dec. Time (ms)'])
        for i in range(game.num_players):
            risk_val = f"{risk_assessment[i]:.2f}" if risk_assessment[i] != 0 else 'N/A'
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
        for comp_key in cohens_d[metrics[0]]:
            row = [comp_key]
            for metric in metrics:
                d = cohens_d[metric][comp_key]
                p = p_values[metric][comp_key]
                row.extend([f"{d:.3f}", f"{p:.4f}" if isinstance(p, float) else p])
            writer.writerow(row)
    
    return results

if __name__ == "__main__":
    print("Initializing Dhumbal game and AI players...")
    game = DhumbalGame()
    rule_ai = RuleBasedAI(0, AIStyle.AGGRESSIVE, "Aggressive")
    search_ai = SearchBasedAI(1, AIStyle.ISMCTS, "ISMCTS")
    learning_ai = LearningBasedAI(2, STATE_SIZE, MAX_ACTION_SIZE, 'ppo', "PPO")
    random_ai = RandomBaselineAI(3, "Random")
    ai_players = [rule_ai, search_ai, learning_ai, random_ai]
    results = simulate_game(game, ai_players)
    print("Game simulation and analysis completed.")