"""
Comprehensive Dhumbal (Jhyap) Card Game Implementation with Rule-Based Agents
============================================================================

A complete, rule-compliant implementation of the Dhumbal card game with four rule-based AI agents
(Aggressive, Conservative, Balanced, Opportunistic) as specified in the methodology section.

Game Rules (per Methodology Section 3.1):
- 2-5 players, each dealt 5 cards from a standard 52-card deck
- Goal: Achieve lowest hand value (≤ 10 points) to call "Jhyap"
- Card values: A=1, 2-10=face value, J=11, Q=12, K=13
- Valid discards: Single cards, same-rank sets (2+ cards), consecutive same-suit sequences (3+ cards)
- Turn: Discard first, then pick from top of discard pile or deck
- Scoring: Winner receives coins equal to opponents' hand values; failed Jhyap callers pay sum of all hand values
- Round ends: When a player calls Jhyap (hand value ≤ 10) or deck is exhausted
- Tie handling: Caller wins only if uniquely lowest; otherwise, lowest non-caller wins

AI Agents (per Methodology Section 3.2.1):
- Aggressive: Calls Jhyap at ≤ 10 points, prioritizes high-value discards, picks up cards ≤ 4 points
- Conservative: Calls Jhyap at ≤ 7 points, selective discards/pickups
- Balanced: Probabilistic Jhyap calls (100% at ≤5 points, 70% at 6-8 points, 40% at 9-10 points)
- Opportunistic: Adapts strategy based on coin balance (aggressive when ahead, conservative when behind)

Implementation Details (per Methodology Section 3.7):
- Python 3.9 with NumPy and SciPy
- Fixed random seed (42) for reproducibility
- Comprehensive logging in JSON format per round
- Statistical analysis with Cohen's d, T-test p-values, and Bonferroni correction, saved in CSV and JSON formats
- Modular design for research purposes
"""

import random
import itertools
import json
import logging
import csv
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.stats import ttest_ind

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Game constants
MAX_PLAYERS = 5
MIN_PLAYERS = 2
HAND_SIZE = 5
JHYAP_THRESHOLD = 10
STARTING_COINS = 10000
MAX_TURNS = 100
MIN_DISCARD_PILE_SIZE = 2
MAX_PAYMENT = 100
AI_CONSERVATIVE_CALL = 7
AI_BALANCED_CALL_1 = 8
AI_BALANCED_CALL_2 = 10
AI_BALANCED_PROB_1 = 0.7
AI_BALANCED_PROB_2 = 0.4

class AIStyle(Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    OPPORTUNISTIC = "opportunistic"

@dataclass
class GameState:
    round_number: int
    current_player: int
    hands: List[List['Card']]
    discard_pile: List['Card']
    deck_size: int
    player_coins: List[int]
    turn_count: int

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
        """Convert RoundResult to a JSON-serializable dictionary."""
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
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self) -> int:
        return hash((self.suit, self.rank))

class DhumbalGame:
    def __init__(self, num_players: int = 4):
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(f"Dhumbal requires {MIN_PLAYERS}-{MAX_PLAYERS} players")
        self.num_players = num_players
        self.player_coins = [STARTING_COINS] * num_players
        self.round_number = 0
        self.game_history: List[RoundResult] = []
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
            # Check if Ace is present and ensure it's at the start
            if 'A' in [card.rank for card in cards]:
                if positions[0] != 0:  # Ace must be at the lowest position
                    return False
            return all(positions[i] == positions[i-1] + 1 for i in range(1, len(positions)))
        except KeyError:
            return False

    def validate_discard(self, cards: List[Card]) -> bool:
        if len(cards) == 0:
            return False
        if len(cards) == 1:
            return True
        return self.validate_same_rank_set(cards) or self.validate_sequence(cards)

    def get_active_players(self) -> List[int]:
        return [i for i, coins in enumerate(self.player_coins) if coins > 0]

    def is_game_over(self) -> bool:
        return len(self.get_active_players()) < MIN_PLAYERS

class RuleBasedAI:
    def __init__(self, player_id: int, style: AIStyle, name: str = None):
        self.player_id = player_id
        self.style = style
        self.name = name or f"AI_{style.value}_{player_id}"
        self.cards_seen: List[Card] = []
        self.strategy_params = self._initialize_strategy_params()

    def _initialize_strategy_params(self) -> Dict[str, float]:
        base_params = {
            'pickup_threshold': 4,
            'discard_high_preference': 0.8,
            'multi_card_bonus': 2.0,
            'sequence_bonus': 3.0,
            'jhyap_threshold_strict': AI_CONSERVATIVE_CALL,
            'jhyap_threshold_risky': JHYAP_THRESHOLD,
            'jhyap_prob_base': 0.6,
            'risk_adjustment': 1.0
        }
        if self.style == AIStyle.AGGRESSIVE:
            base_params.update({
                'pickup_threshold': 4,
                'discard_high_preference': 1.0,
                'jhyap_threshold_strict': JHYAP_THRESHOLD,
                'jhyap_prob_base': 0.8,
                'risk_adjustment': 1.2
            })
        elif self.style == AIStyle.CONSERVATIVE:
            base_params.update({
                'pickup_threshold': 3,
                'discard_high_preference': 0.6,
                'jhyap_threshold_strict': AI_CONSERVATIVE_CALL,
                'jhyap_prob_base': 0.3,
                'risk_adjustment': 0.8
            })
        elif self.style == AIStyle.BALANCED:
            base_params.update({
                'jhyap_prob_base': AI_BALANCED_PROB_1,
                'jhyap_threshold_strict': AI_BALANCED_CALL_1,
                'jhyap_threshold_risky': AI_BALANCED_CALL_2,
                'jhyap_prob_risky': AI_BALANCED_PROB_2
            })
        elif self.style == AIStyle.OPPORTUNISTIC:
            base_params.update({
                'jhyap_prob_base': 0.5,
                'risk_adjustment': 1.0
            })
        return base_params

    def analyze_hand(self, hand: List[Card]) -> Dict[str, Any]:
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

    def _find_same_rank_groups(self, hand: List[Card]) -> List[List[Card]]:
        rank_groups = defaultdict(list)
        for card in hand:
            rank_groups[card.rank].append(card)
        groups = []
        for cards in rank_groups.values():
            if len(cards) >= 2:
                for size in range(2, len(cards) + 1):
                    groups.extend(list(combo) for combo in itertools.combinations(cards, size))
        return groups

    def _find_sequences(self, hand: List[Card]) -> List[List[Card]]:
        sequences = []
        suit_groups = defaultdict(list)
        rank_order = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        for card in hand:
            suit_groups[card.suit].append(card)
        for suit, cards in suit_groups.items():
            if len(cards) < 3:
                continue
            cards.sort(key=lambda x: rank_order.index(x.rank))
            current_seq = [cards[0]]
            for i in range(1, len(cards)):
                if rank_order.index(cards[i].rank) == rank_order.index(cards[i-1].rank) + 1:
                    current_seq.append(cards[i])
                else:
                    if len(current_seq) >= 3:
                        sequences.append(current_seq[:])
                    current_seq = [cards[i]]
            if len(current_seq) >= 3:
                sequences.append(current_seq)
        return sequences

    def _calculate_improvement_potential(self, analysis: Dict) -> float:
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

    def choose_discard(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> List[Card]:
        analysis = self.analyze_hand(hand)
        if analysis['can_call_jhyap']:
            return self._choose_jhyap_level_discard(hand, analysis)
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
            best_discard = max(discard_options, key=lambda x: x[1])[0]
            return best_discard
        return [max(hand, key=lambda x: x.value)] if hand else []

    def _choose_jhyap_level_discard(self, hand: List[Card], analysis: Dict) -> List[Card]:
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

    def _score_discard_option(self, discard: List[Card], remaining_value: int, analysis: Dict) -> float:
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

    def should_pick_from_discard(self, available_cards: List[Card], current_hand: List[Card], game_state: GameState) -> Tuple[bool, Optional[Card]]:
        if not available_cards:
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
                return True, best_card
            return True, best_card
        return False, None

    def _helps_with_combinations(self, new_card: Card, hand: List[Card]) -> bool:
        for card in hand:
            if card.rank == new_card.rank:
                return True
        rank_order = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        try:
            new_pos = rank_order.index(new_card.rank)
            for card in hand:
                if card.suit == new_card.suit:
                    pos = rank_order.index(card.rank)
                    if abs(pos - new_pos) == 1:
                        return True
        except ValueError:
            pass
        return False

    def should_call_jhyap(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> bool:
        hand_value = sum(card.value for card in hand)
        if hand_value > JHYAP_THRESHOLD:
            return False
        if self.style == AIStyle.AGGRESSIVE:
            return hand_value <= JHYAP_THRESHOLD
        elif self.style == AIStyle.CONSERVATIVE:
            return hand_value <= AI_CONSERVATIVE_CALL
        elif self.style == AIStyle.BALANCED:
            if hand_value <= 5:
                return True
            elif hand_value <= AI_BALANCED_CALL_1:
                return random.random() < AI_BALANCED_PROB_1
            elif hand_value <= AI_BALANCED_CALL_2:
                return random.random() < AI_BALANCED_PROB_2
            return False
        elif self.style == AIStyle.OPPORTUNISTIC:
            my_coins = game.player_coins[self.player_id]
            avg_coins = sum(game.player_coins) / game.num_players
            if my_coins > avg_coins:
                self.strategy_params['risk_adjustment'] = 1.2
                self.strategy_params['jhyap_prob_base'] = 0.8
            else:
                self.strategy_params['risk_adjustment'] = 0.8
                self.strategy_params['jhyap_prob_base'] = 0.3
            if hand_value <= self.strategy_params['jhyap_threshold_strict']:
                return True
            if hand_value <= self.strategy_params['jhyap_threshold_risky']:
                prob = self.strategy_params['jhyap_prob_base']
                prob += (JHYAP_THRESHOLD - hand_value) * 0.05
                if game_state.turn_count > 20:
                    prob += 0.15
                return random.random() < prob
        return False

def simulate_round(game: DhumbalGame, ai_players: List[RuleBasedAI], verbose: bool = False, debug: bool = False) -> RoundResult:
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
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
        turn_count=0
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
    discards_per_player = [0] * game.num_players
    while game_state.turn_count < MAX_TURNS:
        ai = ai_players[current_player]
        player_hand = hands[current_player]
        game_state.current_player = current_player
        game_state.deck_size = len(deck)
        game_state.hands = [hand[:] for hand in hands]
        game_state.turn_count += 1
        if verbose:
            logger.info(f"\n--- Turn {game_state.turn_count}: Player {current_player} ({ai.name}) ---")
            logger.info(f"Hand: {[str(c) for c in player_hand]} (value: {game.calculate_hand_value(player_hand)})")
            if discard_pile:
                logger.info(f"Discard top: {discard_pile[-1]}")
        if game.can_call_jhyap(player_hand) and ai.should_call_jhyap(player_hand, game_state, game):
            if verbose:
                logger.info(f"Player {current_player} calls JHYAP with {game.calculate_hand_value(player_hand)} points!")
            return end_round(game, hands, current_player, game_state.turn_count, discard_pile, discards_per_player, verbose, debug)
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
        discards_per_player[current_player] += len(cards_to_discard)
        discard_pile.extend(cards_to_discard)
        for other_ai in ai_players:
            if other_ai.player_id != current_player:
                other_ai.cards_seen.extend(cards_to_discard)
        if verbose:
            logger.info(f"Discarded: {[str(c) for c in cards_to_discard]}")
        if not player_hand:
            if verbose:
                logger.info(f"Player {current_player} has no cards left, ending round")
            return end_round(game, hands, current_player, game_state.turn_count, discard_pile, discards_per_player, verbose, debug)
        top_discard = discard_pile[-1] if discard_pile else None
        should_pick, specific_card = ai.should_pick_from_discard([top_discard] if top_discard else [], player_hand, game_state)
        if should_pick and top_discard:
            player_hand.append(discard_pile.pop())
            if verbose:
                logger.info(f"Picked from discard: {top_discard}")
        else:
            if not deck and len(discard_pile) > 1:  # Ensure at least one card remains in discard pile
                top = discard_pile.pop() if discard_pile else None
                random.shuffle(discard_pile)
                deck.extend(discard_pile[:])
                discard_pile[:] = [top] if top else []
                if debug:
                    logger.debug(f"Reshuffled discard pile into deck, new deck size: {len(deck)}")
            elif not deck and len(discard_pile) <= 1:
                if verbose:
                    logger.info("Insufficient cards in discard pile to reshuffle, ending round")
                hand_values = [game.calculate_hand_value(hand) for hand in hands]
                caller = hand_values.index(min(hand_values))
                return end_round(game, hands, caller, game_state.turn_count, discard_pile, discards_per_player, verbose, debug)
            if deck:
                picked_card = deck.pop()
                player_hand.append(picked_card)
                if verbose:
                    logger.info(f"Picked from deck: {picked_card}")
        if not deck and not discard_pile and not any(len(h) > 0 for h in hands):
            if verbose:
                logger.info("No cards remain, ending round")
            hand_values = [game.calculate_hand_value(hand) for hand in hands]
            caller = hand_values.index(min(hand_values))
            return end_round(game, hands, caller, game_state.turn_count, discard_pile, discards_per_player, verbose, debug)
        current_player = (current_player + 1) % game.num_players
    if verbose:
        logger.info("\nRound exceeded maximum turns, forcing showdown...")
    hand_values = [game.calculate_hand_value(hand) for hand in hands]
    caller = hand_values.index(min(hand_values))
    return end_round(game, hands, caller, game_state.turn_count, discard_pile, discards_per_player, verbose, debug)

def end_round(game: DhumbalGame, hands: List[List[Card]], caller: int, turns_played: int, discard_pile: List[Card], discards_per_player: List[int], verbose: bool = False, debug: bool = False) -> RoundResult:
    hand_values = [game.calculate_hand_value(hand) for hand in hands]
    caller_value = hand_values[caller]
    min_value = min(hand_values)
    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
    non_caller_min = [i for i in min_value_players if i != caller]
    if non_caller_min:
        winner = non_caller_min[0]  # Select first non-caller with minimum value
    else:
        winner = caller  # Only caller has minimum value
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
        logger.info(f"Final balances: {[f'P{i}:{game.player_coins[i]}' for i in range(game.num_players)]}")
    if debug:
        logger.debug(f"End round state: hands={[len(h) for h in hands]}, deck_size={len(discard_pile)}, discard_size={len(discard_pile)}")
    result = RoundResult(
        round_number=game.round_number,
        caller=caller,
        winner=winner,
        hand_values=hand_values,
        coin_changes=coin_changes,
        final_coins=game.player_coins.copy(),
        turns_played=turns_played,
        successful_call=successful_call,
        hands=[hand[:] for hand in hands]
    )
    game.game_history.append(result)
    return result

def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    pooled_std = np.sqrt(((len(group1)-1)*np.var(group1) + (len(group2)-1)*np.var(group2)) /
                        (len(group1)+len(group2)-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0.0

def simulate_full_game(num_players: int = 4, max_rounds: int = 2000, verbose: bool = True, debug: bool = False) -> Dict[str, Any]:
    random.seed(42)
    np.random.seed(42)
    game = DhumbalGame(num_players)
    ai_styles = [AIStyle.AGGRESSIVE, AIStyle.CONSERVATIVE, AIStyle.BALANCED, AIStyle.OPPORTUNISTIC]
    ai_players = [RuleBasedAI(i, ai_styles[i % len(ai_styles)], f"Player_{i}_{ai_styles[i % len(ai_styles)].value}") for i in range(num_players)]
    if verbose:
        logger.info("🃏 DHUMBAL RULE-BASED TOURNAMENT")
        logger.info("=" * 60)
        logger.info(f"Players: {num_players}")
        logger.info(f"Starting coins: {STARTING_COINS}")
        logger.info(f"Maximum rounds: {max_rounds}")
        logger.info("\nAI PLAYERS:")
        for ai in ai_players:
            logger.info(f"  {ai.name} - Style: {ai.style.value}")
        logger.info("")
    round_results = []
    while not game.is_game_over() and game.round_number < max_rounds:
        try:
            result = simulate_round(game, ai_players, verbose, debug)
            round_results.append(result)
            bankrupt_players = [i for i, coins in enumerate(game.player_coins) if coins <= 0]
            if bankrupt_players and verbose:
                for player in bankrupt_players:
                    logger.info(f"Player {player} ({ai_players[player].name}) is bankrupt!")
        except Exception as e:
            if verbose:
                logger.info(f"Error in round {game.round_number + 1}: {e}")
            break
    final_results = analyze_game_results(game, ai_players, round_results, verbose)
    return final_results

def analyze_game_results(game: DhumbalGame, ai_players: List[RuleBasedAI], round_results: List[RoundResult], verbose: bool = True) -> Dict[str, Any]:
    if not round_results:
        return {"error": "No rounds completed"}
    total_rounds = len(round_results)
    final_coins = game.player_coins.copy()
    winner_id = max(range(game.num_players), key=lambda i: final_coins[i])
    winner_counts = Counter(r.winner for r in round_results)
    caller_counts = Counter(r.caller for r in round_results)
    successful_calls = [r for r in round_results if r.successful_call]
    success_rates = {}
    jhyap_calls = [[] for _ in range(game.num_players)]
    hand_values = []
    cards_discarded = [0.0] * game.num_players
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
            game_state = GameState(
                round_number=r.round_number,
                current_player=i,
                hands=r.hands,
                discard_pile=[],
                deck_size=0,
                player_coins=r.final_coins,
                turn_count=r.turns_played
            )
            discard = ai_players[i].choose_discard(r.hands[i], game_state, game)
            cards_data[i].append(len(discard) if discard else 0)
            if r.caller == i:
                jhyap_calls[i].append(r.hand_values[i])
                risk_data[i].append(1 if r.successful_call else 0)
        hand_values.extend(r.hand_values)
    for i in range(game.num_players):
        calls_made = caller_counts.get(i, 0)
        successful = len([r for r in successful_calls if r.caller == i])
        success_rates[i] = (successful / calls_made * 100) if calls_made > 0 else 0
    avg_winning_hand = sum(min(r.hand_values) for r in round_results) / total_rounds
    avg_turns_per_round = sum(r.turns_played for r in round_results) / total_rounds
    total_coins_transferred = sum(sum(abs(change) for change in r.coin_changes) for r in round_results) / 2
    win_rates = [winner_counts.get(i, 0) / total_rounds * 100 for i in range(game.num_players)]
    win_ci = [1.96 * np.std(win_data[i]) / np.sqrt(total_rounds) * 100 for i in range(game.num_players)]
    economic_performance = [sum(r.coin_changes[i] for r in round_results) / total_rounds for i in range(game.num_players)]
    jhyap_success_rates = [success_rates[i] for i in range(game.num_players)]
    cards_discarded = [sum(cards_data[i]) / total_rounds for i in range(game.num_players)]
    risk_assessment = []
    for i in range(game.num_players):
        calls = jhyap_calls[i]
        successes = risk_data[i]
        if len(calls) >= 2 and len(calls) == len(successes):
            risk_assessment.append(np.corrcoef(calls, successes)[0,1])
        else:
            risk_assessment.append(None)  # Indicate insufficient data
    # Cohen's d and T-test p-values with Bonferroni correction
    cohens_d = {}
    p_values = {}
    metrics = ['win', 'economic', 'jhyap', 'cards', 'risk']
    data_lists = [win_data, economic_data, jhyap_data, cards_data, [[r if r is not None else 0 for r in risk_data[i]] if risk_assessment[i] is not None else [0]*total_rounds for i in range(game.num_players)]]
    comparisons = [(i, j) for i in range(game.num_players) for j in range(i+1, game.num_players)]
    adjusted_alpha = 0.05 / len(comparisons) if comparisons else 0.05
    for m, metric in enumerate(metrics):
        cohens_d[metric] = {}
        p_values[metric] = {}
        for i, j in comparisons:
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
            'cards_discarded': cards_discarded,
            'risk_assessment': risk_assessment
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
    # Save to CSV
    with open('metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Player', 'Win Rate', 'Win CI', 'Economic Performance', 'Jhyap Success Rate', 'Cards Discarded', 'Risk Assessment'])
        for i in range(game.num_players):
            risk_val = risk_assessment[i] if risk_assessment[i] is not None else 'N/A'
            writer.writerow([ai_players[i].name, win_rates[i], win_ci[i], economic_performance[i], jhyap_success_rates[i], cards_discarded[i], risk_val])
    with open('statistical_analysis.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Comparison', 'Win d', 'Win p-value', 'Economic d', 'Economic p-value', 'Jhyap d', 'Jhyap p-value', 'Cards d', 'Cards p-value', 'Risk d', 'Risk p-value'])
        for comp in comparisons:
            comp_key = f'P{comp[0]} vs P{comp[1]}'
            row = [comp_key]
            for metric in metrics:
                d = cohens_d[metric][comp_key]
                p = p_values[metric][comp_key] if p_values[metric][comp_key] is not None else 'N/A'
                row.extend([d, p])
            writer.writerow(row)
    # Save full results to JSON
    with open('full_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info("FINAL TOURNAMENT ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"🏆 WINNER: {results['game_summary']['winner_name']}")
        logger.info(f"Rounds played: {total_rounds}")
        logger.info(f"Total coins transferred: {total_coins_transferred}")
        logger.info("\nFINAL STANDINGS:")
        for rank, player_id in enumerate(sorted(range(game.num_players), key=lambda i: final_coins[i], reverse=True), 1):
            coins = final_coins[player_id]
            change = coins - STARTING_COINS
            change_str = f"(+{change})" if change > 0 else f"({change})" if change < 0 else "(±0)"
            logger.info(f"  {rank}. {ai_players[player_id].name}: {coins} coins {change_str}")
        logger.info("\nPLAYER STATISTICS:")
        for i in range(game.num_players):
            logger.info(f"  {ai_players[i].name}:")
            logger.info(f"    Win rate: {win_rates[i]:.1f}% ± {win_ci[i]:.1f}%")
            logger.info(f"    Economic performance: {economic_performance[i]:.1f} coins/game")
            logger.info(f"    Jhyap success rate: {jhyap_success_rates[i]:.1f}%")
            logger.info(f"    Strategic depth: {cards_discarded[i]:.1f} cards/turn")
            risk_val = risk_assessment[i] if risk_assessment[i] is not None else 'N/A'
            logger.info(f"    Risk assessment (corr): {risk_val}")
        logger.info("\nGAME STATISTICS:")
        logger.info(f"  Average winning hand value: {avg_winning_hand:.1f} points")
        logger.info(f"  Average round length: {avg_turns_per_round:.1f} turns")
        logger.info(f"  Successful Jhyap calls: {len(successful_calls)}/{total_rounds}")
        logger.info(f"\nSTATISTICAL ANALYSIS (Bonferroni adjusted α = {adjusted_alpha:.4f}):")
        for metric in metrics:
            logger.info(f"  {metric.capitalize()}:")
            for comp in comparisons:
                comp_key = f'P{comp[0]} vs P{comp[1]}'
                d = cohens_d[metric][comp_key]
                p = p_values[metric][comp_key] if p_values[metric][comp_key] is not None else 'N/A'
                significance = " (significant)" if p != 'N/A' and p < adjusted_alpha else ""
                logger.info(f"    {comp_key}: Cohen's d = {d:.2f}, p-value = {p if p != 'N/A' else p}{significance}")
        logger.info("\nResults saved to metrics.csv, statistical_analysis.csv, and full_results.json")
    return results

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    results = simulate_full_game(num_players=4, max_rounds=1024, verbose=True, debug=False)
    logger.info("\nRULE-BASED TOURNAMENT COMPLETE!")
    logger.info(f"Winner: {results['game_summary']['winner_name']}")
    logger.info(f"Total rounds: {results['game_summary']['total_rounds']}")
    logger.info(f"Average winning hand: {results['game_statistics']['avg_winning_hand_value']} points")