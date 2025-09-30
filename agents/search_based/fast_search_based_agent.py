"""
Optimized Dhumbal (Jhyap) Card Game Implementation with Search-Based Agents
======================================================================================

A performance-optimized, rule-compliant implementation of the Dhumbal card game with two search-based AI agents
(MCTS, ISMCTS). Maintains all game logic and rules while improving execution speed.

Game Rules (per Methodology Section 3.1):
- 2-5 players, each dealt 5 cards from a standard 52-card deck
- Goal: Achieve lowest hand value (≤ 10 points) to call "Jhyap"
- Card values: A=1, 2-10=face value, J=11, Q=12, K=13
- Valid discards: Single cards, same-rank sets (2+ cards), consecutive same-suit sequences (3+ cards)
- Turn: Discard first, then pick from top of discard pile or deck
- Scoring: Winner receives coins equal to opponents' hand values; failed Jhyap callers pay sum of all hand values
- Round ends: When a player calls Jhyap (hand value ≤ 10) or deck is exhausted
- Tie handling: Caller wins only if uniquely lowest; otherwise, lowest non-caller wins

Search-Based Agents (per Methodology Section 3.2.2):
- MCTS: Implements UCB1 with C = √2, random rollouts, 300 iterations
- ISMCTS: Extends MCTS with 5 determinizations for imperfect information

Experimental Design (per Methodology Section 3.4.1):
- Search-Based Simulation: MCTS and ISMCTS compete in 2-player matches for 100 rounds

Optimizations:
- Enhanced action caching with granular keys
- Simplified state hashing
- Efficient data structures (sets, deques)
- Reduced redundant copying
- Asynchronous MCTS rollouts
- Precomputed discard validations
- Reduced logging overhead
- Pruned MCTS search space
- Periodic cache clearing

Author: AI Assistant
Date: September 29, 2025
"""

import random
import itertools
import json
import logging
import csv
import time
from collections import defaultdict, Counter, deque
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import math
import copy
import asyncio
import concurrent.futures

# Configure logging with minimal output
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Game constants
MAX_PLAYERS = 5
MIN_PLAYERS = 2
HAND_SIZE = 5
JHYAP_THRESHOLD = 10
STARTING_COINS = 1000000
MAX_TURNS = 100
MIN_DISCARD_PILE_SIZE = 2
MAX_PAYMENT = 100
MCTS_ITERATIONS = 300  # Reduced for faster decisions
ISMCTS_DETERMINIZATIONS = 5  # Reduced for performance
EXPLORATION_WEIGHT = math.sqrt(2)
MAX_DECISION_TIME = 1.0  # Seconds, reduced for speed
NUM_ROUNDS = 100
CACHE_CLEAR_INTERVAL = 10  # Clear caches every 10 rounds

class AIStyle(Enum):
    MCTS = "mcts"
    ISMCTS = "ismcts"

@dataclass
class GameState:
    round_number: int
    current_player: int
    hands: List[List['Card']]
    discard_pile: deque
    deck: deque
    player_coins: List[int]
    turn_count: int
    phase: str
    done: bool = False
    winner: Optional[int] = None
    action: Optional[Any] = None

    def copy(self) -> 'GameState':
        return GameState(
            round_number=self.round_number,
            current_player=self.current_player,
            hands=[hand[:] for hand in self.hands],
            discard_pile=deque(self.discard_pile),
            deck=deque(self.deck),
            player_coins=self.player_coins.copy(),
            turn_count=self.turn_count,
            phase=self.phase,
            done=self.done,
            winner=self.winner,
            action=self.action
        )

    def __hash__(self) -> int:
        # Simplified hash focusing on critical state components
        return hash((
            self.current_player,
            tuple(len(hand) for hand in self.hands),
            len(self.discard_pile),
            len(self.deck),
            self.turn_count,
            self.phase
        ))

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
        self._hash = hash((suit, rank))

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

    def __eq__(self, other) -> bool:
        return isinstance(other, Card) and self.suit == other.suit and self.rank == other.rank

    def __hash__(self) -> int:
        return self._hash

class DhumbalGame:
    def __init__(self, num_players: int = 2):
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(f"Dhumbal requires {MIN_PLAYERS}-{MAX_PLAYERS} players")
        self.num_players = num_players
        self.player_coins = [STARTING_COINS] * num_players
        self.round_number = 0
        self.game_history: List[RoundResult] = []
        self.SUITS = ['♠', '♥', '♦', '♣']
        self.RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.RANK_ORDER = {rank: i for i, rank in enumerate(self.RANKS)}
        self.valid_discards_cache: Dict[int, List[List[Card]]] = {}

    def create_deck(self) -> deque:
        deck = deque(Card(suit, rank) for suit in self.SUITS for rank in self.RANKS)
        random.shuffle(deck)
        return deck

    def deal_cards(self) -> Tuple[List[List[Card]], deque]:
        deck = self.create_deck()
        hands = [[] for _ in range(self.num_players)]
        for _ in range(HAND_SIZE):
            for player in range(self.num_players):
                if deck:
                    hands[player].append(deck.popleft())
        return hands, deck

    def calculate_hand_value(self, hand: List[Card]) -> int:
        return sum(card.value for card in hand)

    def can_call_jhyap(self, hand: List[Card]) -> bool:
        return self.calculate_hand_value(hand) <= JHYAP_THRESHOLD

    def validate_same_rank_set(self, cards: List[Card]) -> bool:
        return len(cards) >= 2 and all(card.rank == cards[0].rank for card in cards[1:])

    def validate_sequence(self, cards: List[Card]) -> bool:
        if len(cards) < 3 or not all(card.suit == cards[0].suit for card in cards[1:]):
            return False
        try:
            positions = sorted(self.RANK_ORDER[card.rank] for card in cards)
            return all(positions[i] == positions[i-1] + 1 for i in range(1, len(positions)))
        except KeyError:
            return False

    def validate_discard(self, cards: List[Card]) -> bool:
        if not cards:
            return False
        return len(cards) == 1 or self.validate_same_rank_set(cards) or self.validate_sequence(cards)

    def get_active_players(self) -> List[int]:
        return [i for i, coins in enumerate(self.player_coins) if coins > 0]

    def is_game_over(self) -> bool:
        return len(self.get_active_players()) < MIN_PLAYERS

    def precompute_valid_discards(self, hand: List[Card]) -> List[List[Card]]:
        hand_hash = hash(tuple(sorted((c.suit, c.rank) for c in hand)))
        if hand_hash in self.valid_discards_cache:
            return self.valid_discards_cache[hand_hash]
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
                cards.sort(key=lambda x: self.RANK_ORDER[x.rank])
                for size in range(3, len(cards) + 1):
                    for combo in itertools.combinations(cards, size):
                        if self.validate_sequence(list(combo)):
                            actions.append(list(combo))
        self.valid_discards_cache[hand_hash] = actions
        return actions

class DhumbalEnv:
    def __init__(self, game: DhumbalGame, ai_players: List['SearchBasedAI'], current_player: int):
        self.game = game
        self.ai_players = ai_players
        self.current_player = current_player
        self.hands, self.deck = game.deal_cards()
        self.discard_pile = deque([self.deck.popleft()]) if self.deck else deque()
        self.cards_seen = set(self.discard_pile)
        self.state = GameState(
            round_number=game.round_number + 1,
            current_player=current_player,
            hands=self.hands,
            discard_pile=self.discard_pile,
            deck=self.deck,
            player_coins=game.player_coins,
            turn_count=0,
            phase='call'
        )
        self.action_cache: Dict[int, List[Any]] = {}

    def reset(self):
        self.hands, self.deck = self.game.deal_cards()
        self.discard_pile = deque([self.deck.popleft()]) if self.deck else deque()
        self.cards_seen = set(self.discard_pile)
        self.state = GameState(
            round_number=self.game.round_number + 1,
            current_player=self.current_player,
            hands=self.hands,
            discard_pile=self.discard_pile,
            deck=self.deck,
            player_coins=self.game.player_coins,
            turn_count=0,
            phase='call'
        )
        self.action_cache.clear()

    def set_state(self, new_state: GameState):
        self.state = new_state
        self.hands = new_state.hands
        self.discard_pile = new_state.discard_pile
        self.deck = new_state.deck
        self.cards_seen = set(self.discard_pile)

    def get_actions(self) -> List[Any]:
        state_hash = hash(self.state)
        if state_hash in self.action_cache:
            return self.action_cache[state_hash]
        actions = []
        if self.state.phase == 'call':
            actions = [True, False] if self.game.can_call_jhyap(self.state.hands[self.state.current_player]) else [False]
        elif self.state.phase == 'discard':
            actions = self.game.precompute_valid_discards(self.state.hands[self.state.current_player])
        elif self.state.phase == 'pick':
            actions = ['deck']
            if self.state.discard_pile:
                actions.append('discard')
        self.action_cache[state_hash] = actions
        return actions

    def step(self, action: Any) -> Tuple[GameState, float, bool]:
        reward = 0.0
        done = False
        self.state.action = action

        if self.state.phase == 'call':
            if action:
                hand_values = [self.game.calculate_hand_value(hand) for hand in self.state.hands]
                caller = self.state.current_player
                caller_value = hand_values[caller]
                min_value = min(hand_values)
                min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
                if len(min_value_players) == 1 and min_value_players[0] == caller:
                    self.state.winner = caller
                    reward = sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != caller)
                else:
                    non_caller_min = [i for i in min_value_players if i != caller]
                    self.state.winner = min(non_caller_min) if non_caller_min else min_value_players[0]
                    reward = -sum(min(v, MAX_PAYMENT) for v in hand_values)
                self.state.done = True
                return self.state, reward, True
            self.state.phase = 'discard'

        elif self.state.phase == 'discard':
            if self.game.validate_discard(action):
                hand = self.state.hands[self.state.current_player]
                for card in action:
                    if card in hand:
                        hand.remove(card)
                self.state.discard_pile.extend(action)
                self.cards_seen.update(action)
                self.state.phase = 'pick'
            else:
                reward = -10.0
                self.state.done = True
                self.state.winner = (self.state.current_player + 1) % self.game.num_players
                return self.state, reward, True

        elif self.state.phase == 'pick':
            if action == 'discard' and self.state.discard_pile:
                card = self.state.discard_pile.pop()
                self.state.hands[self.state.current_player].append(card)
            elif action == 'deck':
                if not self.state.deck and len(self.state.discard_pile) >= MIN_DISCARD_PILE_SIZE:
                    top = self.state.discard_pile.pop() if self.state.discard_pile else None
                    random.shuffle(self.state.discard_pile)
                    self.state.deck.extend(self.state.discard_pile)
                    self.state.discard_pile = deque([top]) if top else deque()
                if self.state.deck:
                    card = self.state.deck.popleft()
                    self.state.hands[self.state.current_player].append(card)
                else:
                    self.state.done = True
                    hand_values = [self.game.calculate_hand_value(hand) for hand in self.state.hands]
                    self.state.winner = hand_values.index(min(hand_values))
                    reward = -sum(min(v, MAX_PAYMENT) for v in hand_values) if self.state.winner != self.state.current_player else sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != self.state.current_player)
                    return self.state, reward, True
            self.state.turn_count += 1
            self.state.current_player = (self.state.current_player + 1) % self.game.num_players
            self.state.phase = 'call'
            if self.state.turn_count >= MAX_TURNS:
                self.state.done = True
                hand_values = [self.game.calculate_hand_value(hand) for hand in self.state.hands]
                self.state.winner = hand_values.index(min(hand_values))
                reward = -sum(min(v, MAX_PAYMENT) for v in hand_values) if self.state.winner != self.state.current_player else sum(min(v, MAX_PAYMENT) for i, v in enumerate(hand_values) if i != self.state.current_player)
                return self.state, reward, True

        return self.state, reward, done

class Node:
    def __init__(self, state: GameState, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.untried_actions: Optional[List[Any]] = None

    def is_terminal(self) -> bool:
        return self.state.done

    def get_untried_actions(self, env: DhumbalEnv) -> List[Any]:
        if self.untried_actions is None:
            self.untried_actions = env.get_actions()
        return self.untried_actions

    def expand(self, env: DhumbalEnv) -> 'Node':
        action = self.get_untried_actions(env).pop()
        next_state, _, _ = env.step(action)
        child_node = Node(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def ucb_select(self) -> 'Node':
        log_N = math.log(self.visits) if self.visits > 0 else 0
        def ucb(child):
            return (child.reward / child.visits) + EXPLORATION_WEIGHT * math.sqrt(log_N / child.visits) if child.visits > 0 else float('inf')
        return max(self.children, key=ucb)

    def update(self, reward: float):
        self.visits += 1
        self.reward += reward

    def best_child(self) -> 'Node':
        return max(self.children, key=lambda c: c.reward / c.visits if c.visits > 0 else 0)

class MCTS:
    def __init__(self, iterations: int = MCTS_ITERATIONS):
        self.iterations = iterations
        self.state_cache: Dict[int, Any] = {}

    async def rollout(self, env: DhumbalEnv, player_id: int) -> float:
        sim_env = DhumbalEnv(env.game, env.ai_players, env.current_player)
        sim_env.set_state(copy.deepcopy(env.state))
        while not sim_env.state.done:
            actions = sim_env.get_actions()
            if not actions:
                return -10.0
            action = random.choice(actions)
            sim_env.state, reward, done = sim_env.step(action)
            if done:
                return reward if sim_env.state.winner == player_id else -reward
        return 0.0

    async def search(self, state: GameState, env: DhumbalEnv, player_id: int) -> Any:
        state_hash = hash(state)
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]
        root = Node(state)
        start_time = time.time()
        tasks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            i = 0
            while i < self.iterations and (time.time() - start_time) < MAX_DECISION_TIME:
                node = root
                sim_env = DhumbalEnv(env.game, env.ai_players, env.current_player)
                sim_env.set_state(state)
                while node.get_untried_actions(sim_env) == [] and node.children and not node.is_terminal():
                    node = node.ucb_select()
                    sim_env.set_state(node.state)
                if node.get_untried_actions(sim_env) and not node.is_terminal():
                    node = node.expand(sim_env)
                    sim_env.set_state(node.state)
                tasks.append(loop.run_in_executor(executor, lambda: self.rollout(sim_env, player_id)))
                i += 1
                # Early stopping if a dominant action emerges
                if node.visits > 50 and max(c.visits for c in root.children) / root.visits > 0.8:
                    break
            rewards = await asyncio.gather(*[task for task in tasks if not isinstance(task, float)])
            for reward in rewards:
                node = root
                while node is not None:
                    node.update(reward)
                    node = node.parent
        action = root.best_child().state.action
        self.state_cache[state_hash] = action
        return action

class ISMCTS(MCTS):
    def __init__(self, iterations: int = MCTS_ITERATIONS, determinizations: int = ISMCTS_DETERMINIZATIONS):
        super().__init__(iterations)
        self.determinizations = determinizations

    async def search(self, state: GameState, env: DhumbalEnv, player_id: int) -> Any:
        state_hash = hash(state)
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]
        root = Node(state)
        start_time = time.time()
        tasks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            i = 0
            while i < self.iterations and (time.time() - start_time) < MAX_DECISION_TIME:
                det_state = self.determinize(state, env, player_id)
                node = root
                sim_env = DhumbalEnv(env.game, env.ai_players, env.current_player)
                sim_env.set_state(det_state)
                while node.get_untried_actions(sim_env) == [] and node.children and not node.is_terminal():
                    node = node.ucb_select()
                    sim_env.set_state(node.state)
                if node.get_untried_actions(sim_env) and not node.is_terminal():
                    node = node.expand(sim_env)
                    sim_env.set_state(node.state)
                tasks.append(loop.run_in_executor(executor, lambda: self.rollout(sim_env, player_id)))
                i += 1
                if node.visits > 50 and max(c.visits for c in root.children) / root.visits > 0.8:
                    break
            rewards = await asyncio.gather(*[task for task in tasks if not isinstance(task, float)])
            for reward in rewards:
                node = root
                while node is not None:
                    node.update(reward)
                    node = node.parent
        action = root.best_child().state.action
        self.state_cache[state_hash] = action
        return action

    def determinize(self, state: GameState, env: DhumbalEnv, player_id: int) -> GameState:
        det_state = state.copy()
        unseen_cards = [c for c in env.game.create_deck() if c not in det_state.hands[player_id] and c not in det_state.discard_pile]
        total_opp_cards_needed = sum(len(det_state.hands[i]) for i in range(env.game.num_players) if i != player_id)
        if len(unseen_cards) < total_opp_cards_needed:
            unseen_cards = [c for c in env.game.create_deck() if c not in det_state.hands[player_id]]
            random.shuffle(unseen_cards)
        else:
            random.shuffle(unseen_cards)
        remaining_cards = deque(unseen_cards)
        for i in range(env.game.num_players):
            if i != player_id:
                hand_size = len(det_state.hands[i])
                det_state.hands[i] = [remaining_cards.popleft() for _ in range(min(hand_size, len(remaining_cards)))]
        det_state.deck = remaining_cards
        return det_state

class SearchBasedAI:
    def __init__(self, player_id: int, style: AIStyle):
        self.player_id = player_id
        self.style = style
        self.name = f"AI_{style.value}_{player_id}"
        self.mcts = MCTS() if style == AIStyle.MCTS else ISMCTS()
        self.cards_seen: Set[Card] = set()

    async def choose_discard(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> List[Card]:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'discard'
        env.cards_seen = self.cards_seen
        action = await self.mcts.search(env.state, env, self.player_id)
        if isinstance(action, list) and game.validate_discard(action):
            return action
        return [max(hand, key=lambda x: x.value)] if hand else []

    async def should_pick_from_discard(self, discard_pile: deque, current_hand: List[Card], game_state: GameState, game: DhumbalGame) -> Tuple[bool, Optional[Card]]:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'pick'
        env.cards_seen = self.cards_seen
        action = await self.mcts.search(env.state, env, self.player_id)
        if action == 'discard' and discard_pile:
            return True, discard_pile[-1]
        return False, None

    async def should_call_jhyap(self, hand: List[Card], game_state: GameState, game: DhumbalGame) -> bool:
        env = DhumbalEnv(game, [self], self.player_id)
        env.set_state(game_state)
        env.state.phase = 'call'
        env.cards_seen = self.cards_seen
        action = await self.mcts.search(env.state, env, self.player_id)
        return action

async def simulate_round(game: DhumbalGame, ai_players: List[SearchBasedAI], verbose: bool = False, debug: bool = False) -> RoundResult:
    if debug:
        logger.setLevel(logging.DEBUG)
    game.round_number += 1
    hands, deck = game.deal_cards()
    discard_pile = deque([deck.popleft()]) if deck else deque()
    game_state = GameState(
        round_number=game.round_number,
        current_player=0,
        hands=hands,
        discard_pile=discard_pile,
        deck=deck,
        player_coins=game.player_coins,
        turn_count=0,
        phase='call'
    )
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {game.round_number}")
        logger.info(f"Initial discard: {discard_pile[-1] if discard_pile else 'None'}")
        for i in range(game.num_players):
            logger.info(f"Player {i} ({ai_players[i].name}): {[str(c) for c in hands[i]]} (value: {game.calculate_hand_value(hands[i])})")
    current_player = 0
    discards_per_player = [0] * game.num_players
    while game_state.turn_count < MAX_TURNS and not game_state.done:
        ai = ai_players[current_player]
        player_hand = hands[current_player]
        game_state.current_player = current_player
        game_state.deck = deck
        game_state.hands = hands
        game_state.discard_pile = discard_pile
        game_state.turn_count += 1
        if verbose:
            logger.info(f"\n--- Turn {game_state.turn_count}: Player {current_player} ({ai.name}) ---")
            logger.info(f"Hand: {[str(c) for c in player_hand]} (value: {game.calculate_hand_value(player_hand)})")
            if discard_pile:
                logger.info(f"Discard top: {discard_pile[-1]}")
        game_state.phase = 'call'
        if game.can_call_jhyap(player_hand) and await ai.should_call_jhyap(player_hand, game_state, game):
            if verbose:
                logger.info(f"Player {current_player} calls JHYAP with {game.calculate_hand_value(player_hand)} points!")
            return end_round(game, hands, current_player, game_state.turn_count, discard_pile, discards_per_player, verbose, debug)
        game_state.phase = 'discard'
        cards_to_discard = await ai.choose_discard(player_hand, game_state, game)
        if debug:
            logger.debug(f"Evaluated discard: {[str(c) for c in cards_to_discard]}")
        if not game.validate_discard(cards_to_discard) and player_hand:
            cards_to_discard = [max(player_hand, key=lambda x: x.value)]
            if debug:
                logger.debug(f"Invalid discard, falling back to: {[str(c) for c in cards_to_discard]}")
        for card in cards_to_discard:
            if card in player_hand:
                player_hand.remove(card)
        discards_per_player[current_player] += len(cards_to_discard)
        discard_pile.extend(cards_to_discard)
        for ai_player in ai_players:
            ai_player.cards_seen.update(cards_to_discard)
        if verbose:
            logger.info(f"Discarded: {[str(c) for c in cards_to_discard]}")
        game_state.phase = 'pick'
        top_discard = discard_pile[-1] if discard_pile else None
        should_pick, specific_card = await ai.should_pick_from_discard(discard_pile, player_hand, game_state, game)
        if should_pick and top_discard:
            player_hand.append(discard_pile.pop())
            if verbose:
                logger.info(f"Picked from discard: {top_discard}")
        else:
            if not deck and len(discard_pile) >= MIN_DISCARD_PILE_SIZE:
                top = discard_pile.pop() if discard_pile else None
                random.shuffle(discard_pile)
                deck.extend(discard_pile)
                discard_pile = deque([top]) if top else deque()
                if debug:
                    logger.debug(f"Reshuffled discard pile into deck, new deck size: {len(deck)}")
            if deck:
                picked_card = deck.popleft()
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

def end_round(game: DhumbalGame, hands: List[List[Card]], caller: int, turns_played: int, discard_pile: deque, discards_per_player: List[int], verbose: bool = False, debug: bool = False) -> RoundResult:
    hand_values = [game.calculate_hand_value(hand) for hand in hands]
    caller_value = hand_values[caller]
    min_value = min(hand_values)
    min_value_players = [i for i, v in enumerate(hand_values) if v == min_value]
    if len(min_value_players) == 1 and min_value_players[0] == caller:
        winner = caller
    else:
        non_caller_min = [i for i in min_value_players if i != caller]
        winner = min(non_caller_min) if non_caller_min else min_value_players[0]
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
        logger.info(f"Caller: Player {caller} with {caller_value} points")
        for i, value in enumerate(hand_values):
            logger.info(f"Player {i}: {value} points {[str(c) for c in hands[i]]}{' (WINNER)' if i == winner else ''}")
        logger.info(f"{'Successful' if successful_call else 'Failed'} Jhyap call!")
        logger.info(f"Coin changes: {coin_changes}")
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
    with open(f'game_log_round_{game.round_number}.json', 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    return result

def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    pooled_std = np.sqrt(((len(group1)-1)*np.var(group1) + (len(group2)-1)*np.var(group2)) / 
                        (len(group1)+len(group2)-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0.0

async def simulate_game(game: DhumbalGame, ai_players: List[SearchBasedAI], max_rounds: int = NUM_ROUNDS, verbose: bool = True, debug: bool = False) -> Dict[str, Any]:
    round_results = []
    for round_idx in range(max_rounds):
        if game.is_game_over():
            logger.info(f"Game ended early due to bankruptcy after {round_idx} rounds")
            break
        try:
            result = await simulate_round(game, ai_players, verbose, debug)
            round_results.append(result)
            if round_idx % CACHE_CLEAR_INTERVAL == 0:
                game.valid_discards_cache.clear()
                for ai in ai_players:
                    ai.mcts.state_cache.clear()
            if verbose and len([i for i, coins in enumerate(game.player_coins) if coins <= 0]) > 0:
                logger.info(f"Bankrupt players: {[i for i, coins in enumerate(game.player_coins) if coins <= 0]}")
        except Exception as e:
            logger.info(f"Error in round {game.round_number + 1}: {e}")
            break
    return analyze_game_results(game, ai_players, round_results, verbose)

def analyze_game_results(game: DhumbalGame, ai_players: List[SearchBasedAI], round_results: List[RoundResult], verbose: bool = True) -> Dict[str, Any]:
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
    cards_data = [[] for _ in range(game.num_players)]
    win_data = [[] for _ in range(game.num_players)]
    economic_data = [[] for _ in range(game.num_players)]
    jhyap_data = [[] for _ in range(game.num_players)]
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
                discard_pile=deque(),
                deck=deque(),
                player_coins=r.final_coins,
                turn_count=r.turns_played,
                phase='discard'
            )
            discard = ai_players[i].choose_discard(r.hands[i], game_state, game)
            cards_data[i].append(len(discard) if discard else 0)
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
    cards_discarded = [sum(cards_data[i]) / total_rounds for i in range(game.num_players)] if total_rounds > 0 else [0] * game.num_players
    risk_assessment = [np.corrcoef(jhyap_calls[i], risk_data[i])[0,1] if len(jhyap_calls[i]) >= 2 else 0.0 for i in range(game.num_players)]
    cohens_d = {}
    metrics = ['win', 'economic', 'jhyap', 'cards', 'risk']
    data_lists = [win_data, economic_data, jhyap_data, cards_data, [risk_data[i] if len(risk_data[i]) >= 2 else [0]*total_rounds for i in range(game.num_players)]]
    for m, metric in enumerate(metrics):
        cohens_d[metric] = {}
        for i in range(game.num_players):
            for j in range(i+1, game.num_players):
                d = calculate_cohens_d(data_lists[m][i], data_lists[m][j])
                cohens_d[metric][f'P{i} vs P{j}'] = d
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
        'cohens_d': cohens_d,
        'round_details': [r.to_dict() for r in round_results]
    }
    with open(f'game_metrics_rounds_{total_rounds}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Player', 'Win Rate', 'Win CI', 'Economic Performance', 'Jhyap Success Rate', 'Cards Discarded', 'Risk Assessment'])
        for i in range(game.num_players):
            writer.writerow([ai_players[i].name, win_rates[i], win_ci[i], economic_performance[i], jhyap_success_rates[i], cards_discarded[i], risk_assessment[i]])
    with open(f'game_cohens_d_rounds_{total_rounds}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Comparison', 'Win d', 'Economic d', 'Jhyap d', 'Cards d', 'Risk d'])
        comparisons = list(cohens_d['win'].keys())
        for comp in comparisons:
            writer.writerow([comp, cohens_d['win'][comp], cohens_d['economic'][comp], cohens_d['jhyap'][comp], cohens_d['cards'][comp], cohens_d['risk'][comp]])
    with open(f'game_results_rounds_{total_rounds}.json', 'w') as f:
        json.dump(results, f, indent=2)
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"GAME ANALYSIS (Rounds {total_rounds})")
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
            logger.info(f"    Economic performance: {economic_performance[i]:.1f} coins/round")
            logger.info(f"    Jhyap success rate: {jhyap_success_rates[i]:.1f}%")
            logger.info(f"    Strategic depth: {cards_discarded[i]:.1f} cards/round")
            logger.info(f"    Risk assessment (corr): {risk_assessment[i]:.2f}")
        logger.info("\nGAME STATISTICS:")
        logger.info(f"  Average winning hand value: {avg_winning_hand:.1f} points")
        logger.info(f"  Average round length: {avg_turns_per_round:.1f} turns")
        logger.info(f"  Successful Jhyap calls: {len(successful_calls)}/{total_rounds}")
        logger.info(f"\nResults saved to game_metrics_rounds_{total_rounds}.csv, game_cohens_d_rounds_{total_rounds}.csv, game_results_rounds_{total_rounds}.json")
    return results

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    game = DhumbalGame(num_players=2)
    ai_styles = [AIStyle.MCTS, AIStyle.ISMCTS]
    ai_players = [SearchBasedAI(i, ai_styles[i]) for i in range(2)]
    logger.info("🃏 DHUMBAL SEARCH-BASED SIMULATION")
    logger.info("=" * 60)
    logger.info(f"Rounds: {NUM_ROUNDS}")
    logger.info(f"Players: {game.num_players}")
    logger.info(f"Starting coins: {STARTING_COINS}")
    logger.info("\nAI PLAYERS:")
    for ai in ai_players:
        logger.info(f"  {ai.name} - Style: {ai.style.value}")
    logger.info("")
    results = asyncio.run(simulate_game(game, ai_players, max_rounds=NUM_ROUNDS, verbose=True, debug=False))
    logger.info("\n✅ SEARCH-BASED SIMULATION COMPLETE!")
    logger.info(f"Winner: Player {results['game_summary']['final_winner']} ({results['game_summary']['winner_name']})")
    logger.info(f"Total rounds: {results['game_summary']['total_rounds']}")
    logger.info(f"Average winning hand: {results['game_statistics']['avg_winning_hand_value']} points")