"""
Sensitivity Analysis: Aggressive Agent Rule Contributions
==========================================================

Ablation study that isolates each rule of the Aggressive agent to measure
its individual contribution to the agent's win rate advantage.

Methodology:
    - Baseline: Standard Aggressive agent (all rules active)
    - Ablation: One rule at a time is "neutralised" to the Conservative/Balanced
      default while all other rules remain intact.
    - Metric: Win rate over 1024 rounds vs three fixed opponents
      (Conservative, Balanced, Opportunistic)
    - Statistical: Cohen's d and p-value vs baseline are reported
    - Seed: 42 for full reproducibility
"""

import random
import copy
import csv
import json
import itertools
import sys
import os
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import ttest_ind

# ─── INLINE THE FULL GAME ENGINE ─────────────────────────────────────────────
# (same constants / classes as rule_based_agent.py so this file is self-contained)

MAX_PLAYERS      = 5
MIN_PLAYERS      = 2
HAND_SIZE        = 5
JHYAP_THRESHOLD  = 10
STARTING_COINS   = 10_000
MAX_TURNS        = 100
MAX_PAYMENT      = 100
AI_CONSERVATIVE_CALL  = 7
AI_BALANCED_CALL_1    = 8
AI_BALANCED_CALL_2    = 10
AI_BALANCED_PROB_1    = 0.7
AI_BALANCED_PROB_2    = 0.4

N_ROUNDS = 1024   # rounds per experiment arm


class AIStyle(Enum):
    AGGRESSIVE    = "aggressive"
    CONSERVATIVE  = "conservative"
    BALANCED      = "balanced"
    OPPORTUNISTIC = "opportunistic"


@dataclass
class GameState:
    round_number:   int
    current_player: int
    hands:          List[List["Card"]]
    discard_pile:   List["Card"]
    deck_size:      int
    player_coins:   List[int]
    turn_count:     int


@dataclass
class RoundResult:
    round_number:    int
    caller:          int
    winner:          int
    hand_values:     List[int]
    coin_changes:    List[int]
    final_coins:     List[int]
    turns_played:    int
    successful_call: bool
    hands:           List[List["Card"]]


class Card:
    def __init__(self, suit: str, rank: str):
        self.suit  = suit
        self.rank  = rank
        self.value = self._calculate_value()

    def _calculate_value(self) -> int:
        if   self.rank == "A": return 1
        elif self.rank == "J": return 11
        elif self.rank == "Q": return 12
        elif self.rank == "K": return 13
        return int(self.rank)

    def __str__(self):  return f"{self.rank}{self.suit}"
    def __repr__(self): return str(self)
    def __eq__(self, other):
        return isinstance(other, Card) and self.suit == other.suit and self.rank == other.rank
    def __hash__(self): return hash((self.suit, self.rank))


class DhumbalGame:
    SUITS      = ["♠", "♥", "♦", "♣"]
    RANKS      = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
    RANK_ORDER = {r: i for i, r in enumerate(RANKS)}

    def __init__(self, num_players: int = 4):
        self.num_players   = num_players
        self.player_coins  = [STARTING_COINS] * num_players
        self.round_number  = 0
        self.game_history: List[RoundResult] = []

    def create_deck(self):
        deck = [Card(s, r) for s in self.SUITS for r in self.RANKS]
        random.shuffle(deck)
        return deck

    def deal_cards(self):
        deck  = self.create_deck()
        hands = [[] for _ in range(self.num_players)]
        for _ in range(HAND_SIZE):
            for p in range(self.num_players):
                if deck: hands[p].append(deck.pop())
        return hands, deck

    def calculate_hand_value(self, hand): return sum(c.value for c in hand)
    def can_call_jhyap(self, hand): return self.calculate_hand_value(hand) <= JHYAP_THRESHOLD

    def validate_same_rank_set(self, cards):
        return len(cards) >= 2 and all(c.rank == cards[0].rank for c in cards)

    def validate_sequence(self, cards):
        if len(cards) < 3: return False
        if not all(c.suit == cards[0].suit for c in cards): return False
        try:
            pos = sorted(self.RANK_ORDER[c.rank] for c in cards)
            if "A" in [c.rank for c in cards] and pos[0] != 0: return False
            return all(pos[i] == pos[i-1]+1 for i in range(1, len(pos)))
        except KeyError:
            return False

    def validate_discard(self, cards):
        if not cards: return False
        if len(cards) == 1: return True
        return self.validate_same_rank_set(cards) or self.validate_sequence(cards)

    def get_active_players(self): return [i for i, c in enumerate(self.player_coins) if c > 0]
    def is_game_over(self): return len(self.get_active_players()) < MIN_PLAYERS


# ─── RULE-BASED AI ───────────────────────────────────────────────────────────

class RuleBasedAI:
    def __init__(self, player_id: int, style: AIStyle, name: str = None,
                 param_overrides: Dict = None):
        self.player_id  = player_id
        self.style      = style
        self.name       = name or f"AI_{style.value}_{player_id}"
        self.cards_seen: List[Card] = []
        self.strategy_params = self._initialize_strategy_params()
        if param_overrides:
            self.strategy_params.update(param_overrides)

    def _initialize_strategy_params(self) -> Dict:
        base = {
            "pickup_threshold":       4,
            "discard_high_preference": 0.8,
            "multi_card_bonus":       2.0,
            "sequence_bonus":         3.0,
            "jhyap_threshold_strict": AI_CONSERVATIVE_CALL,
            "jhyap_threshold_risky":  JHYAP_THRESHOLD,
            "jhyap_prob_base":        0.6,
            "risk_adjustment":        1.0,
        }
        if self.style == AIStyle.AGGRESSIVE:
            base.update({
                "pickup_threshold":        4,
                "discard_high_preference": 1.0,
                "jhyap_threshold_strict":  JHYAP_THRESHOLD,
                "jhyap_prob_base":         0.8,
                "risk_adjustment":         1.2,
            })
        elif self.style == AIStyle.CONSERVATIVE:
            base.update({
                "pickup_threshold":        3,
                "discard_high_preference": 0.6,
                "jhyap_threshold_strict":  AI_CONSERVATIVE_CALL,
                "jhyap_prob_base":         0.3,
                "risk_adjustment":         0.8,
            })
        elif self.style == AIStyle.BALANCED:
            base.update({
                "jhyap_prob_base":        AI_BALANCED_PROB_1,
                "jhyap_threshold_strict": AI_BALANCED_CALL_1,
                "jhyap_threshold_risky":  AI_BALANCED_CALL_2,
                "jhyap_prob_risky":       AI_BALANCED_PROB_2,
            })
        elif self.style == AIStyle.OPPORTUNISTIC:
            base.update({"jhyap_prob_base": 0.5, "risk_adjustment": 1.0})
        return base

    # ── hand analysis helpers ─────────────────────────────────────────────────
    def _find_same_rank_groups(self, hand):
        rg = defaultdict(list)
        for c in hand: rg[c.rank].append(c)
        groups = []
        for cards in rg.values():
            if len(cards) >= 2:
                for sz in range(2, len(cards)+1):
                    groups.extend(list(combo) for combo in itertools.combinations(cards, sz))
        return groups

    def _find_sequences(self, hand):
        ro = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
        seqs = []
        sg = defaultdict(list)
        for c in hand: sg[c.suit].append(c)
        for suit, cards in sg.items():
            if len(cards) < 3: continue
            cards.sort(key=lambda x: ro.index(x.rank))
            cur = [cards[0]]
            for i in range(1, len(cards)):
                if ro.index(cards[i].rank) == ro.index(cards[i-1].rank)+1:
                    cur.append(cards[i])
                else:
                    if len(cur) >= 3: seqs.append(cur[:])
                    cur = [cards[i]]
            if len(cur) >= 3: seqs.append(cur)
        return seqs

    def analyze_hand(self, hand):
        tv = sum(c.value for c in hand)
        srg = self._find_same_rank_groups(hand)
        sqs = self._find_sequences(hand)
        hc  = [c for c in hand if c.value >= 10]
        mr  = 0
        for g in srg: mr = max(mr, sum(c.value for c in g))
        for s in sqs: mr = max(mr, sum(c.value for c in s))
        if hc: mr = max(mr, max(c.value for c in hc))
        return {
            "total_value":          tv,
            "high_cards":           hc,
            "low_cards":            [c for c in hand if c.value <= 3],
            "same_rank_groups":     srg,
            "sequences":            sqs,
            "can_call_jhyap":       tv <= JHYAP_THRESHOLD,
            "improvement_potential": 0 if tv <= JHYAP_THRESHOLD else mr,
        }

    # ── discard logic ─────────────────────────────────────────────────────────
    def _score_discard_option(self, discard, remaining_value, analysis):
        dv = sum(c.value for c in discard)
        sc = dv * self.strategy_params["discard_high_preference"]
        if len(discard) > 1:
            sc += len(discard) * self.strategy_params["multi_card_bonus"]
        if len(discard) >= 3 and all(c.suit == discard[0].suit for c in discard):
            sc += self.strategy_params["sequence_bonus"]
        if remaining_value <= JHYAP_THRESHOLD:
            sc += 50
        if analysis["total_value"] > 0:
            sc += max(0, (analysis["total_value"] - remaining_value) / analysis["total_value"]) * 10
        return sc * self.strategy_params["risk_adjustment"]

    def _choose_jhyap_level_discard(self, hand, analysis):
        cv   = analysis["total_value"]
        safe = [c for c in hand if (cv - c.value + 5) <= JHYAP_THRESHOLD]
        if safe: return [max(safe, key=lambda x: x.value)]
        for g in analysis["same_rank_groups"]:
            gv = sum(c.value for c in g)
            if (cv - gv + 5) <= JHYAP_THRESHOLD: return g
        return [min(hand, key=lambda x: x.value)] if hand else []

    def choose_discard(self, hand, game_state, game):
        analysis = self.analyze_hand(hand)
        if analysis["can_call_jhyap"]:
            return self._choose_jhyap_level_discard(hand, analysis)
        opts = []
        for c in hand:
            rv = sum(x.value for x in hand if x != c)
            opts.append(([c], self._score_discard_option([c], rv, analysis)))
        for g in analysis["same_rank_groups"]:
            rv = sum(x.value for x in hand if x not in g)
            opts.append((g, self._score_discard_option(g, rv, analysis)))
        for s in analysis["sequences"]:
            rv = sum(x.value for x in hand if x not in s)
            opts.append((s, self._score_discard_option(s, rv, analysis)))
        if opts: return max(opts, key=lambda x: x[1])[0]
        return [max(hand, key=lambda x: x.value)] if hand else []

    # ── pickup logic ─────────────────────────────────────────────────────────
    def _helps_with_combinations(self, new_card, hand):
        for c in hand:
            if c.rank == new_card.rank: return True
        ro = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
        try:
            np_ = ro.index(new_card.rank)
            for c in hand:
                if c.suit == new_card.suit and abs(ro.index(c.rank) - np_) == 1:
                    return True
        except ValueError:
            pass
        return False

    def should_pick_from_discard(self, available_cards, current_hand, game_state):
        if not available_cards: return False, None
        cv  = sum(c.value for c in current_hand)
        thr = self.strategy_params["pickup_threshold"]
        if cv > 15: thr += 2
        elif cv <= 12: thr -= 1
        good = [c for c in available_cards if c.value <= thr]
        if good:
            best = min(good, key=lambda x: x.value)
            return True, best
        return False, None

    # ── jhyap call logic ─────────────────────────────────────────────────────
    def should_call_jhyap(self, hand, game_state, game):
        hv = sum(c.value for c in hand)
        if hv > JHYAP_THRESHOLD: return False
        if self.style == AIStyle.AGGRESSIVE:
            return hv <= JHYAP_THRESHOLD
        elif self.style == AIStyle.CONSERVATIVE:
            return hv <= AI_CONSERVATIVE_CALL
        elif self.style == AIStyle.BALANCED:
            if hv <= 5: return True
            elif hv <= AI_BALANCED_CALL_1: return random.random() < AI_BALANCED_PROB_1
            elif hv <= AI_BALANCED_CALL_2: return random.random() < AI_BALANCED_PROB_2
            return False
        elif self.style == AIStyle.OPPORTUNISTIC:
            my_c  = game.player_coins[self.player_id]
            avg_c = sum(game.player_coins) / game.num_players
            if my_c > avg_c:
                self.strategy_params["risk_adjustment"] = 1.2
                self.strategy_params["jhyap_prob_base"] = 0.8
            else:
                self.strategy_params["risk_adjustment"] = 0.8
                self.strategy_params["jhyap_prob_base"] = 0.3
            if hv <= self.strategy_params["jhyap_threshold_strict"]: return True
            if hv <= self.strategy_params["jhyap_threshold_risky"]:
                prob = self.strategy_params["jhyap_prob_base"]
                prob += (JHYAP_THRESHOLD - hv) * 0.05
                if game_state.turn_count > 20: prob += 0.15
                return random.random() < prob
        return False


# ─── ROUND & GAME SIMULATION ─────────────────────────────────────────────────

def _end_round(game, hands, caller, turns_played):
    hv   = [game.calculate_hand_value(h) for h in hands]
    mv   = min(hv)
    noncaller_min = [i for i, v in enumerate(hv) if v == mv and i != caller]
    winner = noncaller_min[0] if noncaller_min else caller
    successful = (winner == caller)
    cc = [0] * game.num_players
    if successful:
        for i in range(game.num_players):
            if i != caller:
                pmt = min(hv[i], MAX_PAYMENT)
                game.player_coins[i] -= pmt
                game.player_coins[caller] += pmt
                cc[i] = -pmt; cc[caller] += pmt
    else:
        total = sum(min(v, MAX_PAYMENT) for v in hv)
        game.player_coins[caller] -= total
        game.player_coins[winner] += total
        cc[caller] = -total; cc[winner] = total
    result = RoundResult(
        round_number=game.round_number, caller=caller, winner=winner,
        hand_values=hv, coin_changes=cc,
        final_coins=game.player_coins.copy(),
        turns_played=turns_played, successful_call=successful,
        hands=[h[:] for h in hands],
    )
    game.game_history.append(result)
    return result


def simulate_round(game, ai_players):
    game.round_number += 1
    hands, deck = game.deal_cards()
    discard_pile = [deck.pop()] if deck else []
    gs = GameState(round_number=game.round_number, current_player=0,
                   hands=[h[:] for h in hands], discard_pile=discard_pile,
                   deck_size=len(deck), player_coins=game.player_coins.copy(),
                   turn_count=0)
    cur = 0
    while gs.turn_count < MAX_TURNS:
        ai   = ai_players[cur]
        hand = hands[cur]
        gs.current_player = cur
        gs.deck_size      = len(deck)
        gs.hands          = [h[:] for h in hands]
        gs.turn_count    += 1

        if game.can_call_jhyap(hand) and ai.should_call_jhyap(hand, gs, game):
            return _end_round(game, hands, cur, gs.turn_count)

        disc = ai.choose_discard(hand, gs, game)
        if not game.validate_discard(disc) and hand:
            disc = [max(hand, key=lambda x: x.value)]
        for c in disc:
            if c in hand: hand.remove(c)
        discard_pile.extend(disc)
        for other in ai_players:
            if other.player_id != cur: other.cards_seen.extend(disc)

        if not hand:
            return _end_round(game, hands, cur, gs.turn_count)

        top = discard_pile[-1] if discard_pile else None
        pick, _ = ai.should_pick_from_discard([top] if top else [], hand, gs)
        if pick and top:
            hand.append(discard_pile.pop())
        else:
            if not deck and len(discard_pile) > 1:
                t = discard_pile.pop()
                random.shuffle(discard_pile)
                deck.extend(discard_pile[:])
                discard_pile[:] = [t]
            elif not deck and len(discard_pile) <= 1:
                hv     = [game.calculate_hand_value(h) for h in hands]
                caller = hv.index(min(hv))
                return _end_round(game, hands, caller, gs.turn_count)
            if deck:
                hand.append(deck.pop())
        cur = (cur + 1) % game.num_players

    hv = [game.calculate_hand_value(h) for h in hands]
    return _end_round(game, hands, hv.index(min(hv)), gs.turn_count)


def run_tournament(ai_players, n_rounds=N_ROUNDS, seed=42):
    """Run n_rounds with the given ai_players list. Returns per-round win flags for player 0."""
    random.seed(seed)
    np.random.seed(seed)
    game       = DhumbalGame(num_players=len(ai_players))
    win_flags  = []
    for _ in range(n_rounds):
        result = simulate_round(game, ai_players)
        win_flags.append(1 if result.winner == 0 else 0)
    return win_flags


# ─── AGGRESSIVE AGENT RULE VARIANTS ──────────────────────────────────────────

def make_baseline_aggressive(pid=0):
    """Full Aggressive agent (all rules active)."""
    return RuleBasedAI(pid, AIStyle.AGGRESSIVE, "Aggressive_BASELINE")


def make_ablated(pid, ablated_rule_label, ablated_params):
    """Aggressive agent with one rule neutralised."""
    agent = RuleBasedAI(pid, AIStyle.AGGRESSIVE, f"Aggressive_{ablated_rule_label}")
    agent.strategy_params.update(ablated_params)
    return agent


def make_opponents(start_pid=1):
    return [
        RuleBasedAI(start_pid,   AIStyle.CONSERVATIVE,  "Conservative"),
        RuleBasedAI(start_pid+1, AIStyle.BALANCED,       "Balanced"),
        RuleBasedAI(start_pid+2, AIStyle.OPPORTUNISTIC,  "Opportunistic"),
    ]


# ─── ABLATION DEFINITIONS ────────────────────────────────────────────────────
#
# Each tuple: (label, description, param_dict_to_override)
# We "neutralise" each rule by reverting it to the *base* (neutral) default.
# ─────────────────────────────────────────────────────────────────────────────

ABLATIONS = [
    (
        "No_MaxDiscard_Pref",
        "discard_high_preference set to neutral (0.8 instead of 1.0)",
        {"discard_high_preference": 0.8},
    ),
    (
        "No_RiskBoost",
        "risk_adjustment set to neutral (1.0 instead of 1.2)",
        {"risk_adjustment": 1.0},
    ),
    (
        "No_AggressiveJhyap",
        "jhyap_threshold_strict lowered to Conservative level (7 instead of 10) — agent only calls Jhyap when hand ≤ 7",
        {"jhyap_threshold_strict": AI_CONSERVATIVE_CALL},   # 7
    ),
    (
        "No_HighJhyapProb",
        "jhyap_prob_base set to neutral (0.6 instead of 0.8)",
        {"jhyap_prob_base": 0.6},
    ),
    (
        "No_AggressivePickup",
        "pickup_threshold set to Conservative level (3 instead of 4)",
        {"pickup_threshold": 3},
    ),
]


def cohens_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2: return 0.0
    pool_std = np.sqrt(((len(a)-1)*np.var(a) + (len(b)-1)*np.var(b)) / (len(a)+len(b)-2))
    return (np.mean(a) - np.mean(b)) / pool_std if pool_std != 0 else 0.0


def ci95(flags):
    n  = len(flags)
    mu = np.mean(flags)
    se = np.std(flags) / np.sqrt(n)
    return mu * 100, 1.96 * se * 100


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  SENSITIVITY ANALYSIS — AGGRESSIVE AGENT RULE CONTRIBUTION")
    print(f"  Rounds per arm: {N_ROUNDS}   Seed: 42")
    print("=" * 68)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n[1/6] Running BASELINE (all Aggressive rules active) …")
    baseline_players = [make_baseline_aggressive(0)] + make_opponents(1)
    baseline_flags   = run_tournament(baseline_players, N_ROUNDS, seed=42)
    base_wr, base_ci = ci95(baseline_flags)
    print(f"      Baseline win rate: {base_wr:.2f}% ± {base_ci:.2f}%")

    # ── Ablations ─────────────────────────────────────────────────────────────
    rows = []
    for idx, (label, description, params) in enumerate(ABLATIONS, start=2):
        print(f"\n[{idx}/{len(ABLATIONS)+1}] Ablating rule: {label}")
        print(f"      ({description})")
        abl_players = [make_ablated(0, label, params)] + make_opponents(1)
        abl_flags   = run_tournament(abl_players, N_ROUNDS, seed=42)
        abl_wr, abl_ci = ci95(abl_flags)
        _, p_val = ttest_ind(baseline_flags, abl_flags, equal_var=False)
        d        = cohens_d(baseline_flags, abl_flags)
        drop     = base_wr - abl_wr          # positive = rule helps
        pct_drop = (drop / base_wr * 100) if base_wr > 0 else 0
        sig      = "significant" if p_val < 0.05 else "not significant"
        print(f"      Ablated win rate : {abl_wr:.2f}% ± {abl_ci:.2f}%")
        print(f"      Win-rate drop    : {drop:+.2f} pp  ({pct_drop:+.1f}%)")
        print(f"      Cohen's d        : {d:.4f}")
        print(f"      p-value          : {p_val:.4e}  ({sig})")
        rows.append({
            "Rule_Ablated":        label,
            "Description":         description,
            "Baseline_WR_%":       round(base_wr,  3),
            "Baseline_CI_±%":      round(base_ci,  3),
            "Ablated_WR_%":        round(abl_wr,   3),
            "Ablated_CI_±%":       round(abl_ci,   3),
            "WR_Drop_pp":          round(drop,      3),
            "WR_Drop_%_relative":  round(pct_drop,  2),
            "Cohens_d":            round(d,          4),
            "p_value":             round(p_val,      6),
            "Significant_p<0.05":  "Yes" if p_val < 0.05 else "No",
        })

    # ── Sort by contribution (largest drop = most important rule) ─────────────
    rows.sort(key=lambda r: r["WR_Drop_pp"], reverse=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = os.path.join(os.path.dirname(__file__), "aggressive_sensitivity_analysis.csv")
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(os.path.dirname(__file__), "aggressive_sensitivity_analysis.json")
    summary = {
        "experiment": "Aggressive Agent Rule Sensitivity Analysis",
        "rounds_per_arm":  N_ROUNDS,
        "seed":            42,
        "n_ablation_arms": len(ABLATIONS),
        "opponents":       ["Conservative", "Balanced", "Opportunistic"],
        "baseline": {
            "label":     "Aggressive_BASELINE",
            "win_rate%": round(base_wr, 3),
            "ci_95%":    round(base_ci, 3),
        },
        "ablation_results": rows,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)

    # ── Print ranked summary ──────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  RANKED RULE CONTRIBUTIONS  (↓ drop = rule matters more)")
    print("=" * 68)
    print(f"{'Rank':<5} {'Rule':<28} {'Drop (pp)':<11} {'Cohen d':<10} {'Sig?'}")
    print("-" * 68)
    for rank, r in enumerate(rows, 1):
        sig = "Yes" if r["Significant_p<0.05"] == "Yes" else "No"
        print(f"{rank:<5} {r['Rule_Ablated']:<28} {r['WR_Drop_pp']:>+.3f} pp   "
              f"{r['Cohens_d']:>+.4f}   {sig}")
    print("-" * 68)
    print(f"\nResults saved to:\n   {out_csv}\n   {out_json}")


if __name__ == "__main__":
    main()
